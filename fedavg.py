# Copyright 2020, The TensorFlow Federated Authors.
# Copyright 2020, Ronald Seoh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An implementation of the Federated Averaging algorithm.

This code is largely based on the `simple_fedavg` implementation from TensorFlow Federated,
although with slightly different code organization and additional functionalities for experimentation.

fedavg.py: Logics for the server side of the Federated Averaging algorithm. See fedavg_client.py for
the ciient-side logic.
"""
import tensorflow as tf
import tensorflow_federated as tff
import attr
import numpy as np

import fedavg_client
import utils

@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
    """Structure for state on the server.

    Fields:
    -   `model_weights`: A dictionary of model's trainable variables.
    -   `optimizer_state`: Variables of optimizer.
    -   'round_num': Current round index
    """
    model_weights = attr.ib()
    optimizer_state = attr.ib()
    round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
    """Structure for tensors broadcasted by server during federated optimization.

    Fields:
    -   `model_weights`: A dictionary of model's trainable tensors.
    -   `round_num`: Round index to broadcast. We use `round_num` as an example to
          show how to broadcast auxiliary information that can be helpful on
          clients. It is not explicitly used, but can be applied to enable
          learning rate scheduling.
    """
    model_weights = attr.ib()
    round_num = attr.ib()


@tf.function
def build_server_broadcast_message(server_state):
    """Builds `BroadcastMessage` for broadcasting.

    This method can be used to post-process `ServerState` before broadcasting.
    For example, perform model compression on `ServerState` to obtain a compressed
    state that is sent in a `BroadcastMessage`.

    Args:
      server_state: A `ServerState`.

    Returns:
      A `BroadcastMessage`.
    """
    return BroadcastMessage(
        model_weights=server_state.model_weights,
        round_num=server_state.round_num)


@tf.function
def update_server(model, server_optimizer, server_state, weights_delta):
    """Updates `server_state` based on `weights_delta`.

    Args:
      model: A `KerasModelWrapper` or `tff.learning.Model`.
      server_optimizer: A `tf.keras.optimizers.Optimizer`. If the optimizer
        creates variables, they must have already been created.
      server_state: A `ServerState`, the state to be updated.
      weights_delta: A nested structure of tensors holding the updates to the
        trainable variables of the model.

    Returns:
      An updated `ServerState`.
    """
    # Initialize the model with the current state.
    model_weights = model.weights
    tff.utils.assign(model_weights, server_state.model_weights)
    tff.utils.assign(server_optimizer.variables(), server_state.optimizer_state)

    # Apply the update to the model.
    grads_and_vars = tf.nest.map_structure(
        lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
        tf.nest.flatten(model_weights.trainable))

    server_optimizer.apply_gradients(grads_and_vars, name='update_server')

    # Create a new state based on the updated model.
    return tff.utils.update_state(
        server_state,
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=server_state.round_num + 1)


def build_federated_averaging_process(
    model_fn,
    model_input_spec,
    initial_trainable_weights=None,
    initial_non_trainable_weights=None,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)):
    """Builds the TFF computations for optimization using federated averaging.

    Args:
      model_fn: A no-arg function that returns a
        `utils.KerasModelWrapper`.
      server_optimizer_fn: A no-arg function that returns a
        `tf.keras.optimizers.Optimizer` for server update.
      client_optimizer_fn: A no-arg function that returns a
        `tf.keras.optimizers.Optimizer` for client update.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    @tff.tf_computation
    def server_init_tf():
        model = model_fn()

        server_optimizer = server_optimizer_fn()
        utils.initialize_optimizer_vars(model, server_optimizer)
            
        if (initial_trainable_weights is not None) \
        and (initial_non_trainable_weights is not None):
            initial_server_state = ServerState(
                model_weights=utils.ModelWeights(
                    trainable=initial_trainable_weights,
                    non_trainable=initial_non_trainable_weights),
                optimizer_state=server_optimizer.variables(),
                round_num=0)
        else:
            initial_server_state = ServerState(
                model_weights=model.weights,
                optimizer_state=server_optimizer.variables(),
                round_num=0)

        return initial_server_state


    # Type for server state
    server_state_type = server_init_tf.type_signature.result
    
    # Type for model weights
    model_weights_type = server_state_type.model_weights
    
    
    # Type for client states
    # Using this dummy function to obtain type signatures from it
    @tff.tf_computation
    def get_dummy_client_state():
        return fedavg_client.ClientState(client_serial=0, visit_count=0)

    client_state_type = get_dummy_client_state.type_signature.result


    # Server updating logic
    @tff.tf_computation(server_state_type, model_weights_type.trainable)
    def server_update_fn(server_state, model_delta):
        model = model_fn()
        server_optimizer = server_optimizer_fn()
        utils.initialize_optimizer_vars(model, server_optimizer)

        return update_server(model, server_optimizer, server_state, model_delta)


    # Server messages generation
    @tff.tf_computation(server_state_type)
    def server_message_fn(server_state):
        return build_server_broadcast_message(server_state)


    # Type for server messages (to clients)
    server_message_type = server_message_fn.type_signature.result

    # Type for TF datasets
    tf_dataset_type = tff.SequenceType(model_input_spec) # For individual clients
    federated_dataset_type = tff.type_at_clients(tf_dataset_type)


    # Client updating logic
    @tff.tf_computation(tf_dataset_type, server_message_type, client_state_type)
    def client_update_fn(tf_dataset, server_message, client_state):

        model = model_fn()
        client_optimizer = client_optimizer_fn()
        
        # Note: update_client() is a tf function
        return fedavg_client.update_client(
            model, tf_dataset, server_message, client_state, client_optimizer)

    
    # One round of FedAvg logic
    @tff.federated_computation(tff.type_at_server(server_state_type),
                               tff.type_at_clients(client_state_type),
                               federated_dataset_type)
    def run_one_round(server_state, client_states, federated_dataset):
        """Orchestration logic for one round of computation.

        Args:
        server_state: A `ServerState`.
        federated_dataset: A federated `tf.data.Dataset` with placement
            `tff.CLIENTS`.

        Returns:
        A tuple of updated `ServerState` and `tf.Tensor` of average loss.
        """
        # Prepare server_message to be sent to the clients,
        # based on the server_state from previous round
        server_message = tff.federated_map(server_message_fn, server_state)

        # Update the clients with the new server_message and dataset
        client_outputs, new_client_state = tff.federated_map(
            client_update_fn,
            (
                federated_dataset,
                tff.federated_broadcast(server_message),
                client_states,
            )
        )

        round_model_delta = tff.federated_mean(
            client_outputs.weights_delta, weight=client_outputs.client_weight)

        # Update server state given the current round's completion
        server_state = tff.federated_map(
            server_update_fn, (server_state, round_model_delta))

        round_loss_metric = tff.federated_mean(
            client_outputs.model_output, weight=client_outputs.client_weight)

        return server_state, new_client_state, round_loss_metric

    
    @tff.federated_computation
    def server_init_tff():
        """Orchestration logic for server model initialization."""
        return tff.federated_value(server_init_tf(), tff.SERVER)


    return tff.templates.IterativeProcess(
        initialize_fn=server_init_tff, next_fn=run_one_round)
