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

This is intended to be a minimal stand-alone implementation of Federated
Averaging, suitable for branching as a starting point for algorithm
modifications; see `tff.learning.build_federated_averaging_process` for a
more full-featured implementation.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import tensorflow as tf
import tensorflow_federated as tff

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
def server_update(model, server_optimizer, server_state, weights_delta):
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
    server_optimizer.apply_gradients(grads_and_vars, name='server_update')

    # Create a new state based on the updated model.
    return tff.utils.update_state(
        server_state,
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=server_state.round_num + 1)


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


def build_federated_averaging_process(
    model_fn,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)):
    """Builds the TFF computations for optimization using federated averaging.

    Args:
      model_fn: A no-arg function that returns a
        `simple_fedavg_tf.KerasModelWrapper`.
      server_optimizer_fn: A no-arg function that returns a
        `tf.keras.optimizers.Optimizer` for server update.
      client_optimizer_fn: A no-arg function that returns a
        `tf.keras.optimizers.Optimizer` for client update.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    dummy_model = model_fn()

    @tff.tf_computation
    def server_init_tf():
        model = model_fn()
        server_optimizer = server_optimizer_fn()
        utils.initialize_optimizer_vars(model, server_optimizer)
        return ServerState(
            model_weights=model.weights,
            optimizer_state=server_optimizer.variables(),
            round_num=0)

    server_state_type = server_init_tf.type_signature.result

    model_weights_type = server_state_type.model_weights

    @tff.tf_computation(server_state_type, model_weights_type.trainable)
    def server_update_fn(server_state, model_delta):
        model = model_fn()
        server_optimizer = server_optimizer_fn()
        utils.initialize_optimizer_vars(model, server_optimizer)
        return server_update(model, server_optimizer, server_state, model_delta)

    @tff.tf_computation(server_state_type)
    def server_message_fn(server_state):
        return build_server_broadcast_message(server_state)

    server_message_type = server_message_fn.type_signature.result
    tf_dataset_type = tff.SequenceType(dummy_model.input_spec)

    @tff.tf_computation(tf_dataset_type, server_message_type)
    def client_update_fn(tf_dataset, server_message):
        model = model_fn()
        client_optimizer = client_optimizer_fn()
        return fedavg_client.client_update(model, tf_dataset, server_message, client_optimizer)

    federated_server_state_type = tff.type_at_server(server_state_type)
    federated_dataset_type = tff.type_at_clients(tf_dataset_type)

    @tff.federated_computation(federated_server_state_type,
                               federated_dataset_type)
    def run_one_round(server_state, federated_dataset):
        """Orchestration logic for one round of computation.

        Args:
        server_state: A `ServerState`.
        federated_dataset: A federated `tf.data.Dataset` with placement
            `tff.CLIENTS`.

        Returns:
        A tuple of updated `ServerState` and `tf.Tensor` of average loss.
        """
        server_message = tff.federated_map(server_message_fn, server_state)
        server_message_at_client = tff.federated_broadcast(server_message)

        client_outputs = tff.federated_map(
            client_update_fn, (federated_dataset, server_message_at_client))

        weight_denom = client_outputs.client_weight
        round_model_delta = tff.federated_mean(
            client_outputs.weights_delta, weight=weight_denom)

        server_state = tff.federated_map(server_update_fn,
                                        (server_state, round_model_delta))
        round_loss_metric = tff.federated_mean(
            client_outputs.model_output, weight=weight_denom)

        return server_state, round_loss_metric

    @tff.federated_computation
    def server_init_tff():
        """Orchestration logic for server model initialization."""
        return tff.federated_value(server_init_tf(), tff.SERVER)

    return tff.templates.IterativeProcess(
        initialize_fn=server_init_tff, next_fn=run_one_round)
