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
import sys

import tensorflow as tf
import tensorflow_federated as tff
import attr
import numpy as np


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput(object):
    """Structure for outputs returned from clients during federated optimization.

    Fields:
    -   `weights_delta`: A dictionary of updates to the model's trainable
      variables.
    -   `client_weight`: Weight to be used in a weighted mean when
      aggregating `weights_delta`.
    -   `model_output`: A structure matching
      `tff.learning.Model.report_local_outputs`, reflecting the results of
      training on the input dataset.
    """
    weights_delta = attr.ib()
    client_weight = attr.ib()
    model_output = attr.ib()


@attr.s
class ClientState(object):
    
    client_serial = attr.ib()
    visit_count = attr.ib()


@tf.function
def update_client(model, dataset, server_message, client_state, client_optimizer):
    """Performans client local training of `model` on `dataset`.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      server_message: A `BroadcastMessage` from server.
      client_optimizer: A `tf.keras.optimizers.Optimizer`.

    Returns:
      A 'ClientOutput`.
    """

    # Apply the new version of model from server
    tf.print("Anonymous client", client_state.client_serial, ": updated the model with server message.")

    tff.utils.assign(model.weights, server_message.model_weights)

    # Start the training on the client side.
    tf.print("Anonymous client", client_state.client_serial, ": training start.")

    # Total number of data points processed by
    # this client's optimizer
    num_examples = tf.constant(0, dtype=tf.int32)

    loss_sum = tf.constant(0, dtype=tf.float32)
    
    batch_count = tf.constant(0, dtype=tf.float32)

    # Explicit use `iter` for dataset is a trick that makes TFF more robust in
    # GPU simulation and slightly more performant in the unconventional usage
    # of large number of small datasets.
    for batch in iter(dataset):

        batch_count += 1

        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch)

        grads = tape.gradient(outputs.loss, model.weights.trainable)
        grads_and_vars = zip(grads, model.weights.trainable)

        client_optimizer.apply_gradients(grads_and_vars)

        batch_size = tf.shape(batch[0])[0]

        num_examples += batch_size
        
        tf.print("Anonymous client", client_state.client_serial, ": batch", batch_count, ",", num_examples, "examples processed")

        loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)
 
    # Divided the update by the batch size
    client_weight = tf.cast(num_examples, tf.float32)

    tf.print("Anonymous client", client_state.client_serial, ": training finished.", num_examples, " examples processed, loss:", loss_sum / client_weight)

    # Compare the weight values with the one from server message
    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model.weights.trainable,
                                          server_message.model_weights.trainable)
    
    # ClientState update
    new_client_state = ClientState(client_serial=client_state.client_serial, visit_count=client_state.visit_count + 1)

    if num_examples == 0:
        # Don't divide by 0
        return ClientOutput(weights_delta, client_weight, loss_sum), new_client_state
    else:
        return ClientOutput(weights_delta, client_weight, loss_sum / client_weight), new_client_state
