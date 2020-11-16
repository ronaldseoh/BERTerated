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

import collections
import attr
import tensorflow as tf
import tensorflow_federated as tff


ModelWeights = collections.namedtuple('ModelWeights', 'trainable non_trainable')
ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')


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


@tf.function
def client_update(model, dataset, server_message, client_optimizer):
    """Performans client local training of `model` on `dataset`.

    Args:
      model: A `tff.learning.Model`.
      dataset: A 'tf.data.Dataset'.
      server_message: A `BroadcastMessage` from server.
      client_optimizer: A `tf.keras.optimizers.Optimizer`.

    Returns:
      A 'ClientOutput`.
    """
    model_weights = model.weights
    initial_weights = server_message.model_weights
    tff.utils.assign(model_weights, initial_weights)

    num_examples = tf.constant(0, dtype=tf.int32)
    loss_sum = tf.constant(0, dtype=tf.float32)
    # Explicit use `iter` for dataset is a trick that makes TFF more robust in
    # GPU simulation and slightly more performant in the unconventional usage
    # of large number of small datasets.
    for batch in iter(dataset):
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch)
        grads = tape.gradient(outputs.loss, model_weights.trainable)
        grads_and_vars = zip(grads, model_weights.trainable)
        client_optimizer.apply_gradients(grads_and_vars)
        batch_size = tf.shape(batch['x'])[0]
        num_examples += batch_size
        loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)

    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    client_weight = tf.cast(num_examples, tf.float32)
  
    return ClientOutput(weights_delta, client_weight, loss_sum / client_weight)
