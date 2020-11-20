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
import collections
import time

import tensorflow as tf
import tensorflow_federated as tff

import huggingface_keras_layers


ModelWeights = collections.namedtuple('ModelWeights', 'trainable non_trainable')
ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')


class KerasModelWrapper(object):
    """A standalone keras wrapper to be used in TFF."""

    def __init__(self, keras_model, input_spec, loss):
      """A wrapper class that provides necessary API handles for TFF.

      Args:
        keras_model: A `tf.keras.Model` to be trained.
        input_spec: Metadata of dataset that desribes the input tensors, which
          will be converted to `tff.Type` specifying the expected type of input
          and output of the model.
        loss: A `tf.keras.losses.Loss` instance to be used for training.
      """
      self.keras_model = keras_model
      self.input_spec = input_spec
      self.loss = loss

    def forward_pass(self, batch_input, training=True):
      """Forward pass of the model to get loss for a batch of data.

      Args:
        batch_input: A `collections.abc.Mapping` with two keys, `x` for inputs and
          `y` for labels.
        training: Boolean scalar indicating training or inference mode.

      Returns:
        A scalar tf.float32 `tf.Tensor` loss for current batch input.
      """
      preds = self.keras_model(batch_input[0], training=training)
      loss = self.loss(batch_input[1], preds)
      return ModelOutputs(loss=loss)

    @property
    def weights(self):
        return ModelWeights(
            trainable=self.keras_model.trainable_variables,
            non_trainable=self.keras_model.non_trainable_variables)

    def from_weights(self, model_weights):
        tff.utils.assign(self.keras_model.trainable_variables, 
            list(model_weights.trainable))
        tff.utils.assign(self.keras_model.non_trainable_variables,
            list(model_weights.non_trainable))


def initialize_optimizer_vars(model, optimizer):
    """Creates optimizer variables to assign the optimizer's state."""
    model_weights = model.weights

    model_delta = tf.nest.map_structure(tf.zeros_like, model_weights.trainable)

    # Create zero gradients to force an update that doesn't modify.
    # Force eagerly constructing the optimizer variables. Normally Keras lazily
    # creates the variables on first usage of the optimizer. Optimizers such as
    # Adam, Adagrad, or using momentum need to create a new set of variables shape
    # like the model weights.
    grads_and_vars = tf.nest.map_structure(
        lambda x, v: (tf.zeros_like(x), v), tf.nest.flatten(model_delta),
        tf.nest.flatten(model_weights.trainable))

    optimizer.apply_gradients(grads_and_vars)

    assert optimizer.variables()


def keras_evaluate(model, test_data, metric):
    metric.reset_states()

    for batch in test_data:
        preds = model(batch[0], training=False)
        metric.update_state(y_true=batch[1], y_pred=preds)

    return metric.result()


def convert_huggingface_mlm_to_keras(huggingface_model, max_seq_length, batch_size):

    input_ids = tf.keras.Input(
        shape=[max_seq_length], batch_size=batch_size, dtype="int32")

    main_layer_output = huggingface_model.layers[0](input_ids)
    
    # Currently, the existing MLM output head in HuggingFace models
    # is not immediately Keras serializable.
    # So we try to create our own head here
    mlm_layer = huggingface_keras_layers.StandaloneTFMobileBertMLMHead(
        hidden_size=huggingface_model.config.hidden_size,
        hidden_act=huggingface_model.config.hidden_act,
        initializer_range=huggingface_model.config.initializer_range,
        layer_norm_eps=huggingface_model.config.layer_norm_eps,
        vocab_size=huggingface_model.config.vocab_size,
        embedding_size=huggingface_model.config.embedding_size,
    )
    
    mlm_output = mlm_layer(main_layer_output[0])

    mlm_layer.set_weights(huggingface_model.layers[1].get_weights())

    return tf.keras.Model(input_ids, mlm_output)

