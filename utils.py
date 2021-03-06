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
"""utils.py: Various utility classes and functions used throughout the whole project.

"""
import collections
import time

import tensorflow as tf
import tensorflow_federated as tff
import transformers
import attr

import huggingface_keras_layers


ModelWeights = collections.namedtuple('ModelWeights', 'trainable non_trainable')
ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')


class KerasModelWrapper(object):
    """A standalone keras wrapper to be used in TFF."""

    def __init__(self, keras_model, input_spec, loss,
                 tf_device_identifier=None):
        """A wrapper class that provides necessary API handles for TFF.

        Args:
          keras_model: A `tf.keras.Model` to be trained.
          input_spec: Metadata of dataset that desribes the input tensors, which
            will be converted to `tff.Type` specifying the expected type of input
            and output of the model.
          loss: A `tf.keras.losses.Loss` instance to be used for training.
        """
        
        self.tf_device_identifier = tf_device_identifier
        
        if self.tf_device_identifier is not None:
            with tf.device(self.tf_device_identifier):
                self.keras_model = keras_model
                self.loss = loss
        else:
            self.keras_model = keras_model
            self.loss = loss

        self.input_spec = input_spec

    def forward_pass(self, batch_input, training=True):
        """Forward pass of the model to get loss for a batch of data.

        Args:
          batch_input: A tuple, the first element containing inputs and
            the second one for labels.
          training: Boolean scalar indicating training or inference mode.

        Returns:
          A scalar tf.float32 `tf.Tensor` loss for current batch input.
        """
        
        if self.tf_device_identifier is not None:

            with tf.device(self.tf_device_identifier):
                preds = self.keras_model(batch_input[0], training=training)

                loss = self.loss(batch_input[1], preds)

                return ModelOutputs(loss=loss)
        else:

            preds = self.keras_model(batch_input[0], training=training)

            loss = self.loss(batch_input[1], preds)

            return ModelOutputs(loss=loss)

    @property
    def weights(self):
        return ModelWeights(
            trainable=self.keras_model.trainable_variables,
            non_trainable=self.keras_model.non_trainable_variables)

    def from_weights(self, model_weights):
        # Update weights
        tff.utils.assign(
            self.keras_model.trainable_variables, 
            list(model_weights.trainable))

        tff.utils.assign(
            self.keras_model.non_trainable_variables,
            list(model_weights.non_trainable))


@attr.s(eq=False, frozen=False, slots=True)
class OptimizerOptions(object):

    init_lr = attr.ib(default=0.01)
    num_train_steps = attr.ib(default=10000)
    num_warmup_steps = attr.ib(default=500)
    min_lr_ratio = attr.ib(0.0)
    adam_beta1 = attr.ib(0.99)
    adam_beta2 = attr.ib(0.999)
    adam_epsilon = attr.ib(1e-7)
    weight_decay_rate = attr.ib(0.01)


class MaskedLMCrossEntropy(tf.keras.losses.Loss):

    def call(self, y_true, y_pred):
        return calculate_masked_lm_cross_entropy(y_true, y_pred)
        

class AdamWeightDecay(transformers.AdamWeightDecay):
    
    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(transformers.AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["weight_decay_rate"] = tf.identity(
            self.weight_decay_rate, name="adam_weight_decay_rate"
        )


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


def keras_evaluate(model, test_data, metric,
                   tf_device_identifier=None):
    metric.reset_states()

    if tf_device_identifier is not None:
        with tf.device(tf_device_identifier):
            for batch in test_data:
                preds = model(batch[0], training=False)
                metric.update_state(y_true=batch[1], y_pred=preds)
    else:
        for batch in test_data:
            preds = model(batch[0], training=False)
            metric.update_state(y_true=batch[1], y_pred=preds)

    return metric.result()


def calculate_masked_lm_cross_entropy(y_true, y_pred):

    # Need to filter out positions with the label '-100'
    masked_positions = tf.where(tf.not_equal(y_true, -100))

    y_true_reduced = tf.gather_nd(y_true, masked_positions)
    y_pred_reduced = tf.gather_nd(y_pred, masked_positions)

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_reduced, y_pred_reduced, from_logits=True)

    return loss

def convert_huggingface_mlm_to_keras(huggingface_model, max_seq_length,
                                     use_pretrained_mlm_weights=False):

    input_ids = tf.keras.Input(
        shape=[max_seq_length], dtype=tf.int32)

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
    
    if use_pretrained_mlm_weights:
        mlm_layer.set_weights(huggingface_model.layers[1].get_weights())

    return tf.keras.Model(input_ids, mlm_output)
