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
import tensorflow as tf


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
      preds = self.keras_model(batch_input['x'], training=training)
      loss = self.loss(batch_input['y'], preds)
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
        preds = model(batch['x'], training=False)
        metric.update_state(y_true=batch['y'], y_pred=preds)

    return metric.result()


def get_masked_input_and_labels(
    inputs, vocab_table, special_ids_mask_table, mask_token_id,
    mlm_probability=0.15, **kwargs):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    Based on the codes from https://github.com/huggingface/transformers/blob/eb3bd73ce35bfef56eeb722d697f2d39a06a8f8d/src/transformers/data/data_collator.py#L270
    Adapted for our TensorFlow workloads.
    """
    
    def get_special_tokens_mask(val):
        return special_ids_mask_table.lookup(val)
    
    labels = tf.identity(inputs)

    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = tf.fill(tf.shape(labels), mlm_probability)

    # We shouldn't mask out any special tokens
    if 'special_tokens_mask' not in kwargs:
        special_tokens_mask = tf.vectorized_map(get_special_tokens_mask, tf.cast(labels, dtype=tf.int64))

    special_tokens_mask = tf.cast(special_tokens_mask, dtype=tf.bool)

    probability_matrix = tf.where(special_tokens_mask, x=0.0, y=probability_matrix)

    masked_indices = tf.compat.v1.distributions.Bernoulli(probs=probability_matrix).sample()
    masked_indices = tf.cast(masked_indices, dtype=tf.bool)
    
    labels = tf.where(~masked_indices, x=tf.constant(-100, dtype=tf.int64), y=labels)  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = tf.compat.v1.distributions.Bernoulli(probs=0.8).sample(sample_shape=tf.shape(labels))
    indices_replaced = tf.cast(indices_replaced, dtype=tf.bool)
    indices_replaced = indices_replaced & masked_indices

    inputs = tf.where(indices_replaced, x=mask_token_id, y=inputs)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = tf.compat.v1.distributions.Bernoulli(probs=0.5).sample(sample_shape=tf.shape(labels))
    indices_random = tf.cast(indices_random, dtype=tf.bool)
    indices_random = indices_random & masked_indices & ~indices_replaced

    random_words = tf.random.uniform(shape=tf.shape(labels), minval=0, maxval=vocab_table.size(), dtype=tf.int64)
    inputs = tf.where(indices_random, x=random_words, y=inputs)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
