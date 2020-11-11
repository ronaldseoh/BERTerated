import transformers
import tensorflow as tf


def get_masked_input_and_labels(inputs, tokenizer, mlm_probability=0.15, **kwargs):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    Based on the codes from https://github.com/huggingface/transformers/blob/eb3bd73ce35bfef56eeb722d697f2d39a06a8f8d/src/transformers/data/data_collator.py#L270
    Adapted for TensorFlow.
    """
    labels = tf.identity(inputs)

    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = tf.fill(labels.shape, mlm_probability)

    # We shouldn't mask out any special tokens
    if 'special_tokens_mask' not in kwargs:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.numpy()
        ]

    special_tokens_mask = tf.cast(special_tokens_mask, dtype=tf.bool)

    probability_matrix = tf.where(special_tokens_mask, x=0.0, y=probability_matrix)

    masked_indices = tf.compat.v1.distributions.Bernoulli(probs=probability_matrix).sample()
    masked_indices = tf.cast(masked_indices, dtype=tf.bool)
    
    labels = tf.where(~masked_indices, x=-100, y=labels)  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = tf.compat.v1.distributions.Bernoulli(probs=0.8).sample(sample_shape=labels.get_shape())
    indices_replaced = tf.cast(indices_replaced, dtype=tf.bool)
    indices_replaced = indices_replaced & masked_indices

    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    inputs = tf.where(indices_replaced, x=mask_token_id, y=inputs)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = tf.compat.v1.distributions.Bernoulli(probs=0.5).sample(sample_shape=labels.get_shape())
    indices_random = tf.cast(indices_random, dtype=tf.bool)
    indices_random = indices_random & masked_indices & ~indices_replaced

    random_words = tf.random.uniform(shape=labels.get_shape(), minval=0, maxval=len(tokenizer), dtype=tf.int32)
    inputs = tf.where(indices_random, x=random_words, y=inputs)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
