import tensorflow as tf


# Based on the answers from
# https://stackoverflow.com/questions/42334646/tensorflow-pad-unknown-size-tensor-to-a-specific-size/51936821#51936821
def dynamic_padding(inp, min_size, constant_values):

    pad_size = min_size - tf.shape(inp)[1]
    paddings = [[0,0], [0, pad_size]] # assign here, during graph execution

    return tf.cast(tf.pad(inp, paddings, constant_values=constant_values), dtype=tf.int32)
    
    
def get_masked_input_and_labels(
    inputs, vocab_table, special_ids_mask_table, mask_token_id,
    mlm_probability=0.15, **kwargs):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    Based on the codes from https://github.com/huggingface/transformers/blob/eb3bd73ce35bfef56eeb722d697f2d39a06a8f8d/src/transformers/data/data_collator.py#L270
    Adapted for our TensorFlow workloads.
    """
    
    labels = tf.identity(inputs)

    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = tf.fill(tf.shape(labels), mlm_probability)

    # We shouldn't mask out any special tokens
    if 'special_tokens_mask' not in kwargs:
        special_tokens_mask = tf.vectorized_map(special_ids_mask_table.lookup, tf.cast(labels, dtype=tf.int32))

    special_tokens_mask = tf.cast(special_tokens_mask, dtype=tf.bool)

    probability_matrix = tf.where(special_tokens_mask, x=0.0, y=probability_matrix)

    masked_indices = tf.compat.v1.distributions.Bernoulli(probs=probability_matrix).sample()
    masked_indices = tf.cast(masked_indices, dtype=tf.bool)
    
    labels = tf.where(~masked_indices, x=tf.constant(-100, dtype=tf.int32), y=labels)  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = tf.compat.v1.distributions.Bernoulli(probs=0.8).sample(sample_shape=tf.shape(labels))
    indices_replaced = tf.cast(indices_replaced, dtype=tf.bool)
    indices_replaced = indices_replaced & masked_indices

    inputs = tf.where(indices_replaced, x=mask_token_id, y=inputs)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = tf.compat.v1.distributions.Bernoulli(probs=0.5).sample(sample_shape=tf.shape(labels))
    indices_random = tf.cast(indices_random, dtype=tf.bool)
    indices_random = indices_random & masked_indices & ~indices_replaced

    random_words = tf.random.uniform(shape=tf.shape(labels), minval=0, maxval=tf.cast(vocab_table.size(), dtype=tf.int32), dtype=tf.int32)
    inputs = tf.where(indices_random, x=random_words, y=inputs)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


# New preprocessing steps based on TF.text tokenizer
def tokenize_and_mask(x):
    # TF.text tokenizer returns RaggedTensor. Convert this to a regular tensor.
    # Note: In the third dimension, 2nd and 3rd indexes contain some sort of offset information,
    # which we will ignore for now.
    tokenized = mobilebert_tokenizer_tf_text.tokenize(x).to_tensor()[:, :, 0]

    # Add special tokens: [CLS]
    cls_tensor_for_tokenized = tf.constant(mobilebert_tokenizer.cls_token_id, shape=[len(x), 1], dtype=tf.int32)
    tokenized_with_special_tokens = tf.concat([cls_tensor_for_tokenized, tokenized], axis=1)

    # Truncate if the sequence is already longer than BERT_MAX_SEQ_LENGTH
    tokenized_with_special_tokens = tf.cond(
        tf.greater_equal(tf.shape(tokenized_with_special_tokens)[1], BERT_MAX_SEQ_LENGTH),
        true_fn=lambda: tokenized_with_special_tokens[:, 0:BERT_MAX_SEQ_LENGTH-1],
        false_fn=lambda: tokenized_with_special_tokens)     

    # Add special tokens: [SEP]
    sep_tensor_for_tokenized = tf.constant(mobilebert_tokenizer.sep_token_id, shape=[len(x), 1], dtype=tf.int32)
    tokenized_with_special_tokens = tf.concat([tokenized_with_special_tokens, sep_tensor_for_tokenized], axis=1)

    # Padding with [PAD]
    # Final sequence should have the length of BERT_MAX_SEQ_LENGTH
    # Pad only if necessary
    tokenized_with_special_tokens = tf.cond(
        tf.less(tf.shape(tokenized_with_special_tokens)[1], BERT_MAX_SEQ_LENGTH),
        true_fn=lambda: dynamic_padding(tokenized_with_special_tokens, BERT_MAX_SEQ_LENGTH, mobilebert_tokenizer.pad_token_id),
        false_fn=lambda: tokenized_with_special_tokens)  

    tokenized_with_special_tokens = tf.cast(tokenized_with_special_tokens, dtype=tf.int32)

    # Random masking for the BERT MLM
    masked, labels = utils.get_masked_input_and_labels(
        tokenized_with_special_tokens,
        mobilebert_vocab_lookup_table,
        mobilebert_special_ids_mask_table,
        tf.constant(mobilebert_tokenizer.mask_token_id, dtype=tf.int32))

    # Squeeze out the first dimension
    masked = tf.squeeze(masked)
    labels = tf.squeeze(labels)

    # Manually settting the shape here so that TensorFlow graph
    # could know the sizes in advnace
    masked.set_shape(BERT_MAX_SEQ_LENGTH)
    labels.set_shape(BERT_MAX_SEQ_LENGTH)
    
    return masked, labels
