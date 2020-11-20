import tensorflow as tf
import tensorflow_text as tf_text


def convert_huggingface_tokenizer(huggingface_tokenizer,
                                  suffix_indicator="##",
                                  max_chars_per_token=None,
                                  split_unknown_characters=True,
                                  lower_case=True,
                                  keep_whitespace=False,
                                  normalization_form=None,
                                  preserve_unused_token=True,
                                  dtype=tf.int32):

    vocab_lookup_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=list(huggingface_tokenizer.vocab.keys()),
            values=tf.constant(list(huggingface_tokenizer.vocab.values()), dtype=tf.int64)),
        default_value=0)

    special_ids_mask_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(huggingface_tokenizer.all_special_ids, dtype=dtype),
            values=tf.constant(1, dtype=dtype, shape=len(huggingface_tokenizer.all_special_ids)),
            key_dtype=dtype, value_dtype=dtype),
        default_value=tf.constant(0, dtype=dtype))

    tokenizer_tf_text = tf_text.BertTokenizer(
        vocab_lookup_table=vocab_lookup_table,
        suffix_indicator="##",
        max_bytes_per_word=huggingface_tokenizer.wordpiece_tokenizer.max_input_chars_per_word,
        max_chars_per_token=None,
        token_out_type=dtype,
        unknown_token=huggingface_tokenizer.unk_token,
        split_unknown_characters=True,
        lower_case=True,
        keep_whitespace=False,
        normalization_form=None,
        preserve_unused_token=True)
        
    return tokenizer_tf_text, vocab_lookup_table, special_ids_mask_table


# Based on the answers from
# https://stackoverflow.com/questions/42334646/tensorflow-pad-unknown-size-tensor-to-a-specific-size/51936821#51936821
def dynamic_padding(inp, min_size, constant_values, dtype=tf.int32):

    pad_size = min_size - tf.shape(inp)[1]
    paddings = [[0,0], [0, pad_size]] # assign here, during graph execution

    return tf.cast(tf.pad(inp, paddings, constant_values=constant_values), dtype=dtype)
    
    
def get_masked_input_and_labels(inputs,
                                vocab_table, special_ids_mask_table, mask_token_id,
                                mlm_probability=0.15, dtype=tf.int32, **kwargs):
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
        special_tokens_mask = tf.map_fn(special_ids_mask_table.lookup, tf.cast(labels, dtype=dtype))

    special_tokens_mask = tf.cast(special_tokens_mask, dtype=tf.bool)

    probability_matrix = tf.where(special_tokens_mask, x=0.0, y=probability_matrix)

    masked_indices = tf.compat.v1.distributions.Bernoulli(probs=probability_matrix).sample()
    masked_indices = tf.cast(masked_indices, dtype=tf.bool)
    
    labels = tf.where(~masked_indices, x=tf.constant(-100, dtype=dtype), y=labels)  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = tf.compat.v1.distributions.Bernoulli(probs=0.8).sample(sample_shape=tf.shape(labels))
    indices_replaced = tf.cast(indices_replaced, dtype=tf.bool)
    indices_replaced = indices_replaced & masked_indices

    inputs = tf.where(indices_replaced, x=mask_token_id, y=inputs)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = tf.compat.v1.distributions.Bernoulli(probs=0.5).sample(sample_shape=tf.shape(labels))
    indices_random = tf.cast(indices_random, dtype=tf.bool)
    indices_random = indices_random & masked_indices & ~indices_replaced

    random_words = tf.random.uniform(shape=tf.shape(labels), minval=0, maxval=tf.cast(vocab_table.size(), dtype=dtype), dtype=dtype)
    inputs = tf.where(indices_random, x=random_words, y=inputs)
    
    # Sample weights: Completely ignore the data points with the label '-100'
    sample_weights = tf.identity(labels)
    sample_weights = tf.where(
        labels==-100, x=tf.constant(0, dtype=dtype), y=tf.constant(1, dtype=dtype))

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, sample_weights


# New preprocessing steps based on TF.text tokenizer
def tokenize_and_mask(x, max_seq_length,
                      bert_tokenizer_tf_text, vocab_lookup_table, special_ids_mask_table,
                      cls_token_id, sep_token_id, pad_token_id, mask_token_id,
                      dtype=tf.int32):
    # TF.text tokenizer returns RaggedTensor. Convert this to a regular tensor.
    # Note: In the third dimension, 2nd and 3rd indexes contain some sort of offset information,
    # which we will ignore for now.
    tokenized = bert_tokenizer_tf_text.tokenize(x).to_tensor()[:, :, 0]

    # Add special tokens: [CLS]
    cls_tensor_for_tokenized = tf.constant(cls_token_id, shape=[len(x), 1], dtype=dtype)
    tokenized_with_special_tokens = tf.concat([cls_tensor_for_tokenized, tokenized], axis=1)

    # Truncate if the sequence is already longer than max_seq_length
    tokenized_with_special_tokens = tf.cond(
        tf.greater_equal(tf.shape(tokenized_with_special_tokens)[1], max_seq_length),
        true_fn=lambda: tokenized_with_special_tokens[:, 0:max_seq_length-1],
        false_fn=lambda: tokenized_with_special_tokens)

    # Add special tokens: [SEP]
    sep_tensor_for_tokenized = tf.constant(sep_token_id, shape=[len(x), 1], dtype=dtype)
    tokenized_with_special_tokens = tf.concat([tokenized_with_special_tokens, sep_tensor_for_tokenized], axis=1)

    # Padding with [PAD]
    # Final sequence should have the length of max_seq_length
    # Pad only if necessary
    tokenized_with_special_tokens = tf.cond(
        tf.less(tf.shape(tokenized_with_special_tokens)[1], max_seq_length),
        true_fn=lambda: dynamic_padding(tokenized_with_special_tokens, max_seq_length, pad_token_id),
        false_fn=lambda: tokenized_with_special_tokens)

    tokenized_with_special_tokens = tf.cast(tokenized_with_special_tokens, dtype=dtype)

    # Random masking for the BERT MLM
    masked, labels, sample_weights = get_masked_input_and_labels(
        tokenized_with_special_tokens,
        vocab_lookup_table, 
        special_ids_mask_table,
        tf.constant(mask_token_id, dtype=tf.int32))

    # Squeeze out the first dimension
    #masked = tf.squeeze(masked)
    #labels = tf.squeeze(labels)
    #sample_weights = tf.squeeze(sample_weights)

    # Manually settting the shape here so that TensorFlow graph
    # could know the sizes in advnace
    masked.set_shape([None, max_seq_length])
    labels.set_shape([None, max_seq_length])
    sample_weights.set_shape([None, max_seq_length])
    
    return masked, labels, sample_weights
