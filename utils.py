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
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
