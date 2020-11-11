import transformers
import tensorflow as tf


def get_masked_input_and_labels(encoded_texts, tokenizer):
    
    collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    inputs, labels = collator.mask_tokens(encoded_texts)
    
    inputs = tf.convert_to_tensor(inputs.numpy())
    labels = tf.convert_to_tensor(labels.numpy())
    
    return inputs, labels
