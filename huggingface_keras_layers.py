# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 Ronald Seoh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Backported Keras Layers from the HuggingFace library. 

TFF currently do not support Keras subclass models. Hence we take the individual layers within
the original Huggingface models and put them together again as a Keras functional model.
However, apart from the layers decorated with @keras_serializable (mostly the main layer part),
the head layers cannot be directly imported as they use their own config object to initialize
each layers.

Probably not the best solution, but for now we've decided to backport the relevant Keras layer
classes from Huggingface here and make them Keras serializable by directly feeding in relevant
parameters to the __init__().

"""
import tensorflow as tf
import transformers


# Uplfted from
# https://huggingface.co/transformers/_modules/transformers/modeling_tf_mobilebert.html#TFMobileBertPredictionHeadTransform
# Mainly done to avoid using `config` in __init__()
class StandaloneTFMobileBertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, 
                 hidden_size, hidden_act, initializer_range, layer_norm_eps, **kwargs):

        super().__init__(**kwargs)
        
        # Let's store all the inputs first
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.dense = tf.keras.layers.Dense(
            self.hidden_size,
            kernel_initializer=transformers.modeling_tf_utils.get_initializer(self.initializer_range),
            name="dense")

        if isinstance(hidden_act, str):
            self.transform_act_fn = transformers.activations_tf.get_tf_activation(self.hidden_act)
        else:
            self.transform_act_fn = self.hidden_act

        self.LayerNorm = transformers.modeling_tf_mobilebert.TFLayerNorm(
            self.hidden_size, epsilon=self.layer_norm_eps, name="LayerNorm")

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
        
    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "hidden_act" : self.hidden_act,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
        }


class StandaloneTFMobileBertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size, hidden_act, initializer_range, layer_norm_eps,
                 vocab_size, embedding_size, **kwargs):
        super().__init__(**kwargs)
        
        # Let's store all the inputs first
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        
        self.transform = StandaloneTFMobileBertPredictionHeadTransform(
            self.hidden_size, self.hidden_act, self.initializer_range, self.layer_norm_eps, name="transform")

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,), initializer="zeros", trainable=True, name="bias")

        self.dense = self.add_weight(
            shape=(self.hidden_size - self.embedding_size, self.vocab_size),
            initializer="zeros",
            trainable=True,
            name="dense/weight",
        )

        self.decoder = self.add_weight(
            shape=(self.vocab_size, self.embedding_size),
            initializer="zeros",
            trainable=True,
            name="decoder/weight",
        )

        super().build(input_shape)

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = tf.matmul(hidden_states, tf.concat([tf.transpose(self.decoder), self.dense], axis=0))
        hidden_states = hidden_states + self.bias
        return hidden_states
        
    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "hidden_act" : self.hidden_act,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
        }


class StandaloneTFMobileBertMLMHead(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size, hidden_act, initializer_range, layer_norm_eps,
                 vocab_size, embedding_size,**kwargs):

        super().__init__(**kwargs)
        
        # Let's store all the inputs first
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        
        self.predictions = StandaloneTFMobileBertLMPredictionHead(
            self.hidden_size, self.hidden_act, self.initializer_range, self.layer_norm_eps,
            self.vocab_size, self.embedding_size, name="predictions")

    def call(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "hidden_act" : self.hidden_act,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
        }

