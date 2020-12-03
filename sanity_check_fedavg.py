# Copyright 2020, The TensorFlow Federated Authors.
# Copyright 2020, Ronald Seoh.
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
"""Simple FedAvg to train EMNIST.

This is the modified version of the script included
in the original simple_fedavg implementation from TFF.
A much smaller CNN model than BERT is used. We use this
to test out the changes in our version of FedAvg.
"""

import collections
import functools
import math

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import transformers

import fedavg
import fedavg_client
import utils

# Training hyperparameters
flags.DEFINE_integer('total_rounds', 256, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 2,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 100, 'Minibatch size of test data.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.1, 'Client learning rate.')

FLAGS = flags.FLAGS


def get_emnist_dataset():
    """Loads and preprocesses the EMNIST dataset.

    Returns:
      A `(emnist_train, emnist_test)` tuple where `emnist_train` is a
      `tff.simulation.ClientData` object representing the training data and
      `emnist_test` is a single `tf.data.Dataset` representing the test data of
      all clients.
    """
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        only_digits=True)

    def element_fn(element):
        return (tf.expand_dims(element['pixels'], -1), element['label'])

    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 418 for Federated EMNIST
        return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
            count=FLAGS.client_epochs_per_round).batch(
                FLAGS.batch_size, drop_remainder=False)

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(
            FLAGS.test_batch_size, drop_remainder=False)

    emnist_train = emnist_train.preprocess(preprocess_train_dataset)
    emnist_test = preprocess_test_dataset(
        emnist_test.create_tf_dataset_from_all_clients())

    return emnist_train, emnist_test


def create_original_fedavg_cnn_model(only_digits=True):
    """The CNN model used in https://arxiv.org/abs/1602.05629.

    This function is duplicated from research/optimization/emnist/models.py to
    make this example completely stand-alone.

    Args:
        only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only EMNIST dataset. If False, uses 62 outputs for the larger
        dataset.

    Returns:
        An uncompiled `tf.keras.Model`.
    """
    data_format = 'channels_last'
    input_shape = [28, 28, 1]

    max_pool = functools.partial(
        tf.keras.layers.MaxPooling2D,
        pool_size=(2, 2),
        padding='same',
        data_format=data_format)

    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=5,
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu)

    model = tf.keras.models.Sequential([
        conv2d(filters=32, input_shape=input_shape),
        max_pool(),
        conv2d(filters=64),
        max_pool(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10 if only_digits else 62),
        tf.keras.layers.Activation(tf.nn.softmax),
    ])

    return model


def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)

def client_optimizer_fn(optimizer_options=None):
          
    # Declare the optimizer object first
    optimizer = transformers.AdamWeightDecay(learning_rate=FLAGS.client_learning_rate)
        
    # Then start changing its parameters
    
    # Do something about the learning rate schedule here
    
    # Update other parameters
    optimizer.beta_1 = optimizer_options.adam_beta1
    optimizer.beta_2 = optimizer_options.adam_beta2
    optimizer.epsilon = optimizer_options.adam_epsilon
    #optimizer.weight_decay_rate = optimizer_options.weight_decay_rate
    
    return optimizer


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Train/test dataset
    train_data, test_data = get_emnist_dataset()
    

    def tff_model_fn():
        """Constructs a fully initialized model for use in federated averaging."""
        keras_model = create_original_fedavg_cnn_model(only_digits=True)

        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        return utils.KerasModelWrapper(keras_model, test_data.element_spec, loss)

    iterative_process = fedavg.build_federated_averaging_process(
        model_fn=tff_model_fn, 
        model_input_spec=test_data.element_spec,
        server_optimizer_fn=server_optimizer_fn, 
        client_optimizer_fn=client_optimizer_fn)

    server_state = iterative_process.initialize()
    
    client_states = {}
    
    # Initialize client states for all clients
    for i, client_id in enumerate(train_data.client_ids):
        client_optimizer_options = utils.OptimizerOptions(
            init_lr=0.01,
            num_train_steps=10000,
            num_warmup_steps=500,
            min_lr_ratio=0.0,
            adam_beta1=0.99,
            adam_beta2=0.999,
            adam_epsilon=1e-7,
            weight_decay_rate=0.01,
        )

        client_states[client_id] = fedavg_client.ClientState(
            client_serial=i,
            num_processed=0,
            optimizer_options=client_optimizer_options)

    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model = tff_model_fn()

    for round_num in range(FLAGS.total_rounds):

        sampled_client_serials = np.random.choice(
            len(train_data.client_ids),
            size=FLAGS.train_clients_per_round,
            replace=False)

        # Generate client datasets
        sampled_train_data = []
        sampled_client_states = []
        
        for client_serial in sampled_client_serials:
            client_data = train_data.create_tf_dataset_for_client(train_data.client_ids[client_serial])
            
            # Check the client lengths and put appropriate number of
            # training steps into OptimizerOptions
            # Apparently iterating through each of them is 
            # the only way to get the lengths of tf.data.Dataset
            # This is not very cool tbh.
            client_data_length = 0
            
            for _ in client_data:
                client_data_length = client_data_length + 1

            client_train_steps = math.ceil(client_data_length / FLAGS.batch_size)
            
            client_states[train_data.client_ids[client_serial]].optimizer_options.num_train_steps = client_train_steps

            sampled_train_data.append(client_data)
            sampled_client_states.append(client_states[train_data.client_ids[client_serial]])

        server_state, new_client_states, train_metrics = iterative_process.next(
            server_state, sampled_client_states, sampled_train_data)

        print(f'Round {round_num} training loss: {train_metrics}')
        print()
        
        # Update client states
        print("Updating client states.")

        for state in new_client_states:
            client_states[train_data.client_ids[state.client_serial]] = state

        print()

        if round_num % FLAGS.rounds_per_eval == 0:
            model.from_weights(server_state.model_weights)
            
            accuracy = utils.keras_evaluate(model.keras_model, test_data, metric)

            print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')
            print()


if __name__ == '__main__':
  app.run(main)
