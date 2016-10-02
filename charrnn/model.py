from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import data_reader


class Model():
    def __init__(self, batch_size, vocab_size, embedding_size, num_lstm_layers, labels):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_lstm_layers = num_lstm_layers
        self.lstm_size = 8
        self.nb_classes = len(labels)

    def init_graph(self):
        # Variable sequence length
        self.inputs = tf.placeholder(tf.int32, [self.batch_size, None])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, None])
        self.outputs = self.architecture(self.inputs)

    def architecture(self, inputs):
        embedding_weights = tf.Variable(
            tf.random_uniform((self.vocab_size, self.embedding_size), -1.0, 1.0) ,name='embedding_matrix')
        emb = tf.nn.embedding_lookup(embedding_weights, inputs, partition_strategy='mod', name=None, validate_indices=True)
        # Define a multilayer LSTM cell
        # https://www.tensorflow.org/versions/r0.10/tutorials/recurrent/index.html
        # https://www.tensorflow.org/versions/r0.10/api_docs/python/rnn_cell.html
        cell_list = [
            tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
            for _ in range(self.num_lstm_layers)]
        # Multilayer LSTM
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list, state_is_tuple=True)
        # Initial state of the LSTM memory.
        # Keep state in graph memory to use between batches
        # http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
        self.initial_state = multi_cell.zero_state(self.batch_size, tf.float32)
        print('initial_state: ', self.initial_state)
        # Convert to variables so that the state can be stored between batches
        self.state = tf.python.util.nest.pack_sequence_as(
            self.initial_state,
            [tf.Variable(var, trainable=False)
             for var in tf.python.util.nest.flatten(self.initial_state)])
        print('self.state: ', self.state)
        # Define the rnn through time
        lstm_output, new_state = tf.nn.dynamic_rnn(
            cell=multi_cell, inputs=emb, initial_state=self.state)
        # Force the initial state to be set to the new state for the next batch before returning the output
        store_states = [
            lstm_state.assign(new_lstm_state) for (lstm_state, new_lstm_state) in zip(
                tf.python.util.nest.flatten(self.state),
                tf.python.util.nest.flatten(new_state))]
        with tf.control_dependencies(store_states):
            lstm_output = tf.identity(lstm_output)
        # Define output layer
        logit_weights = tf.Variable(
            tf.truncated_normal((self.lstm_size, self.nb_classes), stddev=0.1), name='logit_weights')
        logit_bias = tf.Variable(tf.zeros((self.nb_classes)), name='logit_bias')
        # Reshape so that we can apply the linear transformation to all outputs
        output_flat = tf.reshape(lstm_output, (-1, self.lstm_size))
        # Apply last layer transformation
        self.logits_flat = tf.matmul(output_flat, logit_weights) + logit_bias
        probs_flat = tf.nn.softmax(self.logits_flat)
        self.probs = tf.reshape(probs_flat, (self.batch_size, -1, self.nb_classes))
        return self.probs

    def reset_state(self, sess):
        for s in tf.python.util.nest.flatten(self.state):
            sess.run(s.initializer)

    def init_train_op(self, optimizer):
        targets_flat = tf.reshape(self.targets, (-1, ))
        # Get the loss over all outputs
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.logits_flat, targets_flat, name='x_entropy')
        print('loss: ', loss.get_shape())
        self.loss = tf.reduce_mean(loss)
        print('loss: ', loss.get_shape())
        self.train_op = optimizer.minimize(self.loss)



def main():
    labels = data_reader.char_list
    print('labels: ', labels)

    num_steps = 3
    batch_size = 1
    vocab_size = 10
    embedding_size = 3
    sequence_length = 3
    num_lstm_layers = 1

    model = Model(batch_size, vocab_size, embedding_size, num_lstm_layers, labels)
    model.init_graph()
    optimizer = tf.train.AdamOptimizer(0.002)
    model.init_train_op(optimizer)

    _x = np.asarray([[3, 3, 1]])
    print('_x: ', _x.shape, _x)

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        [out, st] = sess.run(
            [model.outputs, model.state],
            feed_dict={model.inputs: _x})
        print('out: ', out.shape, out)
        print('st: ', st)
        loss, _ = sess.run(
            [model.loss, model.train_op],
            feed_dict={model.inputs: _x, model.targets: _x})
        print('loss: ', loss)

        # ist = sess.run(model.state)
        # print('ist: ', ist)
        # model.reset_state(sess)
        # ist = sess.run(model.state)
        # print('ist: ', ist)
        # [out, st] = sess.run(
        #     [model.outputs, model.state],
        #     feed_dict={model.inputs: _x})
        # print('out: ', out.shape, out)
        # print('st: ', st)
        # [out, st] = sess.run(
        #     [model.outputs, model.state],
        #     feed_dict={model.inputs: _x})
        # print('out: ', out.shape, out)
        # print('st: ', st)

if __name__ == "__main__":
    main()
