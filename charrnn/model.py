from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import data_reader


class Model():
    def __init__(self, batch_size, embedding_size, num_lstm_layers, labels, save_path):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_lstm_layers = num_lstm_layers
        self.lstm_size = 4
        self.labels = labels
        self.label_map = {val: idx for idx, val in enumerate(labels)}
        self.vocab_size = len(labels)
        self.save_path = save_path

    def init_graph(self):
        # Variable sequence length
        self.inputs = tf.placeholder(tf.int32, [None, None])
        self.targets = tf.placeholder(tf.int32, [None, None])
        self.sample_temperature = tf.placeholder_with_default(1.0, [])
        self.init_train_architecture()
        self.saver = tf.train.Saver(tf.trainable_variables())
        # self.sample_output = self.init_sample_architecture()

    def init_train_architecture(self):
        self.embedding_weights = tf.Variable(
            tf.truncated_normal((self.vocab_size, self.embedding_size), 0.01) ,name='embedding_matrix')
        self.embedding = tf.nn.embedding_lookup(
            self.embedding_weights, self.inputs, name='input_embedding')
        # Define a multilayer LSTM cell
        # https://www.tensorflow.org/versions/r0.10/tutorials/recurrent/index.html
        # https://www.tensorflow.org/versions/r0.10/api_docs/python/rnn_cell.html
        cell_list = [
            tf.nn.rnn_cell.LSTMCell(self.lstm_size, state_is_tuple=True, use_peepholes=True)
            for _ in range(self.num_lstm_layers)]
        # Multilayer LSTM
        self.multi_cell_lstm = tf.nn.rnn_cell.MultiRNNCell(cell_list, state_is_tuple=True)
        # Initial state of the LSTM memory.
        # Keep state in graph memory to use between batches
        # http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
        self.initial_state = self.multi_cell_lstm.zero_state(self.batch_size, tf.float32)
        print('initial_state: ', self.initial_state)
        # Convert to variables so that the state can be stored between batches
        self.state = tf.python.util.nest.pack_sequence_as(
            self.initial_state,
            [tf.Variable(var, trainable=False)
             for var in tf.python.util.nest.flatten(self.initial_state)])
        print('self.state: ', self.state)
        # Define the rnn through time
        lstm_output, new_state = tf.nn.dynamic_rnn(
            cell=self.multi_cell_lstm, inputs=self.embedding, initial_state=self.state)
        # Force the initial state to be set to the new state for the next batch before returning the output
        store_states = [
            lstm_state.assign(new_lstm_state) for (lstm_state, new_lstm_state) in zip(
                tf.python.util.nest.flatten(self.state),
                tf.python.util.nest.flatten(new_state))]
        with tf.control_dependencies(store_states):
            lstm_output = tf.identity(lstm_output)
        # Define output layer
        self.logit_weights = tf.Variable(
            tf.truncated_normal((self.lstm_size, self.vocab_size), stddev=0.01), name='logit_weights')
        self.logit_bias = tf.Variable(tf.zeros((self.vocab_size)), name='logit_bias')
        # Reshape so that we can apply the linear transformation to all outputs
        output_flat = tf.reshape(lstm_output, (-1, self.lstm_size))
        # Apply last layer transformation
        self.logits_flat = tf.matmul(output_flat, self.logit_weights) + self.logit_bias
        probs_flat = tf.nn.softmax(self.logits_flat)
        self.probs = tf.reshape(probs_flat, (self.batch_size, -1, self.vocab_size))
        # return self.probs

    def init_train_op(self, optimizer):
        targets_flat = tf.reshape(self.targets, (-1, ))
        # Get the loss over all outputs
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.logits_flat, targets_flat, name='x_entropy')
        print('loss: ', loss.get_shape())
        self.loss = tf.reduce_sum(loss)
        print('loss: ', loss.get_shape())
        self.train_op = optimizer.minimize(self.loss)

    # def init_sample_architecture(self):
    #     batch_size = 1
    #     char_embedding = self.embedding[:, 0, :]
    #     # Create an intial state with batch size 1
    #     initial_state = self.multi_cell_lstm.zero_state(batch_size, tf.float32)
    #     # Convert to variables so that the state can be stored between batches
    #     self.sample_state = tf.python.util.nest.pack_sequence_as(
    #         initial_state,
    #         [tf.Variable(var, trainable=False)
    #          for var in tf.python.util.nest.flatten(initial_state)])
    #     cell_output, new_state = self.multi_cell_lstm(char_embedding, self.sample_state)
    #     # Force the initial state to be set to the new state for the next batch before returning the output
    #     store_states = [
    #         lstm_state.assign(new_lstm_state) for (lstm_state, new_lstm_state) in zip(
    #             tf.python.util.nest.flatten(self.sample_state),
    #             tf.python.util.nest.flatten(new_state))]
    #     with tf.control_dependencies(store_states):
    #         cell_output = tf.identity(cell_output)
    #     print('cell_output: ', cell_output.get_shape())
    #     logits = tf.matmul(cell_output, self.logit_weights) + self.logit_bias
    #     print('logits: ', logits.get_shape())
    #     logits_temp = logits / self.sample_temperature
    #     print('logits_temp: ', logits_temp.get_shape())
    #     prob = tf.exp(logits_temp) / tf.reduce_sum(tf.exp(logits_temp))
    #     print('prob: ', prob.get_shape())
    #     # prob = tf.nn.softmax(logits)
    #     return prob

    def sample(self, session, prime, sample_length, temperature=1.0):
        self.reset_train_state(session)
        label_idx_list = range(self.vocab_size)
        # Prime state
        print('prime: ', prime)
        for char in prime:
            char_idx = self.label_map[char]
            out = session.run(self.probs,
                     feed_dict={self.inputs: np.asarray([[char_idx]])})
            # print('out: ', out, 'out[0,0]: ', out[0,0])
            sample_label = np.random.choice(label_idx_list, size=(1),  p=out[0,0])
            print('out: ', out, 'sample_label: ', sample_label)
        output_sample = prime
        print('start sampling')
        # Sample for sample_length steps
        for _ in range(sample_length):
            sample_label = np.random.choice(label_idx_list, size=(1),  p=out[0,0])
            print('out: ', out[0,0], 'sample_label: ', sample_label)
            # print('sample_label: ', sample_label)
            output_sample += self.labels[sample_label[0]]
            out = session.run(self.probs,
                     feed_dict={self.inputs: np.asarray([sample_label])})
            # print('s: ', s)
            # print('emb: ', emb)
        return output_sample

    # def sample(self, session, prime, sample_length, temperature=1.0):
    #     self.reset_sample_state(session)
    #     label_idx_list = range(self.vocab_size)
    #     # Prime state
    #     print('prime: ', prime)
    #     for char in prime:
    #         char_idx = self.label_map[char]
    #         out = session.run(self.sample_output,
    #                  feed_dict={self.inputs: np.asarray([[char_idx]]),
    #                             self.sample_temperature: temperature})
    #         sample_label = np.random.choice(label_idx_list, size=(1),  p=out[0])
    #         print('out: ', out, 'sample_label: ', sample_label)
    #     output_sample = prime
    #     print('start sampling')
    #     # print('label_idx_list: ', label_idx_list)
    #     # Sample for sample_length steps
    #     for _ in range(sample_length):
    #         sample_label = np.random.choice(label_idx_list, size=(1),  p=out[0])
    #         print('out: ', out, 'sample_label: ', sample_label)
    #         # print('sample_label: ', sample_label)
    #         output_sample += self.labels[sample_label[0]]
    #         out, s, emb = session.run([self.sample_output, self.sample_state, self.embedding],
    #                  feed_dict={self.inputs: np.asarray([sample_label]),
    #                             self.sample_temperature: temperature})
    #         # print('s: ', s)
    #         # print('emb: ', emb)
    #     return output_sample

    # def reset_sample_state(self, sess):
    #     for s in tf.python.util.nest.flatten(self.sample_state):
    #         sess.run(s.initializer)

    def reset_train_state(self, sess):
        for s in tf.python.util.nest.flatten(self.state):
            sess.run(s.initializer)

    def save(self, sess):
        self.saver.save(sess, self.save_path)

    def restore(self, sess):
        self.saver.restore(sess, self.save_path)



def main():
    labels = data_reader.char_list

    print('labels: ', labels)

    batch_size = 4
    embedding_size = 4
    num_lstm_layers = 1
    batch_len = 10

    batch_generator = data_reader.get_batch_generator(data_reader.data, batch_size, batch_len)

    save_path = './model.tf'
    model = Model(batch_size, embedding_size, num_lstm_layers, labels, save_path)
    model.init_graph()
    optimizer = tf.train.AdamOptimizer(0.1)
    model.init_train_op(optimizer)

    # _x = np.asarray([[2, 2, 1]])
    # print('_x: ', _x.shape, _x)

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        # [out, st] = sess.run(
        #     [model.outputs, model.state],
        #     feed_dict={model.inputs: _x})
        # print('out: ', out.shape, out)
        # print('st: ', st)
        input_batch, target_batch = next(batch_generator)
        print('input_batch: ', input_batch.shape, input_batch)
        print('target_batch: ', target_batch.shape, target_batch)
        for i in range(1000):
            # model.reset_train_state(sess)
            print('i: ', i)
            input_batch, target_batch = next(batch_generator)
            # print('input_batch: ', input_batch.shape, input_batch)
            # print('target_batch: ', target_batch.shape, target_batch)
            loss, _ = sess.run(
                [model.loss, model.train_op],
                feed_dict={model.inputs: input_batch, model.targets: target_batch})
            print('loss: ', loss)
        model.save(sess)

    tf.reset_default_graph()
    model = Model(1, embedding_size, num_lstm_layers, labels, save_path)
    model.init_graph()
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        model.restore(sess)
        sample = model.sample(sess, prime='abcabc', sample_length=10, temperature=0.1)
        print('sample: ', sample)

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
