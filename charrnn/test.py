from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf


num_steps = 3
batch_size = 1
vocab_size = 10
embedding_size = 3
sequence_length = 3
num_lstm_layers = 2
x_ids = tf.placeholder(tf.int32, [None, sequence_length])
W = tf.Variable(
    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0) ,name='embedding_matrix')
emb = tf.nn.embedding_lookup(W, x_ids, partition_strategy='mod', name=None, validate_indices=True)


lstm_size = 8
# https://www.tensorflow.org/versions/r0.10/tutorials/recurrent/index.html
# https://www.tensorflow.org/versions/r0.10/api_docs/python/rnn_cell.html
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
cell_list = [lstm_cell] * num_lstm_layers

# Multilayer LSTM
multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_lstm_layers, state_is_tuple=True)

# Initial state of the LSTM memory.
# Keep state in graph memory to use between batches
# http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
# (c_state_init, m_state_init) = multi_cell.zero_state(batch_size, tf.float32)
initial_state = multi_cell.zero_state(batch_size, tf.float32)
print('initial_state: ', initial_state)

# Convert to variables
initial_state = tf.python.util.nest.pack_sequence_as(
    initial_state,
    [tf.Variable(var, trainable=False) for var in tf.python.util.nest.flatten(initial_state)])
print('initial_state: ', initial_state)

# state = []
# # initial_state = lstm_cell.zero_state(batch_size, tf.float32)
#
# c_state_init = tf.placeholder_with_default(c_state_init, shape=(None, lstm_size))
# m_state_init = tf.placeholder_with_default(m_state_init, shape=(None, lstm_size))
# initial_state = tf.nn.rnn_cell.LSTMStateTuple(c=c_state_init, h=m_state_init)

# Define the rnn through time
# output, (c_state, m_state) = tf.nn.dynamic_rnn(cell=multi_cell, inputs=emb, initial_state=initial_state)
output, state = tf.nn.dynamic_rnn(cell=multi_cell, inputs=emb, initial_state=initial_state)
# (c_state, m_state) = state

# Force the initial state to be set to the new state for the next batch before returning the output
state_flatten = tf.python.util.nest.flatten(state)
initial_state_flatten = tf.python.util.nest.flatten(initial_state)
with tf.control_dependencies([init_state.assign(new_state)
                              for (init_state, new_state) in zip(initial_state_flatten, state_flatten)]):
    output = tf.identity(output)
# initial_state = state

def reset_init_state(sess):
    global initial_state
    [sess.run(s.initializer) for s in tf.python.util.nest.flatten(initial_state)]
    # [state.assign() for state in tf.python.util.nest.flatten(initial_state)]

# outputs = []
# state = (c_state_init, m_state_init)
# with tf.variable_scope('RNN'):
#     for step in range(num_steps):
#         if step > 0:
#             tf.get_variable_scope().reuse_variables()
#         cell_output, state = lstm_cell(emb[:, step, :], state)
#         outputs.append(cell_output)

# Concatenate list of outputs into tensor
#output = tf.reshape(tf.concat(1, outputs), [batch_size, -1, lstm_size])

_x = np.asarray([[3, 3, 1]])
print('_x: ', _x.shape, _x)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    [out, st] = sess.run([output, state] , feed_dict={x_ids: _x})
    print('out: ', out.shape, out)
    print('st: ', st)
    ist = sess.run(initial_state)
    print('ist: ', ist)
    reset_init_state(sess)
    ist = sess.run(initial_state)
    print('ist: ', ist)
    [out, st] = sess.run([output, state] , feed_dict={x_ids: _x})
    print('out: ', out.shape, out)
    print('st: ', st)
    [out, st] = sess.run([output, state] , feed_dict={x_ids: _x})
    print('out: ', out.shape, out)
    print('st: ', st)
