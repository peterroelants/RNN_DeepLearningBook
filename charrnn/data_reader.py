from __future__ import print_function
from __future__ import division

import numpy as np

filepath = '/Users/peterroelants/Programming/DeepLearningBook/charrnn/tiny-shakespeare.txt'
# Read file
# with open(filepath, 'r') as f:
#     data = f.read()

data = '0123456789'

# print('data: ', data)



# Create list of characters + reverse mapping
char_set = set()
for ch in data:
   char_set.add(ch)

char_list = list(char_set)
char_list.sort()
print('char_list: ', char_list)

char_dict = {val: idx for idx, val in enumerate(char_list)}
print('char_dict: ', char_dict)


data_len = len(data)
print('data_len: ', data_len)


def get_sample(data, start_idx, length):
    # Get a sample and wrap around the data string
    return [char_dict[data[i % data_len]] for i in xrange(start_idx, start_idx+length)]

def get_input_target_sample(data, start_idx, length):
    sample = get_sample(data, start_idx, length+1)
    inpt = sample[0:length]
    trgt = sample[1:length+1]
    return inpt, trgt

def get_batch(data, start_idxs, batch_len):
    batch_size = len(start_idxs)
    input_batch = np.zeros((batch_size, batch_len), dtype=np.int32)
    target_batch = np.zeros((batch_size, batch_len), dtype=np.int32)
    for i, start_idx in enumerate(start_idxs):
        sample = get_sample(data, start_idx, batch_len+1)
        input_batch[i,:] = sample[0:batch_len]
        target_batch[i,:] = sample[1:batch_len+1]
    return input_batch, target_batch


batch_size = 2
batch_len = 11

# start_idxs = np.random.random_integers(0, data_len, batch_size)
# print('start_idxs: ', start_idxs)
#
# input_batch, target_batch = get_batch(data, start_idxs, batch_len)
# print('input_batch: ', input_batch.shape, input_batch)
# print('target_batch: ', target_batch.shape, target_batch)

def get_batch_generator(data, batch_size, batch_len):
    data_len = len(data)
    print('data_len: ', data_len)
    start_idxs = np.random.random_integers(0, data_len, batch_size)
    print('start_idxs: ', start_idxs)
    while True:
        input_batch, target_batch = get_batch(data, start_idxs, batch_len)
        start_idxs = (start_idxs + batch_len) % data_len
        yield input_batch, target_batch

batch_generator = get_batch_generator(data, batch_size, batch_len)

for i in range(3):
    print('i: ', i)
    input_batch, target_batch = next(batch_generator)
    print('input_batch: ', input_batch.shape, input_batch)
    print('target_batch: ', target_batch.shape, target_batch)




#
#
# input_batch = np.zeros((batch_size, batch_len), dtype=np.int32)
# print('input_batch: ', input_batch.shape)
# target_batch = np.zeros((batch_size, batch_len), dtype=np.int32)
# print('target_batch: ', target_batch.shape)
#
#
# input_str = data[start_idx:start_idx+batch_len]
# target_str = data[start_idx+1:start_idx+batch_len+1]
#
# print('input_str: ', input_str)
# print('target_str: ', target_str)
#
# inpt = np.zeros((batch_size, batch_len), dtype=np.int32)
# print('inpt: ', inpt.shape)
# inpt[0,:] = [char_dict[ch] for ch in input_str]
# print('inpt: ', inpt)


# trgt = np.zeros((batch_size, batch_len), dtype=np.int32)
# print('trgt: ', trgt.shape)
# trgt[0,:] = [char_dict[ch] for ch in target_str]
# print('trgt: ', trgt)
