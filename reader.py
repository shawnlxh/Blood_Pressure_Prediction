#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================


"""Utilities for parsing IHealth text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import json
import pdb

import numpy as np
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()

def _read_users(filename):
  with open(filename, "r") as f:
    return json.loads(f.readline())

def _read_test_users(filename):
  with open(filename, "r") as f:
    test_user_list = json.loads(f.readline())
    test_user_list_filter = []
    for user in test_user_list:
      test_user_list_filter.append(user[-2:])
    return test_user_list_filter


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data]


def bp_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ihealth_data_train.json")
  test_path = os.path.join(data_path, "ihealth_data_test.json")

  train_data = _read_users(train_path)  #list
  test_data = _read_users(test_path)  #list
  return train_data, test_data

def bp_raw_data_batch(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "train_data.json")
  test_path = os.path.join(data_path, "test_data.json")

  train_path_len = os.path.join(data_path, "train_data_len.json")
  test_path_len = os.path.join(data_path, "test_data_len.json")

  train_data = _read_users(train_path)  #list
  train_data_len = _read_users(train_path_len)  #list
  test_data = _read_users(test_path)  #list
  test_data_len = _read_users(test_path_len)  #list
  return train_data, train_data_len, test_data, test_data_len


def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

def bp_iterator(data):
  """Iterate on the raw Blood Pressure data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """

  raw_data = np.array(data)
  data_len = len(raw_data)
  epoch_size = data_len

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = raw_data[i][:-1]
    z = []
    z.append(x)
    x = np.array(z)
    y = np.array(raw_data[i][1:])[:,0]  #低压
    # pdb.set_trace()
    yield (x, y)

def bp_iterator_bidirection(data):
  """Iterate on the raw Blood Pressure data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  #pdb.set_trace()
  data_len = len(data)
  epoch_size = data_len
  #pdb.set_trace()
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    # 输入数据集
    user_item = np.array(data[i])  #shape=(4,21)
    x = user_item[:-1, 0:17]
    x_expand = []
    x_expand.append(x.tolist())
    x = np.array(x_expand)
    # 用户属性数据
    x_profile = user_item[0, 17:21]
    # 预测目标数据
    y = user_item[1:, 0:1]  #低压
    yield (x, x_profile, y)

def bp_iterator_batch(data, datalen, batch_size, n_steps):
  """Iterate on the raw Blood Pressure data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  #pdb.set_trace()
  raw_data = np.array(data)
  data_len = len(raw_data)
  raw_datalen = np.array(datalen)
  epoch_size = data_len // batch_size

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    # 输入数据集
    x = raw_data[i * batch_size:(i + 1) *batch_size, :-1, :]
    # 预测目标数据
    y = raw_data[i * batch_size:(i + 1) *batch_size, 1:, 0:1]  #低压
    # 数据长度
    # w = raw_datalen[i * batch_size:(i + 1) *batch_size]
    # 以one-hot向量方式表示的数据长度
    raw_datalen_batch = raw_datalen[i * batch_size:(i + 1) *batch_size]
    z = np.zeros((batch_size,n_steps,1))
    z[range(batch_size),raw_datalen_batch-1,:] = 1
    # pdb.set_trace()
    yield (x, y, z)

def bp_iterator_batch_secondary(data, datalen, batch_size, n_steps):
  """Iterate on the raw Blood Pressure data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  #pdb.set_trace()
  raw_data = np.array(data)
  data_len = len(raw_data)
  raw_datalen = np.array(datalen)
  epoch_size = data_len // batch_size

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    # 输入数据集
    x = raw_data[i * batch_size:(i + 1) *batch_size, :-1, 0:17]
    x_profile = raw_data[i * batch_size:(i + 1) *batch_size, 0, 17:21]
    # 预测目标数据
    y = raw_data[i * batch_size:(i + 1) *batch_size, 1:, 0:1]  #低压
    # 数据长度
    # w = raw_datalen[i * batch_size:(i + 1) *batch_size]
    # 以one-hot向量方式表示的数据长度
    raw_datalen_batch = raw_datalen[i * batch_size:(i + 1) *batch_size]
    z = np.zeros((batch_size,n_steps,1))
    z[range(batch_size),raw_datalen_batch-1,:] = 1
    # pdb.set_trace()
    yield (x, x_profile, y, z)

def bp_iterator_batch_senior_01(data, datalen, batch_size, n_steps):
  """Iterate on the raw Blood Pressure data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  #pdb.set_trace()
  raw_data = np.array(data)
  data_len = len(raw_data)
  raw_datalen = np.array(datalen)
  epoch_size = data_len // batch_size

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    # 输入数据集
    x = raw_data[i * batch_size:(i + 1) *batch_size, :-1, :]
    x_profile = raw_data[i * batch_size:(i + 1) *batch_size, 0, 17:21]
    # 预测目标数据
    y = raw_data[i * batch_size:(i + 1) *batch_size, 1:, 0:1]  #低压
    # 数据长度
    # w = raw_datalen[i * batch_size:(i + 1) *batch_size]
    # 以one-hot向量方式表示的数据长度
    raw_datalen_batch = raw_datalen[i * batch_size:(i + 1) *batch_size]
    z = np.zeros((batch_size,n_steps,1))
    z[range(batch_size),raw_datalen_batch-1,:] = 1
    # pdb.set_trace()
    yield (x, x_profile, y, z)

def bp_iterator_batch_bidirectional(data, batch_size):
  """Iterate on the raw Blood Pressure data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  #pdb.set_trace()
  raw_data = np.array(data)
  data_len = len(raw_data)
  epoch_size = data_len // batch_size

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    # 输入数据集
    x = raw_data[i * batch_size:(i + 1) *batch_size, :-1, 0:17]
    x_profile = raw_data[i * batch_size:(i + 1) *batch_size, 0, 17:21]
    # 预测目标数据
    y = raw_data[i * batch_size:(i + 1) *batch_size, 1:, 0:1]  #低压
    # pdb.set_trace()
    yield (x, x_profile, y)