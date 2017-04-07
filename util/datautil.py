# -*- coding:utf8 -*-
import json
from numpy.random import shuffle
from conf.profile import TOKEN_UNK, TOKEN_EOS, TOKEN_START
import numpy as np


# 读取语料,返回语料列表,格式为[[Q1, EQ1, A1, EA1],...]
def load_corpus(corpus_name, encoding='utf8'):
    return json.load(open(corpus_name), encoding)


# 根据语料和词典生成训练batch
def batch_generator(corpus, batch_size, word_to_index):
    current_index = 0
    while True:
        if current_index + batch_size > len(corpus):
            shuffle(corpus)
            current_index = 0

        r_q = list()    # 问题文本
        r_a = list()    # 回复文本
        r_qe = list()   # 问题情绪标签
        r_ae = list()   # 回复情绪标签

        for [q,qe],[a,ae] in corpus[current_index: current_index+batch_size]:
            r_q.append(seq_index(word_to_index, q.strip().split(" ")))
            r_a.append(seq_index(word_to_index, a.strip().split(" ")))
            r_qe.append(qe)
            r_ae.append(ae)

        current_index += batch_size

        yield [r_q, r_a, r_qe, r_ae]


# 根据文本序列和词典生成index序列
def seq_index(vocab_to_idx, seq):
    res = [vocab_to_idx.get(s, vocab_to_idx[TOKEN_UNK]) for s in seq]
    res.append(vocab_to_idx[TOKEN_EOS])
    return res


# 将输入转化为 [ max_seq_len, batch_size ]矩阵,并使用<pad>对句子进行填充.
def batch_op(inputs, pad_idx, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.ones(shape=[batch_size, max_sequence_length], dtype=np.int32) * pad_idx

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i,j] = element

    inputs_time_major = inputs_batch_major.swapaxes(0,1)
    #inputs_time_major = inputs_batch_major

    return inputs_time_major, sequence_lengths


def dinput_op(inputs, pad_idx, start_idx, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    dinput_batch_major = np.ones(shape=[batch_size, max_sequence_length], dtype=np.int32) * pad_idx

    for i in range(batch_size):dinput_batch_major[i,0] = start_idx

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq[:-1]):
                dinput_batch_major[i, j+1] = element

    dinput_time_major = dinput_batch_major.swapaxes(0,1)

    return dinput_time_major, sequence_lengths