# -*- coding:utf8 -*-
import json
from numpy.random import shuffle

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

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

        r_q = []
        r_a = []
        r_qe = []
        r_ae = []

        for [q,qe],[a,ae] in corpus[current_index: current_index+batch_size]:
            r_q.append(seq_index(word_to_index, q.strip().split(" ")))
            r_a.append(seq_index(word_to_index, a.strip().split(" ")))
            r_qe.append(qe)
            r_ae.append(ae)

        current_index += batch_size

        yield [r_q, r_a, r_qe, r_ae]

# 根据文本序列和词典生成index序列
def seq_index(word_to_index, seq):
    return [word_to_index.get(s, word_to_index[UNK_TOKEN]) for s in seq]