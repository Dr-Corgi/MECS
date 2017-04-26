# -*- coding:utf8 -*-
from model.lstm.model import Config, Model
import tensorflow as tf
from util.datautil import loadPretrainedVector

# 对训练数据进行切割
# splitData()


config = Config()
#config.is_pretrained = False
model = Model(config)
sess = tf.Session()
model.variables_init(sess)
#model.train(sess)
'''

vocab_to_idx, idx_to_vocab, vocab_embed = loadPretrainedVector(30, 50, "./dict/vector/wiki.zh.small.text.vector")

for k in vocab_to_idx.keys():
    print(k, vocab_to_idx[k])

for k in idx_to_vocab.keys():
    print(k, idx_to_vocab[k])
'''