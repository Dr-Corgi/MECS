# -*- coding:utf8 -*-
from util.datautil import splitData
from model.blstm_atten.model import Config, Model
from util.datautil import load_corpus, batch_generator
import tensorflow as tf

# 对训练数据进行切割
# splitData()

config = Config()
model = Model(config)
sess = tf.Session()
model.variables_init(sess)
model.train(sess)