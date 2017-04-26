# -*- coding:utf8 -*-
from model.blstm.model import Config, Model
import tensorflow as tf

# 对训练数据进行切割
# splitData()

config = Config()
config.is_pretrained = False
model = Model(config)
sess = tf.Session()
model.variables_init(sess)
model.train(sess)