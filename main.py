# -*- coding:utf8 -*-
from model.lstm.model import Config, Model
import tensorflow as tf
from util.dictutil import loadPretrainedVector

# 对训练数据进行切割
# splitData()


config = Config()
config.is_pretrained = False
model = Model(config)
sess = tf.Session()
model.variables_init(sess)
model.train(sess)
model.loss_tracker.savefig(50, config.save_path)

resonse = model.generate(sess, "我 对此 感到 非常 开心")
print(resonse)

'''
vocab_to_idx, idx_to_vocab, vocab_embed = loadPretrainedVector(30, 50, "./dict/vector/wiki.zh.text200.vector")

for k in vocab_to_idx.keys():
    if u"他"==k:
        print(k, vocab_to_idx[k])

'''

#for k in idx_to_vocab.keys():
#    print(k, idx_to_vocab[k])

#for i in vocab_embed:
#    print(i)
