# -*- coding:utf8 -*-
'''
实现最最基本的ENCODER-DECODER结构
实现模型的存取功能
使用SAMPLE和BEAM-SEARCH获得比较好的结果
'''

import tensorflow as tf
from util.dictutil import load_dict
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, static_rnn
from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example
import numpy as np

# ----- 控制GPU资源 -----
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

# ----- 配置路径和参数 -----
dict_name = "./../dict/dict_30000.dict"


word_to_idx, idx_to_word = load_dict(dict_name)

class Config(object):
    def __init__(self, vocab_size):
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_step = 25
        self.embedding_size = 128
        self.vocab_size = vocab_size
        self.max_grad_norm = 15

class Model(object):

    def __init__(self, is_training, config):

        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_step
        self.embedding_size = embedding_size = config.embedding_size
        self.vocab_size = vocab_size = config.vocab_size
        self.lr = lr = config.learning_rate

        # ----- 输入变量 X, Y -----
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        with tf.device(":/cpu:0"):
            embedding = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                dtype=tf.float32)

        self.inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        with tf.variable_scope("encoder"):

            self.encoder_cell = LSTMCell(embedding_size)

            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_final_state,
              encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell,
                                                                         self.encoder_cell,
                                                                         self.inputs)

            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            encoder_final_state_c = tf.concat(
                (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

            encoder_final_state_h = tf.concat(
                (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

            encoder_final_state = LSTMStateTuple(
                c=encoder_final_state_c,
                h=encoder_final_state_h
            )

        with tf.variable_scope("decoder"):

            self.decoder_cell = LSTMCell(embedding_size*2)
            self.decoder_W = tf.Variable(
                tf.random_uniform([embedding_size*2, vocab_size], -1.0, 1.0),
                dtype=tf.float32)
            self.decoder_b = tf.Variable(tf.zeros([vocab_size], dtype=tf.float32))

        with tf.variable_scope("RNN"):

            #decoder_input = tf.nn.embedding_lookup(embedding, word_to_idx['<start>'])
            outputs = []

            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                decoder_out, decoder_state = static_rnn(self.decoder_cell,
                                                        self.inputs[:,time_step,:],
                                                        encoder_final_state)
                decoder_output = tf.add(tf.matmul(decoder_out, self.decoder_W), self.decoder_b)
                outputs.append(decoder_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, embedding_size])

        logits = tf.matmul(output, self.decoder_W) + self.decoder_b

        loss = sequence_loss_by_example([logits],
                                        [tf.reshape(self._targets, [-1])],
                                        [tf.ones([batch_size * num_steps])])

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = decoder_state
        self._logits = logits

        if not is_training:
            self._prob = tf.nn.softmax(logits)
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))


