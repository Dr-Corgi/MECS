# -*- coding:utf8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, AttentionCellWrapper, LSTMStateTuple, BasicLSTMCell
from tensorflow.contrib.layers import linear
from util.dictutil import load_dict
from util.datautil import batch_generator, load_corpus, batch_op, seq_index, dinput_op
from conf.profile import TOKEN_EOS, TOKEN_PAD, TOKEN_START, TOKEN_UNK
import numpy as np

'''
最基本的生成模型.
使用单向LSTM作为ENCODER.
使用单向LSTM作为DECODER.
不使用SAMPLE.
不使用BEAM SEARCH.
不使用ATTENTION.
训练过程中采用TARGET中的字符,而不是上一次输出的结果.
'''


# Configuration
class Config(object):
    def __init__(self):
        self.embedding_size = 128
        self.hidden_unit = 128
        self.save_path = "./../save/basic/"
        self.model_name = "BasicModel"
        self.dict_file = "./../dict/dict_500.dict"
        self.corpus_file = "./../data/tiny_data.json"
        self.vocab_to_idx, self.idx_to_vocab = load_dict(self.dict_file)
        self.vocab_size = len(self.vocab_to_idx)
        self.max_batch = 1001
        self.save_step = 200
        self.batch_size = 5
        self.max_generate_len = 10

        self.is_beams = True
        self.beam_size = 3

class Model(object):

    def __init__(self, config):

        self.vocab_to_idx = config.vocab_to_idx
        self.idx_to_vocab = config.idx_to_vocab

        self.save_path = config.save_path
        self.model_name = config.model_name
        self.save_step = config.save_step
        self.max_batch = config.max_batch
        self.max_generate_len = config.max_generate_len

        self.is_beams = config.is_beams
        if self.is_beams: self.beam_size = config.beam_size

        self.idx_start = config.vocab_to_idx[TOKEN_START]
        self.idx_eos = config.vocab_to_idx[TOKEN_EOS]
        self.idx_pad = config.vocab_to_idx[TOKEN_PAD]
        self.idx_unk = config.vocab_to_idx[TOKEN_UNK]

        self.vocab_size = vocab_size = config.vocab_size
        self.embedding_size = embedding_size = config.embedding_size
        self.hidden_unit = hidden_unit = config.hidden_unit

        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='encoder_inputs_length')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_inputs')
        self.decoder_targets = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_targets')

        with tf.device("/cpu:0"):
            self.embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)

        encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        encoder_cell = LSTMCell(num_units=hidden_unit)

        with tf.variable_scope("encoder"):
            [self.encoder_outputs,
             self.encoder_final_state] = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                           inputs=encoder_inputs_embedded,
                                                           dtype=tf.float32,
                                                           time_major=True)

        decoder_cell = LSTMCell(num_units=hidden_unit)
        self.initial_state = self.encoder_final_state
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
        with tf.variable_scope("decoder"):
            (self.decoder_outputs,
             self.decoder_final_state) = tf.nn.dynamic_rnn(decoder_cell,
                                                           decoder_inputs_embedded,
                                                           initial_state=self.initial_state,
                                                           dtype=tf.float32,
                                                           time_major=True)

        self.decoder_logits = linear(self.decoder_outputs, vocab_size)

        self.prob = tf.nn.softmax(self.decoder_logits)

        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=vocab_size, dtype=tf.float32),
            logits=self.decoder_logits,
        )

        self.loss = tf.reduce_mean(stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def variables_init(self, sess):
        sess.run(tf.global_variables_initializer())

    def train(self, sess):
        loss_track = []
        for batch in range(self.max_batch):
            fd = self.next_feed()
            _, l = sess.run([self.train_op, self.loss], fd)
            loss_track.append(l)

            if batch % self.save_step == 0 and batch != 0:
                print('batch {}'.format(batch))
                print('  minibatch loss: {}').format(sess.run(self.loss, fd))
                predict_ = sess.run(self.decoder_prediction, fd)
                self.__print_result(fd[self.decoder_targets], predict_)
                self.save(sess, batch)

    def next_feed(self):
        [r_q, r_a, r_qe, r_ae] = batcher.next()
        encoder_inputs_, encoder_inputs_length_ = batch_op(r_q, self.idx_pad)
        decoder_targets_, _ = batch_op(r_a, self.idx_pad)
        decoder_inputs_, _ = dinput_op(r_a, self.idx_pad, self.idx_start)

        return {
            self.encoder_inputs: encoder_inputs_,
            self.encoder_inputs_length: encoder_inputs_length_,
            self.decoder_inputs:decoder_inputs_,
            self.decoder_targets: decoder_targets_
        }

    # 生成回复的方法.
    # 思路:   1.首先将Question传入Encoder,得到隐藏状态c;
    #        2.传入<start>生成第一个单词,并得到decoder的新隐藏状态c;
    #        3.传入刚才生成的单词,继续预测下一个单词.
    def generate(self, sess, inp):
        inp_index = [seq_index(self.vocab_to_idx, inp)]
        einp, einp_len = batch_op(inp_index, self.idx_pad)
        state_ = sess.run([self.encoder_final_state],
                          feed_dict = {self.encoder_inputs: einp,
                                       self.encoder_inputs_length: einp_len})
        '''
        result_index = []
        result_str = ""

        result_index.append(predict_[0,0])
        result_str += self.idx_to_vocab[predict_[0,0]]

        while result_index[-1] != self.idx_eos:
            predict_,last_state = sess.run([self.decoder_prediction, self.decoder_final_state],
                            feed_dict = {self.initial_state: last_state,
                                         self.decoder_inputs:[[self.idx_start]]})
            result_index.append(predict_[0,0])
            result_str += self.idx_to_vocab[predict_[0,0]]
            if len(result_index) > self.max_generate_len:
                break

        print result_str
        '''

        if self.is_beams:
            beams = [(0.0, "", [])]
            tdata = self.vocab_to_idx[TOKEN_START]
            prob_, state_ = sess.run([self.prob, self.decoder_final_state],
                                     feed_dict={self.decoder_inputs: [[tdata]],
                                                self.initial_state: state_})

            y = np.log(1e-20 + prob_.reshape(-1))
            top_indices = np.argsort(-y)
            b = beams[0]
            beam_candidates = []
            for bc in xrange(self.beam_size):
                vocab_idx = top_indices[bc]
                beam_candidates.append((b[0]+y[vocab_idx], b[1]+self.idx_to_vocab[vocab_idx], vocab_idx, state_))
            beam_candidates.sort(key=lambda x:x[0], reverse=True)
            beams = beam_candidates[:self.beam_size]
            for _ in range(self.max_generate_len-1):
                beam_candidates = []
                for b in beams:
                    tdata = np.int32(b[2])
                    prob_, state_ = sess.run([self.prob, self.decoder_final_state],
                                             feed_dict={self.decoder_inputs: [[tdata]],
                                                        self.initial_state: b[3]})
                    y = np.log(1e-20 + prob_.reshape(-1))
                    top_indices = np.argsort(-y)
                    for bc in xrange(self.beam_size):
                        vocab_idx = top_indices[bc]
                        beam_candidates.append((b[0]+y[vocab_idx], b[1]+self.idx_to_vocab[vocab_idx], vocab_idx, state_))
                    beam_candidates.sort(key=lambda x:x[0], reverse=True)
                    beams = beam_candidates[:self.beam_size]

            return beams[0][1]

        else:
            tdata = self.vocab_to_idx[TOKEN_START]
            predict_, state_ = sess.run([self.decoder_prediction, self.decoder_final_state],
                                     feed_dict={self.decoder_inputs: [[tdata]],
                                                self.initial_state: state_})

            tdata = np.int32(predict_[0,0])
            response = self.idx_to_vocab[tdata]
            for _ in range(self.max_generate_len-1):
                predict_, state_ = sess.run([self.decoder_prediction, self.decoder_final_state],
                                            feed_dict={self.decoder_inputs: [[tdata]],
                                                       self.initial_state: state_})
                tdata = np.int32(predict_[0,0])
                response += self.idx_to_vocab[tdata]

            return response

    def __print_result(self, tar, pred):
        for i, (target, pred) in enumerate(zip(tar.T, pred.T)):
            print('  sample {}:'.format(i+1))
            str_tar = ""
            for j in target: str_tar += self.idx_to_vocab[j]
            print('    target     > ')
            print(str_tar)
            str_pred = ""
            for j in pred: str_pred += self.idx_to_vocab[j]
            print('    predicted     > ')
            print(str_pred)
            if i >= 2:
                break
            print ""

    def save(self, sess, step):
        saver = tf.train.Saver()
        saver.save(sess, self.save_path + self.model_name, global_step=step)

    def restore(self, sess, step):
        saver = tf.train.Saver()
        saver.restore(sess, self.save_path+self.model_name+'-'+str(step))
        return sess


if __name__ == "__main__":

    config = Config()
    model = Model(config)
    sess = tf.Session()
    batcher = batch_generator(load_corpus(config.corpus_file),
                              batch_size=config.batch_size,
                              word_to_index=config.vocab_to_idx)
    model.variables_init(sess)
    #model.train(sess)
    #model.save(sess, 100)
    #sess = tf.Session()
    sess = model.restore(sess, 800)
    response = model.generate(sess, "你 好")
    print response
