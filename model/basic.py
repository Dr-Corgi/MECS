# -*- coding:utf8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, AttentionCellWrapper, LSTMStateTuple, BasicLSTMCell
from util.dictutil import load_dict
from util.datautil import batch_generator, load_corpus, batch_op, seq_index
from conf.profile import TOKEN_EOS, TOKEN_PAD, TOKEN_START, TOKEN_UNK

'''
最基本的生成模型.
使用单向LSTM作为ENCODER.
使用单向LSTM作为DECODER.
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
        self.max_batch = 500
        self.batches_in_epoch = 100


class Model(object):

    def __init__(self, config):
        #sess = tf.Session()

        self.vocab_to_idx = config.vocab_to_idx
        self.idx_to_vocab = config.idx_to_vocab

        self.idx_start = config.vocab_to_idx[TOKEN_START]
        self.idx_eos = config.vocab_to_idx[TOKEN_EOS]
        self.idx_pad = config.vocab_to_idx[TOKEN_PAD]
        self.idx_unk = config.vocab_to_idx[TOKEN_UNK]

        self.vocab_size = vocab_size = config.vocab_size
        self.embedding_size = embedding_size = config.embedding_size
        self.hidden_unit = hidden_unit = config.hidden_unit

        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='encoder_inputs_length')
        self.decoder_targets = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_targets')
        #self.decoder_targets_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='decoder_targets_length')

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
        decoder_length, batch_size = tf.unstack(tf.shape(self.decoder_targets))

        W = tf.Variable(tf.random_uniform([hidden_unit, vocab_size], -1.0, 1.0), dtype=tf.float32)
        b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

        start_time_slice = tf.ones(shape=[batch_size], dtype=tf.int32, name='START') * self.idx_start
        start_embedded = tf.nn.embedding_lookup(self.embeddings, start_time_slice)

        pad_time_slice = tf.ones(shape=[batch_size], dtype=tf.int32, name='PAD') * self.idx_pad
        pad_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)

        def loop_fn_initial():
            initial_elements_finished = (0 >= decoder_length)
            initial_input = start_embedded
            initial_cell_state = self.encoder_final_state
            initial_cell_output = None
            initial_loop_state = None
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

            def get_next_input():
                output_logits = tf.add(tf.matmul(previous_output, W), b)
                prediction = tf.argmax(output_logits, axis=1)
                next_input = tf.nn.embedding_lookup(self.embeddings, prediction)
                return next_input

            elements_finished = (time >= decoder_length)
            finished = tf.reduce_all(elements_finished)
            input = tf.cond(finished, lambda: pad_embedded, get_next_input)
            state = previous_state
            output = previous_output
            loop_state = None

            return (elements_finished, input, state, output, loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:
                assert previous_output is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        with tf.variable_scope("decoder"):
            decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)

        decoder_outputs = decoder_outputs_ta.stack()
        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))

        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

        self.decoder_prediction = tf.argmax(decoder_logits, 2)

        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=vocab_size, dtype=tf.float32),
            logits=decoder_logits,
        )

        self.loss = tf.reduce_mean(stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, sess):

        sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver()

        loss_track = []

        for batch in range(config.max_batch):
            fd = self.next_feed()
            _, l = sess.run([self.train_op, self.loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % config.batches_in_epoch == 0:
                print('batch {}'.format(batch))
                print('  minibatch loss: {}').format(sess.run(self.loss, fd))
                predict_ = sess.run(self.decoder_prediction, fd)
                for i, (target, pred) in enumerate(zip(fd[self.decoder_targets].T, predict_.T)):
                    print('  sample {}:'.format(i+1))
                    str_tar = ""
                    for j in target: str_tar += config.idx_to_vocab[j]
                    #print('    target     > {}'.format([idx_to_vocab[j] for j in target]))
                    print('    target     > ')
                    print(str_tar)

                    str_pred = ""
                    for j in pred: str_pred += config.idx_to_vocab[j]
                    #print('    predicted > {}'.format([idx_to_vocab[j] for j in pred]))
                    print('    predicted     > ')
                    print(str_pred)
                    if i >= 2:
                        break
                    print ""
                #saver.save(sess, './../save/basic/my-model', global_step=batch)

    def next_feed(self):
        [r_q, r_a, r_qe, r_ae, r_ql, r_al] = batcher.next()
        encoder_inputs_, encoder_inputs_length_ = batch_op(r_q, self.idx_pad)
        decoder_targets_, _ = batch_op(r_a, self.idx_pad)

        return {
            self.encoder_inputs: encoder_inputs_,
            self.encoder_inputs_length: encoder_inputs_length_,
            self.decoder_targets: decoder_targets_
        }

    def predict(self, sess, inp):
        inp_index = [seq_index(self.vocab_to_idx, inp)]
        einp, einp_len = batch_op(inp_index, self.idx_pad)
        predict_ = sess.run(self.decoder_prediction,
                            feed_dict = {self.encoder_inputs: einp,
                                         self.encoder_inputs_length: einp_len,
                                         self.decoder_targets:[[self.idx_unk]]})

        for p in predict_.T:
            for j in p:
                print self.idx_to_vocab[j]

    '''
    while True:
        print ""
        user_inp = raw_input("  say > ")
        user_words = [user_inp.strip().split(" ")]
        user_words_idx = [[ vocab_to_idx.get(w, idx_unk)for w in words]for words in user_words]
        user_inp_batch, user_inp_batch_len = batch_op(user_words_idx, idx_pad)
        any_target = [[idx_pad],[idx_pad],[idx_pad],[idx_pad],[idx_pad],[idx_pad],[idx_pad],[idx_pad],[idx_pad],[idx_pad]]

        predict_ = sess.run(decoder_prediction, feed_dict={encoder_inputs: user_inp_batch,
                                                           encoder_inputs_length: user_inp_batch_len,
                                                           decoder_targets: any_target})
        for i, pred in enumerate(predict_.T):
            str = ""
            for j in pred:
                if j != idx_pad:
                    str += idx_to_vocab[j] + " "
            print('robot > {}'.format(str))

    '''


if __name__ == "__main__":

    config = Config()
    model = Model(config)
    sess = tf.Session()
    batcher = batch_generator(load_corpus(config.corpus_file),
                              batch_size=5,
                              word_to_index=config.vocab_to_idx)
    model.train(sess)
    model.predict(sess, "你 好")
