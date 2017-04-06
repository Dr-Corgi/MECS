# -*- coding:utf8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, AttentionCellWrapper, GRUCell, LSTMStateTuple, BasicLSTMCell
from util.dictutil import load_dict
from util.datautil import batch_generator, load_corpus, batch_op
from conf.profile import TOKEN_EOS, TOKEN_PAD, TOKEN_START, TOKEN_UNK

model_path = "./stc_model"


dict_name = "./../dict/dict_500.dict"
vocab_to_idx, idx_to_vocab = load_dict(dict_name)

batcher = batch_generator(load_corpus("./../data/tiny_data.json"), batch_size=5, word_to_index=vocab_to_idx)

# model
idx_start = vocab_to_idx[TOKEN_START]
idx_eos = vocab_to_idx[TOKEN_EOS]
idx_pad = vocab_to_idx[TOKEN_PAD]
idx_unk = vocab_to_idx[TOKEN_UNK]

vocab_size = len(vocab_to_idx)
embedding_size = 100

encoder_hidden_unit = 100
decoder_hidden_unit = encoder_hidden_unit

attn_length = 200

is_sample = False

#saver = tf.train.Saver()

if True:

    sess = tf.Session()

    encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='encoder_inputs')
    encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='encoder_inputs_length')
    decoder_targets = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_targets')
    decoder_targets_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='decoder_targets_length')

    with tf.device("/cpu:0"):
        embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)

    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

    encoder_cell = GRUCell(num_units=encoder_hidden_unit)

    encoder_cell = AttentionCellWrapper(cell=encoder_cell, attn_length=attn_length, state_is_tuple=True)

    with tf.variable_scope("encoder"):
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                 inputs=encoder_inputs_embedded,
                                                                 dtype=tf.float32,
                                                                 time_major=True)

    decoder_cell = GRUCell(num_units=decoder_hidden_unit)
    decoder_cell = AttentionCellWrapper(cell=decoder_cell, attn_length=attn_length, state_is_tuple=True)

    decoder_length, batch_size = tf.unstack(tf.shape(decoder_targets))
    decoder_length = decoder_length

    W = tf.Variable(tf.random_uniform([decoder_hidden_unit, vocab_size], -1.0, 1.0), dtype=tf.float32)
    b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

    start_time_slice = tf.ones(shape=[batch_size], dtype=tf.int32, name='START') * idx_start
    start_embedded = tf.nn.embedding_lookup(embeddings, start_time_slice)

    pad_time_slice = tf.ones(shape=[batch_size], dtype=tf.int32, name='PAD') * idx_pad
    pad_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

    eos_time_slice = tf.ones(shape=[batch_size], dtype=tf.int32, name='EOS') * idx_eos
    eos_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)

    def loop_fn_initial():
        initial_elements_finished = (0 >= decoder_length)
        initial_input = start_embedded
        initial_cell_state = encoder_final_state
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
            next_input = tf.nn.embedding_lookup(embeddings, prediction)
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

    decoder_prediction = tf.argmax(decoder_logits, 2)

    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
    )

    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    loss_track = []

    max_batches = 1000
    batches_in_epoch = 20

    def next_feed():
        [r_q, r_a, r_qe, r_ae, r_ql, r_al] = batcher.next()
        encoder_inputs_, encoder_inputs_length_ = batch_op(r_q, idx_pad)
        decoder_targets_, _ = batch_op(r_a, idx_pad)

        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_inputs_length_,
            decoder_targets: decoder_targets_
        }


    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}').format(sess.run(loss, fd))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (target, pred) in enumerate(zip(fd[decoder_targets].T, predict_.T)):
                print('  sample {}:'.format(i+1))
                str_tar = ""
                for j in target: str_tar += idx_to_vocab[j]
                #print('    target     > {}'.format([idx_to_vocab[j] for j in target]))
                print('    target     > ')
                print(str_tar)

                str_pred = ""
                for j in pred: str_pred += idx_to_vocab[j]
                #print('    predicted > {}'.format([idx_to_vocab[j] for j in pred]))
                print('    predicted     > ')
                print(str_pred)
                if i >= 2:
                    break
                print ""
            saver.save(sess, './../save/basic/my-model', global_step=batch)


    #new_saver = tf.train.import_meta_graph('my-model-80.meta')
    #new_saver.restore(sess, tf.train.latest_checkpoint('./'))


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
