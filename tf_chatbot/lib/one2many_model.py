from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example, sequence_loss, embedding_rnn_decoder
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, BasicLSTMCell, MultiRNNCell, EmbeddingWrapper, static_rnn
from tf_chatbot.lib import data_utils
from tf_chatbot.configs.config import EMOTION_TYPE


class One2ManyModel(object):
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size),
                    dtype)

            softmax_loss_function = sampled_loss

        def single_cell():
            return GRUCell(size)

        if use_lstm:
            def single_cell():
                return BasicLSTMCell(size)
        cell = single_cell()
        if num_layers > 1:
            cell = MultiRNNCell([single_cell() for _ in range(num_layers)])

        def one2many_f(encoder_inputs, decoder_input_dict, do_decode):
            num_decoder_symbols_dict = {0: target_vocab_size, 1: target_vocab_size, 2: target_vocab_size,
                                        3: target_vocab_size, 4: target_vocab_size, 5: target_vocab_size}
            return one2many_rnn_seq2seq(
                encoder_inputs=encoder_inputs,
                decoder_inputs_dict=decoder_input_dict,
                cell=cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols_dict=num_decoder_symbols_dict,
                embedding_size=size,
                feed_previous=do_decode,
                output_projection=output_projection,
                dtype=tf.float32
            )

        # Feeds for inputs
        self.encoder_inputs = []
        self.decoder_inputs_dict = data_utils.DICT_LIST(EMOTION_TYPE)
        self.target_weights = data_utils.DICT_LIST(EMOTION_TYPE)

        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for j in range(len(EMOTION_TYPE)):  # six emotion types
            for i in range(buckets[-1][1] + 1):
                self.decoder_inputs_dict[j].append(tf.placeholder(tf.int32, shape=[None],
                                                                  name="decoder{0}_{1}".format(i, j)))
                self.target_weights[j].append(tf.placeholder(dtype, shape=[None],
                                                             name="weight{0}_{1}".format(i, j)))

        # targets are decoder inputs shifted by one
        targets = data_utils.DICT_LIST(EMOTION_TYPE)
        for j in range(len(EMOTION_TYPE)):
            targets[j] = [self.decoder_inputs_dict[j][i + 1]
                          for i in range(len(self.decoder_inputs_dict[j]) - 1)]

        if forward_only:
            self.outputs, self.losses = model_with_buckets(
                self.encoder_inputs, self.decoder_inputs_dict,
                targets, self.target_weights, buckets, lambda x, y: one2many_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in range(len(buckets)):
                    for j in range(len(EMOTION_TYPE)):
                        self.outputs[j][b] = [
                            tf.matmul(output, output_projection[0]) + output_projection[1]
                            for output in self.outputs[j][b]]

        else:
            self.outputs, self.losses = model_with_buckets(
                self.encoder_inputs, self.decoder_inputs_dict, targets,
                self.target_weights, buckets,
                lambda x, y: one2many_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = data_utils.DICT_LIST(EMOTION_TYPE)
            self.updates = data_utils.DICT_LIST(EMOTION_TYPE)
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                for j in range(len(EMOTION_TYPE)):
                    gradients = tf.gradients(self.losses[j][b], params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                     max_gradient_norm)
                    self.gradient_norms[j].append(norm)
                    if j == 0:
                        self.updates[j].append(opt.apply_gradients(
                            zip(clipped_gradients, params), global_step=self.global_step))
                    else:
                        self.updates[j].append(opt.apply_gradients(
                            zip(clipped_gradients, params)))

        self.saver = tf.train.Saver(tf.global_variables())

    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs = []
        decoder_inputs = data_utils.DICT_LIST(EMOTION_TYPE)

        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            for i in range(len(EMOTION_TYPE)):
                decoder_pad_size = decoder_size - len(decoder_input[i]) - 1
                decoder_inputs[i].append([data_utils.GO_ID] + decoder_input[i] +
                                         [data_utils.PAD_ID] * decoder_pad_size)

        batch_encoder_inputs = []
        batch_decoder_inputs = data_utils.DICT_LIST(EMOTION_TYPE)
        batch_weights = data_utils.DICT_LIST(EMOTION_TYPE)

        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

        for j in range(len(EMOTION_TYPE)):
            for length_idx in range(decoder_size):
                batch_decoder_inputs[j].append(
                    np.array([decoder_inputs[j][batch_idx][length_idx]
                              for batch_idx in range(self.batch_size)], dtype=np.int32))

                batch_weight = np.ones(self.batch_size, dtype=np.float32)
                for batch_idx in range(self.batch_size):
                    if length_idx < decoder_size - 1:
                        target = decoder_inputs[j][batch_idx][length_idx + 1]
                    if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                        batch_weight[batch_idx] = 0.0
                batch_weights[j].append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def step(self, session, encoder_inputs, decoder_inputs_dict, target_weights_dict,
             bucket_id, forward_only):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        for j in range(len(EMOTION_TYPE)):
            if len(decoder_inputs_dict[j]) != decoder_size:
                raise ValueError("Decoder[%d] length must be equal to the one in bucket,"
                                 " %d != %d." % (j, len(decoder_inputs_dict[j]), decoder_size))
            if len(target_weights_dict[j]) != decoder_size:
                raise ValueError("Weights[%d] length must be equal to the one in bucket,"
                                 " %d != %d" % (j, len(target_weights_dict[j]), decoder_size))

        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for j in range(len(EMOTION_TYPE)):
            for l in range(decoder_size):
                input_feed[self.decoder_inputs_dict[j][l].name] = decoder_inputs_dict[j][l]
                input_feed[self.target_weights[j][l].name] = target_weights_dict[j][l]

            last_target = self.decoder_inputs_dict[j][decoder_size].name
            input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            updates_feed = {j: self.updates[j][bucket_id] for j in range(len(EMOTION_TYPE))}
            gnorm_feed = {j: self.gradient_norms[j][bucket_id] for j in range(len(EMOTION_TYPE))}
            loss_feed = {j: self.losses[j][bucket_id] for j in range(len(EMOTION_TYPE))}
            output_feed = [updates_feed,
                           gnorm_feed,
                           loss_feed]
        else:
            loss_feed = {j: self.losses[j][bucket_id] for j in range(len(EMOTION_TYPE))}
            pred_feed = {j: self.outputs[j][bucket_id] for j in range(len(EMOTION_TYPE))}
            output_feed = [loss_feed, pred_feed]

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None
        else:
            return None, outputs[0], outputs[1]


def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    for j in range(len(EMOTION_TYPE)):
        if len(targets[j]) < buckets[-1][1]:
            raise ValueError("Length of targets[%d] (%d) must be at least that of last"
                             "bucket (%d)." % (j, len(targets[j]), buckets[-1][1]))
        if len(weights[j]) < buckets[-1][1]:
            raise ValueError("Length of weights[%d] (%d) must be at least that of last"
                             "bucket (%d)." % (j, len(weights[j]), buckets[-1][1]))
    all_decoder_inputs = []
    all_targets = []
    all_weights = []
    for i in range(len(EMOTION_TYPE)):
        all_decoder_inputs.append(decoder_inputs[i])
        all_targets.append(targets[i])
        all_weights.append(weights[i])
    # all_inputs = encoder_inputs + decoder_inputs + targets + weights
    all_inputs = encoder_inputs + all_decoder_inputs + all_targets + all_weights
    losses = data_utils.DICT_LIST(EMOTION_TYPE)
    outputs = data_utils.DICT_LIST(EMOTION_TYPE)
    with tf.variable_scope(name, "model_with_buckets", all_inputs):
        for j, bucket in enumerate(buckets):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                cut_decoder_inputs = {}
                for i in range(len(EMOTION_TYPE)):
                    cut_decoder_inputs[i] = decoder_inputs[i][:bucket[1]]
                bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                            cut_decoder_inputs)
                # outputs.append(bucket_outputs)
                for i in range(len(EMOTION_TYPE)):
                    outputs[i].append(bucket_outputs[i])
                for i in range(len(EMOTION_TYPE)):
                    if per_example_loss:
                        losses[i].append(
                            sequence_loss_by_example(
                                outputs[i][-1],
                                targets[i][:bucket[1]],
                                weights[i][:bucket[1]],
                                softmax_loss_function=softmax_loss_function))
                    else:
                        losses[i].append(
                            sequence_loss(
                                outputs[i][-1],
                                targets[i][:bucket[1]],
                                weights[i][:bucket[1]],
                                softmax_loss_function=softmax_loss_function))

    return outputs, losses


def one2many_rnn_seq2seq(encoder_inputs,
                         decoder_inputs_dict,
                         cell,
                         num_encoder_symbols,
                         num_decoder_symbols_dict,
                         embedding_size,
                         output_projection=None,
                         feed_previous=False,
                         dtype=None,
                         scope=None):

    outputs_dict = {}
    state_dict = {}

    with tf.variable_scope("one2many_rnn_seq2seq"):
        # Encoder.
        encoder_cell = EmbeddingWrapper(
            cell,
            embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        _, encoder_state = static_rnn(
            encoder_cell, encoder_inputs, dtype=dtype)

        # Decoder.
        for name, decoder_inputs in decoder_inputs_dict.items():
            num_decoder_symbols = num_decoder_symbols_dict[name]

            with tf.variable_scope("one2many_decoder_" + str(name)):
                decoder_cell = cell
                if isinstance(feed_previous, bool):
                    outputs, state = embedding_rnn_decoder(
                        decoder_inputs,
                        encoder_state,
                        decoder_cell,
                        num_decoder_symbols,
                        embedding_size,
                        output_projection=output_projection,
                        feed_previous=feed_previous)
                else:
                    raise NotImplementedError()
            outputs_dict[name] = outputs
            state_dict[name] = state

    return outputs_dict, state_dict
