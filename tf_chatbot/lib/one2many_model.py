from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from tensorflow.contrib.legacy_seq2seq import sequence_loss
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, BasicLSTMCell, MultiRNNCell, EmbeddingWrapper, static_rnn, \
    OutputProjectionWrapper, RNNCell
from tf_chatbot.lib import data_utils
from tf_chatbot.configs.config import EMOTION_TYPE
from tensorflow.python.ops import array_ops, nn_ops
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear
from tensorflow.python.util import nest


class DAGRUCell(RNNCell):
    def __init__(self, num_units, da_num_units, input_size=None, activation=tf.tanh):
        self._num_units = num_units
        self._activation = activation

        self._da_num_units = da_num_units
        #self._da_w = tf.Variable(initial_value=tf.random_uniform(shape=[num_units * 2, self._da_num_units]),
        #                         dtype=tf.float32)
        #self._da_wd = tf.Variable(initial_value=tf.random_uniform(shape=[self._da_num_units, num_units]),
        #                          dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, da_state, scope=None):
        with tf.variable_scope(scope or "da_gru_cell"):
            with tf.variable_scope("gates"):
                r, u = array_ops.split(
                    value=_linear(
                        [inputs, state], 2 * self._num_units, True, 1.0, scope=scope),
                    num_or_size_splits=2,
                    axis=1)
                r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("da"):
                #da_wx = tf.matmul(tf.concat([inputs, state], axis=1), self._da_w)
                #da_r = tf.sigmoid(da_wx)
                with tf.variable_scope("wx"):
                    da_r = tf.sigmoid(_linear(tf.concat([inputs, state], axis=1), output_size=self._da_num_units, bias=False))

                new_da_state = da_state * da_r

                #da_wd = self._activation(tf.matmul(new_da_state, self._da_wd))
                with tf.variable_scope("w"):
                    da_wd = self._activation(_linear(new_da_state, self._num_units, bias=False))

            with tf.variable_scope("candidate"):
                c = self._activation(_linear([inputs, r * state],
                                             self._num_units, True,
                                             scope=scope) + da_wd)
            new_h = u * state + (1 - u) * c
        return new_h, new_h, new_da_state


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
                 use_sample=False,
                 forward_only=False,
                 beam_forward_only=True,
                 beam_search_size=1,
                 da_size=64,
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
        self.beam_search_size = beam_search_size
        self.use_sample = use_sample

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
            #return DAGRUCell(size, da_size)

        if use_lstm:
            def single_cell():
                return BasicLSTMCell(size)
        cell = single_cell()
        if num_layers > 1:
            cell = MultiRNNCell([single_cell() for _ in range(num_layers)])

        decoder_cell = DAGRUCell(size, da_size)

        self.model_encoder_states = {emo_idx: {} for emo_idx in EMOTION_TYPE.keys()}
        self.model_attention_states = {emo_idx: {} for emo_idx in EMOTION_TYPE.keys()}

        def one2many_f(encoder_inputs, decoder_input_dict, da_initial_state, do_decode, bucket_id):
            num_decoder_symbols_dict = {0: target_vocab_size, 1: target_vocab_size, 2: target_vocab_size,
                                        3: target_vocab_size, 4: target_vocab_size, 5: target_vocab_size}
            return self.one2many_rnn_seq2seq(
                encoder_inputs=encoder_inputs,
                decoder_inputs_dict=decoder_input_dict,
                da_initial_state=da_initial_state,
                cell=cell,
                decoder_cell=decoder_cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols_dict=num_decoder_symbols_dict,
                embedding_size=size,
                bucket_index=bucket_id,
                feed_previous=do_decode,
                output_projection=output_projection,
                dtype=tf.float32
            )

        # Feeds for inputs
        self.encoder_inputs = []
        self.decoder_inputs_dict = data_utils.gen_dict_list(EMOTION_TYPE)
        self.target_weights_dict = data_utils.gen_dict_list(EMOTION_TYPE)

        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for j in range(len(EMOTION_TYPE)):  # six emotion types
            for i in range(buckets[-1][1] + 1):
                self.decoder_inputs_dict[j].append(tf.placeholder(tf.int32, shape=[None],
                                                                  name="decoder{0}_{1}".format(i, j)))
                self.target_weights_dict[j].append(tf.placeholder(dtype, shape=[None],
                                                                  name="weight{0}_{1}".format(i, j)))

        # targets are decoder inputs shifted by one
        targets = data_utils.gen_dict_list(EMOTION_TYPE)
        for j in range(len(EMOTION_TYPE)):
            targets[j] = [self.decoder_inputs_dict[j][i + 1]
                          for i in range(len(self.decoder_inputs_dict[j]) - 1)]

        self.da_initial_state = tf.placeholder(dtype=tf.float32, shape=[None, da_size])

        with tf.variable_scope("model_with_buckets"):
            self.outputs = data_utils.gen_dict_list(EMOTION_TYPE)
            self.da_states = data_utils.gen_dict_list(EMOTION_TYPE)
            self.losses = data_utils.gen_dict_list(EMOTION_TYPE)
            for bucket_id, bucket in enumerate(buckets):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True if bucket_id > 0 else None):
                    emo_decoder_inputs = {}
                    for emo_idx in range(len(EMOTION_TYPE)):
                        emo_decoder_inputs[emo_idx] = self.decoder_inputs_dict[emo_idx][:bucket[1]]
                    if forward_only:
                        bucket_outputs, _, bucket_da_states = one2many_f(self.encoder_inputs[:bucket[0]],
                                                       emo_decoder_inputs,
                                                       self.da_initial_state,
                                                       True,
                                                       bucket_id)
                    else:
                        bucket_outputs, _, bucket_da_states = one2many_f(self.encoder_inputs[:bucket[0]],
                                                       emo_decoder_inputs,
                                                       self.da_initial_state,
                                                       False,
                                                       bucket_id)

                    for emo_idx in range(len(EMOTION_TYPE)):
                        self.outputs[emo_idx].append(bucket_outputs[emo_idx])
                        self.da_states[emo_idx].append(bucket_da_states[emo_idx])

                    def cal_da_step_loss(da_state, step_len):
                        #da_state_ts = tf.transpose(da_state, [1, 0, 2])
                        #loss = 0
                        #da_state_sents = tf.unstack(da_state_ts, num=self.batch_size)
                        #for sent in da_state_sents:
                        #    da_state_steps = tf.unstack(sent, num=step_len)
                        #    for j in range(step_len-1):
                        #        sub_tensor = tf.subtract(da_state_steps[j+1], da_state_steps[j])
                        #        loss += 0.0001 * tf.reduce_sum(100 **sub_tensor, axis=-1)

                        loss = 0
                        for i in range(1,step_len):
                            batch_d = tf.subtract(da_state[i], da_state[i-1])
                            loss += 0.0001 * tf.reduce_sum(tf.pow(2.0, batch_d))
                            #sent_d = tf.unstack(batch_d, num=self.batch_size)
                            #for _d in sent_d:
                            #    loss += tf.reduce_mean(_d)

                        #print(tf.shape(loss))
                        return loss

                    for emo_idx in range(len(EMOTION_TYPE)):
                        loss_2 = tf.reduce_sum(bucket_da_states[emo_idx][-1])
                        #print(tf.shape(loss_2))
                        loss_3 = cal_da_step_loss(bucket_da_states[emo_idx], data_utils.BUCKETS[bucket_id][1])
                        seq_loss = sequence_loss(self.outputs[emo_idx][-1],
                                          targets[emo_idx][:bucket[1]],
                                          self.target_weights_dict[emo_idx][:bucket[1]],
                                          softmax_loss_function=softmax_loss_function)
                        #print(tf.shape(seq_loss))
                        #self.losses[emo_idx].append(seq_loss + loss_2+loss_3)
                        self.losses[emo_idx].append(seq_loss + loss_2 + loss_3)

        if forward_only or beam_forward_only and output_projection is not None:
            for b in range(len(buckets)):
                for j in range(len(EMOTION_TYPE)):
                    self.outputs[j][b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[j][b]]

        params = tf.trainable_variables()
        if not forward_only and not beam_forward_only:
            self.gradient_norms = data_utils.gen_dict_list(EMOTION_TYPE)
            self.updates = data_utils.gen_dict_list(EMOTION_TYPE)
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

            for b in range(len(buckets)):
                for j in range(len(EMOTION_TYPE)):
                    print("begin to generate gradients - bucket %d emotion %d" % (b,j))
                    gradients = tf.gradients(self.losses[j][b], params)
                    print("generate gradients success - bucket %d emotion %d" % (b,j))
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
        encoder_topics = []
        decoder_inputs = data_utils.gen_dict_list(EMOTION_TYPE)

        for _ in range(self.batch_size):
            encoder_input, encoder_topic, decoder_input = random.choice(data[bucket_id])

            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            encoder_topics.append(encoder_topic)

            for i in range(len(EMOTION_TYPE)):
                decoder_pad_size = decoder_size - len(decoder_input[i]) - 1
                decoder_inputs[i].append([data_utils.GO_ID] + decoder_input[i] +
                                         [data_utils.PAD_ID] * decoder_pad_size)

        batch_encoder_inputs = []
        batch_decoder_inputs = data_utils.gen_dict_list(EMOTION_TYPE)
        batch_weights = data_utils.gen_dict_list(EMOTION_TYPE)

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

        return batch_encoder_inputs, encoder_topics, batch_decoder_inputs, batch_weights

    def step(self, session, encoder_inputs, encoder_topics, decoder_inputs_dict, target_weights_dict,
             bucket_id, forward_only, use_beam_search=False):
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
                input_feed[self.target_weights_dict[j][l].name] = target_weights_dict[j][l]

            last_target = self.decoder_inputs_dict[j][decoder_size].name
            input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        input_feed[self.da_initial_state.name] = encoder_topics

        if not forward_only:
            updates_feed = {j: self.updates[j][bucket_id] for j in range(len(EMOTION_TYPE))}
            gnorm_feed = {j: self.gradient_norms[j][bucket_id] for j in range(len(EMOTION_TYPE))}
            loss_feed = {j: self.losses[j][bucket_id] for j in range(len(EMOTION_TYPE))}
            output_feed = [updates_feed,
                           gnorm_feed,
                           loss_feed]
            outputs = session.run(output_feed, input_feed)
            return outputs[1], outputs[2], None
        else:
            if use_beam_search:
                output_feed = [self.model_attention_states[bucket_id],
                               self.model_encoder_states[bucket_id]]
                outputs = session.run(output_feed, input_feed)

                beams = {emo_idx: [(0.0, [data_utils.GO_ID], data_utils.GO_ID)] * self.beam_search_size for emo_idx in
                         EMOTION_TYPE.keys()}
                # beams = {emo_idx: [(0.0, [], data_utils.GO_ID)] * self.beam_search_size for emo_idx in EMOTION_TYPE.keys()}

                result = data_utils.gen_dict_list(EMOTION_TYPE)

                def numpy_softmax(x):
                    return np.exp(x) / np.sum(np.exp(x), axis=0)

                input_feed = {}
                input_feed[self.model_attention_states[bucket_id].name] = outputs[0]
                input_feed[self.model_encoder_states[bucket_id].name] = outputs[1]
                input_feed[self.da_initial_state.name] = encoder_topics
                step = 0
                run_flag = True

                while step < decoder_size and run_flag:
                    step += 1
                    output_feed = []

                    for emo_idx in EMOTION_TYPE.keys():
                        output_feed.append(self.outputs[emo_idx][bucket_id])

                        for l in range(step):
                            _emo_decoder_inputs = np.array([beam_[1][l] for beam_ in beams[emo_idx]])
                            input_feed[self.decoder_inputs_dict[emo_idx][l].name] = _emo_decoder_inputs

                        for l in range(step, decoder_size):
                            _emo_decoder_inputs = np.array([data_utils.PAD_ID for i in range(len(beams[emo_idx]))])
                            input_feed[self.decoder_inputs_dict[emo_idx][l].name] = _emo_decoder_inputs

                    _outputs = session.run(output_feed, input_feed)

                    _tok_probs = data_utils.gen_dict_list(EMOTION_TYPE)
                    _tok_ids = data_utils.gen_dict_list(EMOTION_TYPE)

                    for emo_idx in EMOTION_TYPE.keys():
                        if step == 1:
                            for _idx in range(self.beam_search_size):
                                _tok_ids[emo_idx].append(
                                    np.random.choice(range(self.target_vocab_size), size=self.beam_search_size,
                                                     replace=False, p=numpy_softmax(_outputs[emo_idx][step - 1][_idx])))
                                _tok_probs[emo_idx].append(
                                    numpy_softmax(_outputs[emo_idx][step - 1][_idx][_tok_ids[emo_idx][_idx]]))

                        else:
                            for _idx in range(self.beam_search_size):
                                _tok_ids[emo_idx].append(
                                    np.random.choice(range(self.target_vocab_size), size=self.beam_search_size,
                                                     replace=False, p=numpy_softmax(_outputs[emo_idx][step - 1][_idx])))
                                _tok_probs[emo_idx].append(
                                    numpy_softmax(_outputs[emo_idx][step - 1][_idx][_tok_ids[emo_idx][_idx]]))

                                # _tok_prob, _tok_id = tf.nn.top_k(tf.nn.softmax(_outputs[emo_idx][step-1][_idx]), self.beam_search_size)
                                # _tok_probs[emo_idx].append(_tok_prob.eval())
                                # _tok_ids[emo_idx].append(_tok_id.eval())

                                # _tok_ids[emo_idx].append(np.argsort(_outputs[emo_idx][step-1][_idx])[-self.beam_search_size:][::-1])
                                # _tok_probs[emo_idx].append(numpy_softmax(_outputs[emo_idx][step-1][_idx][_tok_ids[emo_idx][_idx]]))

                    new_beams = data_utils.gen_dict_list(EMOTION_TYPE)

                    for emo_idx in EMOTION_TYPE.keys():
                        for beam_idx in range(self.beam_search_size):
                            for _idx in range(self.beam_search_size):
                                # score = -(_tok_probs[emo_idx][beam_idx][_idx]) if _tok_ids[emo_idx][beam_idx][_idx] is data_utils.UNK_ID else _tok_probs[emo_idx][beam_idx][_idx]
                                if _tok_ids[emo_idx][beam_idx][_idx] == data_utils.UNK_ID:
                                    continue
                                # print(" idx:", _idx)
                                score = _tok_probs[emo_idx][beam_idx][_idx]
                                new_beams[emo_idx].append(
                                    (beams[emo_idx][beam_idx][0] + score,
                                     beams[emo_idx][beam_idx][1] + [_tok_ids[emo_idx][beam_idx][_idx]],
                                     _tok_ids[emo_idx][beam_idx][_idx]))

                        new_beams[emo_idx].sort(key=lambda x: x[0], reverse=True)

                    unduplicate_set_dict = {emo_idx: set() for emo_idx in EMOTION_TYPE.keys()}
                    beams = data_utils.gen_dict_list(EMOTION_TYPE)

                    for emo_idx in EMOTION_TYPE.keys():
                        for beam_ in new_beams[emo_idx]:
                            if beam_[2] == data_utils.EOS_ID:
                                result[emo_idx].append((beam_[0] / (len(beam_[1]) - 1), beam_[1][1:-1], beam_[2]))
                            else:
                                if str(beam_[1]) not in unduplicate_set_dict[emo_idx]:
                                    unduplicate_set_dict[emo_idx].add(str(beam_[1]))
                                    beams[emo_idx].append(beam_)
                                if len(beams[emo_idx]) == self.beam_search_size:
                                    break

                        beams[emo_idx] = beams[emo_idx][:self.beam_search_size]

                        if step == decoder_size:
                            for beam_ in new_beams[emo_idx]:
                                if len(result[emo_idx]) == self.beam_search_size:
                                    break
                                result[emo_idx].append((beam_[0] / (len(beam_[1]) - 1), beam_[1][1:], beam_[2]))

                    if sum([1 for emo_idx in EMOTION_TYPE.keys() if
                            len(result[emo_idx]) >= self.beam_search_size]) == 6:
                        for emo_idx in EMOTION_TYPE.keys():
                            result[emo_idx].sort(key=lambda x: x[0], reverse=True)
                        run_flag = False

                outputs = {emo_idx: (result[emo_idx][0][1], result[emo_idx][0][0]) for emo_idx in EMOTION_TYPE.keys()}
                return None, None, outputs
            else:
                loss_feed = {j: self.losses[j][bucket_id] for j in range(len(EMOTION_TYPE))}
                pred_feed = {j: self.outputs[j][bucket_id] for j in range(len(EMOTION_TYPE))}
                output_feed = [loss_feed, pred_feed]

                outputs = session.run(output_feed, input_feed)
                return None, outputs[0], outputs[1]

    def one2many_rnn_seq2seq(self,
                             encoder_inputs,
                             decoder_inputs_dict,
                             da_initial_state,
                             cell,
                             decoder_cell,
                             num_encoder_symbols,
                             num_decoder_symbols_dict,
                             embedding_size,
                             bucket_index,
                             num_heads=1,
                             output_projection=None,
                             feed_previous=False,
                             dtype=None,
                             scope=None,
                             initial_state_attention=False):
        outputs_dict = {}
        da_state_dict = {}
        state_dict = {}

        with tf.variable_scope("one2many_rnn_seq2seq"):
            encoder_cell = EmbeddingWrapper(
                cell,
                embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size)
            encoder_outputs, encoder_state = static_rnn(
                encoder_cell, encoder_inputs, dtype=dtype)

            top_states = [
                tf.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
            ]
            attention_states = tf.concat(top_states, 1)

            # for emo_idx in EMOTION_TYPE.keys():
            self.model_encoder_states[bucket_index] = encoder_state
            # self.model_encoder_states[emo_idx][bucket_index].name = encoder_state.name+"_emo"+str(emo_idx)
            self.model_attention_states[bucket_index] = attention_states
            # self.model_attention_states[emo_idx][bucket_index].name = encoder_state.name+"_emo"+str(emo_idx)

            # Decoder.
            for name, decoder_inputs in decoder_inputs_dict.items():
                num_decoder_symbols = num_decoder_symbols_dict[name]

                with tf.variable_scope("one2many_decoder_" + str(name)):
                    output_size = None
                    decoder_cell = decoder_cell
                    if output_projection is None:
                        decoder_cell = OutputProjectionWrapper(cell, num_decoder_symbols)
                        output_size = num_decoder_symbols
                    if isinstance(feed_previous, bool):
                        outputs, state, da_states = embedding_attention_decoder(
                            decoder_inputs,
                            self.model_encoder_states[bucket_index],
                            self.model_attention_states[bucket_index],
                            da_initial_state,
                            decoder_cell,
                            num_decoder_symbols,
                            embedding_size,
                            num_heads=num_heads,
                            output_size=output_size,
                            output_projection=output_projection,
                            feed_previous=feed_previous,
                            initial_state_attention=initial_state_attention)
                    else:
                        raise NotImplementedError()
                outputs_dict[name] = outputs
                state_dict[name] = state
                da_state_dict[name] = da_states

        return outputs_dict, state_dict, da_state_dict


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                da_initial_state,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with tf.variable_scope(
                    scope or "embedding_attention_decoder", dtype=dtype) as scope:

        embedding = tf.get_variable("embedding",
                                    [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs
        ]
        return attention_decoder(
            emb_inp,
            initial_state,
            attention_states,
            da_initial_state,
            cell,
            output_size=output_size,
            num_heads=num_heads,
            loop_function=loop_function,
            initial_state_attention=initial_state_attention)


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      da_initial_state,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with tf.variable_scope(
                    scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        with tf.variable_scope("da_states"):
            da_state = da_initial_state
            da_states = []

        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = tf.get_variable("AttnW_%d" % a,
                                [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                tf.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in range(num_heads):
                with tf.variable_scope("Attention_%d" % a):
                    y = _linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y),
                                      [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = tf.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in range(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = _linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state, da_state = cell(x, state, da_state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with tf.variable_scope(
                        tf.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with tf.variable_scope("AttnOutputProjection"):
                output = _linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)
            da_states.append(da_state)

    return outputs, state, da_states
