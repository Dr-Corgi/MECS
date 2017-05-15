from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import tf_chatbot.lib.data_utils as data_utils
from tensorflow.contrib.legacy_seq2seq import sequence_loss, attention_decoder
from tensorflow.contrib.rnn import GRUCell, BasicLSTMCell, MultiRNNCell, EmbeddingWrapper, static_rnn, OutputProjectionWrapper


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev

    return loop_function

def _extract_sample_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):

    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                prev, output_projection[0], output_projection[1])
            prev_symbol = tf.squeeze(tf.multinomial(prev, 1), axis=1)
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                use_sample=False,
                                initial_state_attention=False):
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with tf.variable_scope("embedding_attention_decoder", dtype=dtype):
        embedding = tf.get_variable("embedding", [num_symbols, embedding_size])

        if feed_previous:
            if use_sample:
                loop_function = _extract_sample_and_embed(
                    embedding, output_projection,
                    update_embedding_for_previous)
            else:
                loop_function = _extract_argmax_and_embed(
                    embedding, output_projection,
                    update_embedding_for_previous)
        else:
            loop_function = None
        emb_inp = [
            tf.nn.embedding_lookup(embedding,i) for i in decoder_inputs]

        return attention_decoder(
            emb_inp,
            initial_state,
            attention_states,
            cell,
            output_size=output_size,
            num_heads=num_heads,
            loop_function=loop_function,
            initial_state_attention=initial_state_attention)


class Seq2SeqModel(object):
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
                 beam_search_size=1,
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
        # Sampled softmax only makes sense if we sample less than vocabulary
        # size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable(
                "proj_w", [
                    self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable(
                "proj_b", [
                    self.target_vocab_size], dtype=dtype)
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

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            return GRUCell(size)

        if use_lstm:
            def single_cell():
                return BasicLSTMCell(size)
        cell = single_cell()
        if num_layers > 1:
            cell = MultiRNNCell([single_cell() for _ in range(num_layers)])

        self.model_encoder_states = {}
        self.model_attention_states = {}
        self.topk_probs = []
        self.topk_ids = []

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode, bucket_id):

            def embedding_attention_sampled_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    num_encoder_symbols,
                    num_decoder_symbols,
                    embedding_size,
                    bucket_index,
                    num_heads=1,
                    output_projection=None,
                    feed_previous=False,
                    initial_state_attention=False,
                    dtype=tf.float32):
                with tf.variable_scope("embedding_attention_sampled_seq2seq"):
                    encoder_cell = EmbeddingWrapper(
                        cell,
                        embedding_classes=num_encoder_symbols,
                        embedding_size=embedding_size
                    )
                    encoder_outputs, encoder_state = static_rnn(
                        encoder_cell, encoder_inputs, dtype=dtype)

                    top_states = [tf.reshape(
                        e, [-1, 1, cell.output_size]) for e in encoder_outputs]
                    attention_states = tf.concat(top_states, 1)

                    self.model_encoder_states[bucket_index] = encoder_state
                    self.model_attention_states[bucket_index] = attention_states

                    output_size = None
                    if output_projection is None:
                        cell = OutputProjectionWrapper(
                            cell, num_decoder_symbols)
                        output_size = num_decoder_symbols

                    if isinstance(feed_previous, bool):
                        return embedding_attention_decoder(
                            decoder_inputs,
                            # encoder_state,
                            self.model_encoder_states[bucket_index],
                            # attention_states,
                            self.model_attention_states[bucket_index],
                            cell,
                            num_decoder_symbols,
                            embedding_size,
                            num_heads=num_heads,
                            output_size=output_size,
                            output_projection=output_projection,
                            feed_previous=feed_previous,
                            use_sample=self.use_sample,
                            initial_state_attention=initial_state_attention)

                    else:
                        raise NotImplementedError()

            return embedding_attention_sampled_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                bucket_index=bucket_id,
                output_projection=output_projection,
                feed_previous=do_decode)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(
                tf.placeholder(
                    tf.int32,
                    shape=[None],
                    name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(
                tf.placeholder(
                    tf.int32,
                    shape=[None],
                    name="decoder{0}".format(i)))
            self.target_weights.append(
                tf.placeholder(
                    dtype,
                    shape=[None],
                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        with tf.variable_scope("model_with_buckets"):
            self.losses = []
            self.outputs = []
            self.decoder_out_state = []
            for bucket_idx, bucket in enumerate(buckets):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True if bucket_idx > 0 else None):
                    if forward_only:
                        bucket_outputs, bucket_outputs_state = seq2seq_f(self.encoder_inputs[:bucket[0]],
                                                                         self.decoder_inputs[:bucket[1]],
                                                                         True,
                                                                         bucket_idx)
                    else:
                        bucket_outputs, bucket_outputs_state = seq2seq_f(self.encoder_inputs[:bucket[0]],
                                                                         self.decoder_inputs[:bucket[1]],
                                                                         False,
                                                                         bucket_idx)
                    self.outputs.append(bucket_outputs)
                    self.decoder_out_state.append(bucket_outputs_state)
                    self.losses.append(
                        sequence_loss(
                            self.outputs[-1],
                            targets[:bucket[1]],
                            self.target_weights[:bucket[1]],
                            softmax_loss_function=softmax_loss_function
                        )
                    )

        if forward_only and output_projection is not None:
            for b in range(len(buckets)):
                self.outputs[b] = [
                    tf.matmul(output, output_projection[0]) + output_projection[1]
                    for output in self.outputs[b]]
                #best_outputs = [tf.argmax(x,1) for x in self.outputs[b]]
                #best_outputs = tf.concat(axis=1, values=[tf.reshape(x, [self.batch_size, 1]) for x in best_outputs])
                _topk_log_probs, _topk_ids = tf.nn.top_k(
                    tf.nn.softmax(self.outputs[b][-1]), beam_search_size)
                self.topk_probs.append(_topk_log_probs)
                self.topk_ids.append(_topk_ids)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    def step(
            self,
            session,
            encoder_inputs,
            decoder_inputs,
            target_weights,
            bucket_id,
            forward_only,
            use_beam_search=False):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError(
                "Encoder length must be equal to the one in bucket,"
                " %d != %d." %
                (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError(
                "Decoder length must be equal to the one in bucket,"
                " %d != %d." %
                (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError(
                "Weights length must be equal to the one in bucket,"
                " %d != %d." %
                (len(target_weights), decoder_size))

        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
            outputs = session.run(output_feed, input_feed)

            # Gradient norm, loss, no outputs.
            return outputs[1], outputs[2], None
        else:
            if use_beam_search:
                output_feed = [self.model_attention_states[bucket_id],
                               self.model_encoder_states[bucket_id]]
                # attention_state, encoder_state
                outputs = session.run(output_feed, input_feed)
                # score, result, last_token, encoder_state
                beams = [(0.0,
                          [data_utils.GO_ID],
                          data_utils.GO_ID,
                          outputs[1])] * 3
                result = []
                step = 0
                attention_state = outputs[0]

                while step < decoder_size and len(
                        result) < self.beam_search_size:
                    step += 1
                    _last_tokens = [beam_[2] for beam_ in beams]
                    _encoder_state = [beam_[3] for beam_ in beams]
                    output_feed = [
                        self.topk_ids[bucket_id],
                        self.topk_probs[bucket_id],
                        self.decoder_out_state[bucket_id]]
                    input_feed = {}
                    input_feed[self.model_attention_states[bucket_id].name] = attention_state
                    input_feed[self.model_encoder_states[bucket_id].name] = np.squeeze(
                        np.array(_encoder_state))
                    for l in range(step):
                        _decoder_inputs = [beam_[1][l] for beam_ in beams]
                        input_feed[self.decoder_inputs[l].name] = _decoder_inputs

                    _tok_ids, _tok_probs, _out_states = session.run(
                        output_feed, input_feed)

                    new_beams = []

                    for beam_idx in range(self.beam_search_size):
                        for _idx in range(self.beam_search_size):
                            new_beams.append(
                                (beams[beam_idx][0] + _tok_probs[beam_idx][_idx],
                                 beams[beam_idx][1] + [
                                    _tok_ids[beam_idx][_idx]],
                                    _tok_ids[beam_idx][_idx],
                                    _out_states[beam_idx]))

                    new_beams.sort(key=lambda x: x[0], reverse=True)
                    beams = []
                    for beam_ in new_beams:
                        if beam_[2] == data_utils.EOS_ID and len(beam_[1]) > 2:
                            result.append(
                                (beam_[0], beam_[1][:-1], beam_[2], beam_[3]))
                        else:
                            beams.append(beam_)
                            if len(beams) == self.beam_search_size:
                                break

                    if step == decoder_size:
                        for beam_ in beams:
                            result.append(beam_)
                            if len(result) == self.beam_search_size:
                                break

                outputs = result[0]
                return None, None, outputs[1]

            else:
                output_feed = [self.losses[bucket_id]]  # Loss for this batch.
                for l in range(decoder_size):  # Output logits.
                    output_feed.append(self.outputs[bucket_id][l])

                outputs = session.run(output_feed, input_feed)
                return None, outputs[0], outputs[1:]

    def get_batch(self, data, bucket_id):

        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            encoder_pad = [data_utils.PAD_ID] * \
                (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
