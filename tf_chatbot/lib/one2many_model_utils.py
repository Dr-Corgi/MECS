from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from tf_chatbot.configs.config import FLAGS, BUCKETS, EMOTION_TYPE
from tf_chatbot.lib import data_utils
#from tf_chatbot.lib import seq2seq_model
from tf_chatbot.lib import one2many_model

_INDEX = ".index"

def create_model(session, forward_only, beam_forward_only=False, use_sample=False):
    model = one2many_model.One2ManyModel(
        source_vocab_size=FLAGS.vocab_size,
        target_vocab_size=FLAGS.vocab_size,
        buckets=BUCKETS,
        size=FLAGS.size,
        num_layers=FLAGS.num_layers,
        max_gradient_norm=FLAGS.max_gradient_norm,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
        use_sample=use_sample,
        use_lstm=False,
        beam_forward_only=beam_forward_only,
        beam_search_size=FLAGS.beam_search_size,
        forward_only=forward_only)

    print("create model success!")

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path + _INDEX):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if ckpt:
            print("Unable to reach checkpoint file %s." % ckpt.model_checkpoint_path)
        print("Create model with fresh parameters")
        session.run(tf.global_variables_initializer())
    return model

def get_predicted_sentence(input_sentence, input_sentence_topics, vocab, rev_vocab, model, sess, use_beam_search=False):
    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)

    bucket_id = min([b for b in range(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    #bucket_id = np.random.choice([b for b in range(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    outputs = {0:[],1:[],2:[],3:[],4:[],5:[]}

    feed_data = {bucket_id: [(input_token_ids, input_sentence_topics, outputs)]}
    encoder_inputs, encoder_topic, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)

    if use_beam_search:
        new_encoder_inputs = []
        new_encoder_topics = []
        new_decoder_inputs = data_utils.gen_dict_list(EMOTION_TYPE)
        new_target_weights = data_utils.gen_dict_list(EMOTION_TYPE)
        for _emo_key, _emo_arrays in decoder_inputs.items():
            for _array in _emo_arrays:
                for _item in _array:
                    _de_input = np.array([_item] * FLAGS.beam_search_size, dtype=np.int32)
                    new_decoder_inputs[_emo_key].append(_de_input)
        for _array in encoder_inputs:
            for _item in _array:
                _en_input = np.array([_item] * FLAGS.beam_search_size, dtype=np.int32)
                new_encoder_inputs.append(_en_input)
        for _ in range(FLAGS.beam_search_size):
            for _item in encoder_topic:
                new_encoder_topics.append(_item)

        for _emo_key, _emo_arrays in target_weights.items():
            for _array in _emo_arrays:
                for _item in _array:
                    _ta_input = np.array([_item] * FLAGS.beam_search_size, dtype=np.int32)
                    new_target_weights[_emo_key].append(_ta_input)

        _, _, output_words = model.step(sess, new_encoder_inputs, new_encoder_topics, new_decoder_inputs, new_target_weights, bucket_id, forward_only=True, use_beam_search=True)
        output_sentences = {}
        for j in range(6):
            output_sentences[j] = " ".join([rev_vocab[tok_id] for tok_id in output_words[j][0]])+"["+str(output_words[j][1])+"]"

    else:
        _, _, output_logits = model.step(sess, encoder_inputs, [encoder_topic], decoder_inputs, target_weights, bucket_id, forward_only=True, use_beam_search=False)

        output_sentences = {}
        for j in range(6):
            for logit in output_logits[j]:
                selected_token_id = int(np.argmax(logit, axis=1))
                if selected_token_id == data_utils.EOS_ID:
                    break
                else:
                    outputs[j].append(selected_token_id)
            output_sentences[j] = " ".join([rev_vocab[output] for output in outputs[j]])

    return output_sentences