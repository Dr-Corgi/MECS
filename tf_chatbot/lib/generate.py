from __future__ import print_function
import os

import tensorflow as tf

from tf_chatbot.configs.config import TEST_DATASET_PATH, FLAGS, EMOTION_TYPE
from tf_chatbot.lib import data_utils
#from tf_chatbot.lib.seq2seq_model_utils import create_model, get_predicted_sentence
from tf_chatbot.lib.one2many_model_utils import create_model, get_predicted_sentence
import json

def predict():
    def _get_test_dataset():
        test_sentences = []
        with open("./tf_chatbot/data/test/ecm_test_data.txt", encoding='utf8') as fin:
            for line in fin:
                test_sentences.append(line.strip())
        return test_sentences

    output_data = []
    with tf.Session() as sess:

        #model = create_model(sess, forward_only=True, use_sample=FLAGS.use_sample)
        if FLAGS.use_beam_search:
            model = create_model(sess, forward_only=False, beam_forward_only=True, use_sample=FLAGS.use_sample)
        else:
            model = create_model(sess, forward_only=True, use_sample=FLAGS.use_sample)
        model.batch_size = 1

        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        test_dataset = _get_test_dataset()

        for sentence in test_dataset:
            predicted_sentence = get_predicted_sentence(sentence, vocab, rev_vocab, model, sess, use_beam_search=FLAGS.use_beam_search)
            print("".join(sentence.strip().split(" ")), '->')
            for i in range(6):
                print(EMOTION_TYPE[i] + ": ")
                print(predicted_sentence[i])
                output_data.append({"post": sentence, "emo":str(i), "res": predicted_sentence[i]})

    json.dump(output_data, open("SMIPG_1_EGG.txt", 'w', encoding='utf8'))

            #results_fh.write(predicted_sentence + '\n')

if __name__ == '__main__':
    test_sentences = []
    with open("../data/test/ecm_test_data.txt", encoding='utf8') as fin:
        for line in fin:
            test_sentences.append(line.strip())
    for sent in test_sentences[:10]:
        print(sent)