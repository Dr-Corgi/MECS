from __future__ import print_function
import os

import tensorflow as tf

from tf_chatbot.configs.config import TEST_DATASET_PATH, FLAGS, EMOTION_TYPE
from tf_chatbot.lib import data_utils
#from tf_chatbot.lib.seq2seq_model_utils import create_model, get_predicted_sentence
from tf_chatbot.lib.one2many_model_utils import create_model, get_predicted_sentence
import json
import gensim

def _load_gensim_model():
    dictionary_path = "./tf_chatbot/data/one2many_da/gensim_model/dictionary.model"
    lsi_path = "./tf_chatbot/data/one2many_da/gensim_model/LSI_model.model"
    Tfidf_path = "./tf_chatbot/data/one2many_da/gensim_model/Tfidf_model.model"

    dictionary = gensim.corpora.Dictionary.load(dictionary_path)
    Tf_idf = gensim.models.TfidfModel.load(Tfidf_path)
    lsi = gensim.models.LsiModel.load(lsi_path)

    return dictionary, Tf_idf, lsi

def predict():
    def _get_test_dataset():
        data = json.load(open(TEST_DATASET_PATH))
        test_sentences = [q for ((q, qe), _) in data]
        return test_sentences

    results_filename = '_'.join(['results', str(FLAGS.num_layers), str(FLAGS.size), str(FLAGS.vocab_size)])
    results_path = os.path.join(FLAGS.results_dir, results_filename)

    dictionary, Tf_idf, lsi = _load_gensim_model()

    with tf.Session() as sess, open(results_path, 'w') as results_fh:

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

            sent_d2b = dictionary.doc2bow(sentence.strip().split(" "))
            sent_lsi = lsi[sent_d2b]
            sent_lsi.sort(key=lambda x: x[1], reverse=True)
            sent_topics = [0] * 64
            for sl in sent_lsi[:10]:
                sent_topics[sl[0]] = 1

            for i in range(len(sent_topics)):
                sent_topics[i] = float(sent_topics[i])

            predicted_sentence = get_predicted_sentence(sentence, sent_topics, vocab, rev_vocab, model, sess, use_beam_search=FLAGS.use_beam_search)
            print("".join(sentence.strip().split(" ")), '->')
            for i in range(6):
                print(EMOTION_TYPE[i] + ": ")
                print("".join(predicted_sentence[i].split(" ")))

            results_fh.write("".join(sentence.strip().split(" ")) + "\n")
            for i in range(6):
                results_fh.write(EMOTION_TYPE[i])
                results_fh.write(" -> ")
                results_fh.write("".join(predicted_sentence[i].split(" ")))
                results_fh.write("\n")
            results_fh.write("\n")

            #results_fh.write(predicted_sentence + '\n')

if __name__ == '__main__':
    dictionary, Tf_idf, lsi = _load_gensim_model()