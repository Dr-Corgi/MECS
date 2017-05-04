import os

import tensorflow as tf

from tf_chatbot.configs.config import TEST_DATASET_PATH, FLAGS
from tf_chatbot.lib import data_utils
from tf_chatbot.lib.seq2seq_model_utils import create_model, get_predicted_sentence


def predict():
    def _get_test_dataset():
        with open(TEST_DATASET_PATH) as test_fh:
            test_sentences = [s.strip() for s in test_fh.readlines()]
        return test_sentences

    results_filename = '_'.join(['results', str(FLAGS.num_layers), str(FLAGS.size), str(FLAGS.vocab_size)])
    results_path = os.path.join(FLAGS.results_dir, results_filename)

    with tf.Session() as sess, open(results_path, 'w') as results_fh:

        model = create_model(sess, forward_only=True)
        model.batch_size = 1

        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        test_dataset = _get_test_dataset()

        for sentence in test_dataset:
            predicted_sentence = get_predicted_sentence(sentence, vocab, rev_vocab, model, sess)
            print(sentence, '->', predicted_sentence)

            results_fh.write(predicted_sentence + '\n')