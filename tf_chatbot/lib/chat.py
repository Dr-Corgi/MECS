import os
import sys

import tensorflow as tf

from tf_chatbot.configs.config import FLAGS
from tf_chatbot.lib import data_utils
from tf_chatbot.lib.seq2seq_model_utils import create_model, get_predicted_sentence

def chat():
    with tf.Session() as sess:

        model = create_model(sess, forward_only=True)
        model.batch_size = 1

        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            predicted_sentence = get_predicted_sentence(sentence, vocab, model, sess)
            print(predicted_sentence)
            print("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()