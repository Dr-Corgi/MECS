import sys
import os
import math
import time

import numpy as np
import tensorflow as tf

from tf_chatbot.lib.seq2seq_model_utils import create_model, create_model_one2many
from tf_chatbot.configs.config import FLAGS, BUCKETS, EMOTION_TYPE
from tf_chatbot.lib.data_utils import read_data, read_data_one2many
from tf_chatbot.lib import data_utils

def train_one2many():
    print("Preparing dialog data in %s" % FLAGS.data_dir)
    train_data, dev_data, _ = data_utils.prepare_dialog_data_one2many(FLAGS.data_dir, FLAGS.vocab_size)

    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model_one2many(sess, forward_only=False)

        print("Reading development and training data (limit:%d)." % FLAGS.max_train_data_size)
        dev_set = read_data_one2many(dev_data)
        train_set = read_data_one2many(train_data, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(BUCKETS))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        step_time = 0.0
        current_step = 0
        previous_losses = []
        loss = {k:0.0 for k in EMOTION_TYPE.keys()}

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()
            encoder_inputs, decoder_inputs_dict, target_weights_dict = model.get_batch(
                train_set, bucket_id)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs_dict,
                                         target_weights_dict, bucket_id, forward_only=False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint

            for k, k_loss in step_loss.items():
                loss[k] += k_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = {k: math.exp(k_loss) if loss < 300 else float('inf') for k,k_loss in loss.items()}
                print("global step %d learning rate %.4f step-time %.2f" %
                      (model.global_step.eval(), model.learning_rate.eval(), step_time))
                print("   perplexity emotion %s: %.2f\n   perplexity emotion %s: %.2f\n"
                      "   perplexity emotion %s: %.2f\n   perplexity emotion %s: %.2f\n"
                      "   perplexity emotion %s: %.2f\n   perplexity emotion %s: %.2f\n"
                      % (EMOTION_TYPE[0], perplexity[0], EMOTION_TYPE[1], perplexity[1],
                         EMOTION_TYPE[2], perplexity[2], EMOTION_TYPE[3], perplexity[3],
                         EMOTION_TYPE[4], perplexity[4], EMOTION_TYPE[5], perplexity[5]))

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)

                previous_losses.append(loss)

                checkpoint_path = os.path.join(FLAGS.model_dir, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                for bucket_id in range(len(BUCKETS)):
                    encoder_inputs, decoder_inputs_dict, target_weights_dict = model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs_dict, target_weights_dict, bucket_id, True)

                    print("  eval: bucket %d" % (bucket_id))
                    for j in range(len(EMOTION_TYPE)):
                        eval_ppx = math.exp(eval_loss[j]) if eval_loss[j] < 300 else float('inf')
                        print("      emotion %s perplexity %.2f" % (EMOTION_TYPE[j], eval_ppx))

                sys.stdout.flush()


def train():
    print("Preparing dialog data in %s" % FLAGS.data_dir)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(FLAGS.data_dir, FLAGS.vocab_size)

    with tf.Session() as sess:
        print ("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, forward_only=False)

        print ("Reading development and training data (limit:%d)." % FLAGS.max_train_data_size)
        dev_set = read_data(dev_data)
        train_set = read_data(train_data, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(BUCKETS))]
        train_total_size = float(sum(train_bucket_sizes))

        train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, forward_only=False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f" %
                       (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)

                previous_losses.append(loss)

                checkpoint_path = os.path.join(FLAGS.model_dir, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                for bucket_id in range(len(BUCKETS)):
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

                sys.stdout.flush()