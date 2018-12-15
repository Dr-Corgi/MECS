import tensorflow as tf

TEST_DATASET_PATH = './tf_chatbot/data/test/test_data.json'
SAVE_DATA_DIR = './tf_chatbot/'

tf.app.flags.DEFINE_string('data_dir', SAVE_DATA_DIR+'data', 'Data directory')
tf.app.flags.DEFINE_string('model_dir', SAVE_DATA_DIR+'nn_models', 'Train directory')
tf.app.flags.DEFINE_string('results_dir', SAVE_DATA_DIR+'results', 'Train directory')

tf.app.flags.DEFINE_float('learning_rate', 0.5, 'Learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.99, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size to use during training')
tf.app.flags.DEFINE_integer('epoch_size', 20, 'Size of epoch')

tf.app.flags.DEFINE_integer('vocab_size', 20000, 'Dialog vocabulary size')
tf.app.flags.DEFINE_integer('size', 128, 'size of each model layer')
tf.app.flags.DEFINE_integer('num_layers', 1, 'Numbers of layers in the model')
tf.app.flags.DEFINE_integer('beam_search_size', 5, 'Size of beam search op')

tf.app.flags.DEFINE_integer('max_train_data_size', 0, 'Limit on the size of training data (0: no limit)')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 2500, 'How many training steps to do per checkpoint')

tf.app.flags.DEFINE_boolean('use_sample', False, 'use sample while generating')
tf.app.flags.DEFINE_boolean('use_beam_search', True, 'use beam search while generating')

FLAGS = tf.app.flags.FLAGS

BUCKETS = [(10, 15), (20, 25), (40, 50)]
#BUCKETS = [(10, 15)]

EMOTION_TYPE = {0: "Other", 1: "Like", 2:"Sadness", 3:"Disgust", 4:"Anger", 5:"Happiness"}
