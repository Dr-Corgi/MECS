import tensorflow as tf
from tf_chatbot.lib.generate import predict

def main(_):
    predict()

if __name__ == '__main__':
    tf.app.run()