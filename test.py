import tensorflow as tf

from tf_chatbot.lib.predict import predict, predict_one2many

def main(_):
    predict_one2many()

if __name__ == "__main__":
    tf.app.run()