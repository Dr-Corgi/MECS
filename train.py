import tensorflow as tf

from tf_chatbot.lib.train import train, train_one2many

def main(_):
    train_one2many()

if __name__ == "__main__":
    tf.app.run()
