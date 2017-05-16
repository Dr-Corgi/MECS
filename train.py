import tensorflow as tf
from tf_chatbot.lib.train import train

def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()
