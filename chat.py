import tensorflow as tf

from tf_chatbot.lib.chat import chat

def main(_):
    chat()

if __name__ == "__main__":
    tf.app.run()