import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell,LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

class CustomCell(RNNCell):

    def __init__(self, num_units, encoder_outp, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self.encoder_outp = encoder_outp
        self._forget_bias = forget_bias
        self._activation = activation
        self.attn_Wc = tf.get_variable('attn_Wc', dtype=tf.float32,
                                      initializer=tf.random_uniform([num_units*2, num_units], -1.0, 1.0))
        self.attn_Wa = tf.get_variable('attn_Wa', dtype=tf.float32,
                                      initializer=tf.random_uniform([num_units, num_units],-1.0, 1.0))

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            concat = _linear([inputs, h], 4 * self._num_units, False)
            i, j, f, o = tf.split(concat,4,1)

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                   self._activation(j))

            new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            #print new_h.shape

            # attention
            with tf.variable_scope("attention"):
                encoder_flat = tf.transpose(self.encoder_outp,[1,0,2])
                scores_flat = tf.matmul(encoder_flat, tf.expand_dims(tf.matmul(new_h, self.attn_Wa),2))
                score_softmax = tf.nn.softmax(scores_flat, dim=1)
                c = tf.reduce_sum(encoder_flat * score_softmax, axis=1)
                h_hat = tf.nn.tanh(tf.matmul(tf.concat([c, new_h],axis=1), self.attn_Wc))

            new_state = LSTMStateTuple(c, h_hat)

            return h_hat, new_state