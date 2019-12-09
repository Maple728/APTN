#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/28 10:23
@desc:
"""
import math
import tensorflow as tf

from models.base_model import BaseModel
from lib.utils import tensordot, get_tf_loss_function


class APTN(BaseModel):

    def __init__(self, model_config):
        # base setting
        self.float_dtype = tf.float32
        self.activate_func = tf.nn.relu

        # config setting
        self.config = model_config
        self.T = model_config.get('T')
        self.n = model_config.get('n')
        self.D = model_config.get('D')
        self.m = model_config.get('m')
        self.v = model_config.get('v')
        self.horizon = model_config.get('horizon')
        self.loss_tf = get_tf_loss_function(model_config.get('loss_function'))
        self._hw = self.T // 2

        # build model
        # ------------ placeholder building -----------
        with tf.name_scope('inputs'):
            # train placeholder
            self.dropout_rate_ph = tf.placeholder(self.float_dtype, name='dropout_rate')
            self.lr_ph = tf.placeholder(self.float_dtype, name='lr')
            # data placeholder
            self.X_skip_ph = tf.placeholder(self.float_dtype,
                                            shape=[None, self.n, self.T, self.D])
            self.X_ph = tf.placeholder(self.float_dtype, shape=[None, self.T, self.D], name='X')
            self.Y_ph = tf.placeholder(self.float_dtype, shape=[None, self.horizon, self.D], name='Y')

        # ------------ model body building -----------
        with tf.variable_scope(self.__class__.__name__):
            # input layer
            # shape -> [batch_size, T, v]
            x_input = self.input_layer(self.X_ph)
            # shape -> [batch_size, n, T, v]
            x_skip_input = self.input_layer(self.X_skip_ph)

            # hop-rnn layer
            # shape -> [batch_size, T, m]
            period_outputs = self.hop_rnn_layer(x_skip_input)

            # encoder and decoder layer
            # shape -> [batch_size, T, m]
            en_outputs = self.att_encoder_layer(x_input, period_outputs)
            # shape -> [batch_size, m]
            de_output = self.att_decoder_layer(en_outputs, x_input)
            # shape -> [batch_size, v]
            output_state = self.att_output_layer(en_outputs[:, -1], de_output)

            # generate predictions
            pred_ta = tf.TensorArray(self.float_dtype, size=self.horizon)
            # shape -> [batch_size, hw, D]
            ar_input = self.X_ph[:, -self._hw:]
            for i in range(self.horizon):
                # shape -> [batch_size, D]
                nn_pred = self.nn_horizon_output_layer(output_state, 'nn_horizon' + str(i))
                ar_pred = self.ar_horizon_output_layer(ar_input, 'ar_horizon' + str(i))
                nn_pred += ar_pred

                pred_ta = pred_ta.write(i, nn_pred)

        # shape -> [batch_size, horizon, D]
        self.Y_pred = tf.transpose(pred_ta.stack(), perm=[1, 0, 2])

        # ------------ optimize building -----------
        with tf.name_scope('train'):
            # loss
            self.loss = self.loss_tf(self.Y_pred, self.Y_ph)
            self.optimizer = tf.train.AdamOptimizer(self.lr_ph)
            self.train_op = self.optimizer.minimize(self.loss)

    def train(self, sess, batch_data, **kwargs):
        # get parameter
        lr = kwargs.get('lr')
        dropout_rate = kwargs.get('dropout_rate', 0.0)

        fd = {self.dropout_rate_ph: dropout_rate,
              self.lr_ph: lr,
              self.X_skip_ph: batch_data[0],
              self.X_ph: batch_data[1],
              self.Y_ph: batch_data[2]}
        _, loss, pred, real = sess.run([self.train_op, self.loss, self.Y_pred, self.Y_ph], feed_dict=fd)
        return loss, pred, real

    def predict(self, sess, batch_data, **kwargs):
        fd = {self.dropout_rate_ph: 0.0,
              self.X_skip_ph: batch_data[0],
              self.X_ph: batch_data[1],
              self.Y_ph: batch_data[2]}
        loss, pred, real = sess.run([self.loss, self.Y_pred, self.Y_ph], feed_dict=fd)
        return loss, pred, real

    def visualize_att(self, sess, batch_data, **kwargs):
        fd = {self.dropout_rate_ph: 0.0,
              self.X_skip_ph: batch_data[0],
              self.X_ph: batch_data[1],
              self.Y_ph: batch_data[2]}
        a_ks = sess.run(self.a_ks, feed_dict=fd)
        return a_ks

    def get_weights(self, name, shape, collections=None):
        return tf.get_variable(name, shape=shape, dtype=self.float_dtype,
                               initializer=tf.glorot_normal_initializer(),
                               collections=collections)

    def get_bias(self, name, shape, collections=None):
        return tf.get_variable(name, shape=shape, dtype=self.float_dtype,
                               initializer=tf.constant_initializer(0.1),
                               collections=collections)

    def get_rnn_cell(self, rnn_hid, dropout_rate):
        keep_prob = 1 - dropout_rate
        rnn = tf.nn.rnn_cell.LSTMCell(rnn_hid, initializer=tf.initializers.orthogonal(),
                                      activation=tf.tanh, forget_bias=0.0)
        rnn = tf.nn.rnn_cell.DropoutWrapper(rnn, output_keep_prob=keep_prob)
        return rnn

    def input_layer(self, x, scope='input_layer'):
        """
        :param x: a tensor with shape [..., input_dim]
        :param scope:
        :return: a tensor with shape [..., v]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer_w = self.get_weights('layer_w', shape=[self.D, self.v])
            layer_b = self.get_bias('layer_b', shape=[self.v])
            return self.activate_func(tensordot(x, layer_w) + layer_b)

    def hop_rnn_layer(self, x, scope='hop_rnn'):
        """
        Recurrent-Skip component
        :param x: <batch_size, n, T, D>
        :param scope:
        :return: <batch_size, T, m>
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            skip_rnn = self.get_rnn_cell(self.m, self.dropout_rate_ph)

            # recurrent-skip
            # <Tc> * <batch_size, rnn_hid>
            skip_h_states = tf.TensorArray(tf.float32, size=self.T)
            for t in range(self.T):
                # init cell state
                s_states = skip_rnn.zero_state(tf.shape(x)[0], tf.float32)
                h_state = None
                for i in range(self.n):
                    h_state, s_states = skip_rnn(x[:, i, t], s_states)

                # store h_state
                skip_h_states = skip_h_states.write(t, h_state)

            # <batch_size, T, m>
            skip_outputs = tf.transpose(skip_h_states.stack(), perm=[1, 0, 2])

        return skip_outputs

    def att_encoder_layer(self, en_x, inputs_from_skip, scope='att_encoder'):
        """ RNNs with skip_inputs
        :param en_x: <batch_size, T, v>
        :param inputs_from_skip: <batch_size, T, m>
        :param scope:
        :return: <batch_size, T, m>
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # input: <batch_size, v, T>
            en_input = tf.transpose(en_x, perm=[0, 2, 1])

            en_rnn = self.get_rnn_cell(self.m, self.dropout_rate_ph)

            # attention weights
            en_w = self.get_weights('en_w', shape=[2 * self.m, self.v])
            en_v = self.get_weights('en_v', shape=[3 * self.v, self.v])
            en_u = self.get_weights('en_u', shape=[self.T, 1])
            en_u_b = self.get_bias('en_u_b', shape=[1])
            en_b = self.get_bias('en_b', shape=[self.v])

            # skip weights
            skip_w_tilde = self.get_weights('skip_w_tilde', [self.m, self.v])
            skip_b_tilde = self.get_weights('skip_bias_tilde', shape=[self.v])

            # T * <batch_size, m>
            x_embeds = tf.TensorArray(tf.float32, self.T)
            en_s_state = en_rnn.zero_state(tf.shape(en_input)[0], tf.float32)

            # x_part
            # <batch_size, v>
            x_part = tf.squeeze(tensordot(en_input, en_u) + en_u_b, axis=-1)
            x_part = self.activate_func(x_part)

            for t in range(self.T):
                # <batch_size, m * 2>
                h_s_concat = tf.concat([en_s_state.h, en_s_state.c], axis=1)
                hs_part = self.activate_func(tf.matmul(h_s_concat, en_w))

                skip_part = self.activate_func(tf.matmul(inputs_from_skip[:, t], skip_w_tilde) + skip_b_tilde)

                # <batch_size, v>
                e_ks = tf.matmul(tf.concat([hs_part, x_part, skip_part], axis=-1), en_v) + en_b
                a_ks = tf.nn.softmax(e_ks / math.sqrt(self.v), axis=-1)

                # <batch_size, v>
                # x_t = tf.matmul(tf.expand_dims(en_input[:, :, t], -2), tf.matrix_diag(a_ks))
                x_t = en_input[:, :, t] * a_ks

                # rnn input
                x_t_tilde = tf.concat([x_t, inputs_from_skip[:, t]], axis=1)

                en_h_state, en_s_state = en_rnn(x_t_tilde, en_s_state)

                # store h_state
                x_embeds = x_embeds.write(t, en_h_state)
                self._a_ks = a_ks

            # <batch_size, T, m>
            en_outputs = tf.transpose(x_embeds.stack(), perm=[1, 0, 2])

        return en_outputs

    def att_decoder_layer(self, inputs_from_en, de_input, scope='att_decoder'):
        """
            Attentions decoder.
        :param inputs_from_en: <batch_size, T, m>. The hidden states encoded.
        :param de_input: <batch_size, T, v>
        :param scope:
        :return: The last decoded hidden states. <batch_size, v>
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            de_rnn = self.get_rnn_cell(self.m, self.dropout_rate_ph)

            de_s_state = de_rnn.zero_state(tf.shape(inputs_from_en)[0], tf.float32)
            de_h_state = None

            # attention weights
            de_v = self.get_weights('de_v', shape=[2 * self.T, self.T])
            de_w = self.get_weights('de_w', shape=[2 * self.m, self.T])
            de_u = self.get_weights('de_u', shape=[self.m, 1])
            de_u_b = self.get_bias('de_u_b', shape=[1])
            de_b = self.get_bias('de_b', shape=[self.T])

            # extra inputs weights
            de_w_tilde = self.get_weights('de_w_tilde', shape=[self.m + self.v, self.m])
            de_b_tilde = self.get_bias('de_bias_tilde', shape=[self.m])

            # shape -> [batch_size, T]
            en_part = tf.squeeze(tensordot(inputs_from_en, de_u) + de_u_b, axis=-1)
            en_part = self.activate_func(en_part)

            for t in range(self.T):
                # <batch_size, 2 * m>
                d_s_concat = tf.concat([de_s_state.h, de_s_state.c], axis=1)
                # shape -> [batch_size, T]
                ds_part = self.activate_func(tf.matmul(d_s_concat, de_w))

                # shape -> [batch_size, T]
                l_ks = tf.matmul(tf.concat([ds_part, en_part], axis=-1), de_v) + de_b
                b_ks = tf.nn.softmax(l_ks / math.sqrt(self.T), axis=-1)

                # <batch_size, T, m>
                c_t = inputs_from_en * tf.expand_dims(b_ks, -1)
                # <batch_size, m>
                c_t = tf.reduce_sum(c_t, axis=-2)

                y_tilde = tf.concat([c_t, de_input[:, t]], axis=-1)
                # shape -> [batch_size, m]
                y_tilde = self.activate_func(tf.matmul(y_tilde, de_w_tilde) + de_b_tilde)

                de_h_state, de_s_state = de_rnn(y_tilde, de_s_state)

        # <batch_size, m>
        return de_h_state

    def att_output_layer(self, input_from_en, input_from_de, scope='att_output'):
        """
        :param input_from_en: [bath_size, m]
        :param input_from_de: [batch_size, m]
        :param scope:
        :return: [batch_size, v]
        """

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            infer_v_y = self.get_weights('v_y', [self.m, self.v])
            infer_w_y = self.get_weights('w_y', [2 * self.m, self.m])

            infer_b_w = self.get_bias('bias_w', shape=[self.m])
            infer_b_v = self.get_bias('bias_v', shape=[self.v])

            # <batch_size, m * 2>
            infer_concat = tf.concat([input_from_en, input_from_de], axis=1)
            # <batch_size, m>
            infer_output = self.activate_func(tf.matmul(infer_concat, infer_w_y) + infer_b_w)
            infer_output = tf.nn.dropout(infer_output, keep_prob=1 - self.dropout_rate_ph)
            # shape -> [batch_size, v]
            infer_output = tf.matmul(infer_output, infer_v_y) + infer_b_v
        return infer_output

    def nn_horizon_output_layer(self, x, scope='nn_horizon'):
        """ Generate a horizon prediction for the neural network part.
        :param x: [..., D]
        :param scope:
        :return: the prediction with shape [..., D]
        """
        keep_prob = 1.0 - self.dropout_rate_ph
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            horizon_w = self.get_weights('horizon_w', shape=[self.v, self.D])
            horizon_b = self.get_bias('horizon_b', shape=[self.D])

            horizon_v = self.get_weights('horizon_v', shape=[self.D, self.D])
            horizon_v_b = self.get_bias('horizon_v_b', shape=[self.D])

            y_pred = self.activate_func(tensordot(x, horizon_w) + horizon_b)
            y_pred = tf.nn.dropout(y_pred, keep_prob=keep_prob)

            # shape -> [..., D]
            y_pred = tensordot(y_pred, horizon_v) + horizon_v_b

        return y_pred

    def ar_horizon_output_layer(self, x, scope='ar_horizon'):
        """ Generate a horizon prediction for the auto regressive part.
        :param x: a tensor with shape [..., hw, D]
        :param scope:
        :return: the prediction with shape [..., D]
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # weights
            ar_w = self.get_weights('ar_w', shape=[self._hw, 1])
            ar_b = self.get_bias('ar_b', shape=[1])

            # shape -> [..., hw, D]
            ar_output = x * ar_w
            # [..., D]
            ar_output = tf.reduce_sum(ar_output, axis=-2) + ar_b

        return ar_output
