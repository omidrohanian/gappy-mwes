from __future__ import print_function

import keras, random, os, sys
from keras import initializers
import keras.backend as K
import tensorflow as tf
from keras import activations, constraints, initializers, regularizers
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.initializers import *
from keras.engine import Layer
from keras.layers.wrappers import Wrapper, TimeDistributed

import numpy as np
import tensorflow as tf

def Highway(value, n_layers, activation="tanh", gate_bias=-2):  
    """ Highway layers:
        a minus bias means the network is biased towards carry behavior in the initial stages"""
    dim = K.int_shape(value)[-1]
    bias = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):
        T_gate = Dense(units=dim, bias_initializer=bias, activation="sigmoid")(value)
        C_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(T_gate)
        transform = Dense(units=dim, activation=activation)(value)
        transform_gated = Multiply()([T_gate, transform])
        carry_gated = Multiply()([C_gate, value])
        value = Add()([transform_gated, carry_gated])
    return value

#-----------------------------------------------------------------------------------#

# from http://curlba.sh/jhartog/Mihail/blob/f19c455dcd804536a5895b5d5494119b4315e23b/lib/python2.7/site-packages/tensorflow/python/ops/gen_manip_ops.py

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
def roll(input, shift, axis, name=None):
  r"""Rolls the elements of a tensor along an axis.

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  _ctx = _context._context
  if _ctx is None :#or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "Roll", input=input, shift=shift, axis=axis, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "Tshift", _op.get_attr("Tshift"),
              "Taxis", _op.get_attr("Taxis"))
    _execute.record_gradient(
      "Roll", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name, "Roll", name,
        _ctx._post_execution_callbacks, input, shift, axis)
      return _result
    #except _core._FallbackException:
    #  return roll_eager_fallback(
    #      input, shift, axis, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)

#-----------------------------------------------------------------------------------#

# https://github.com/JHart96/keras_gcn_sequence_labelling
class SpectralGraphConvolution(Layer):
    def __init__(self, output_dim,
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None,
                 b_regularizer=None, bias=True, 
                 self_links=True, consecutive_links=True, 
                 backward_links=True, edge_weighting=False, **kwargs):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node

        self.self_links = self_links
        self.consecutive_links = consecutive_links
        self.backward_links = backward_links
        self.edge_weighting = edge_weighting

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights

        self.input_dim = None
        self.W = None
        self.b = None
        self.num_nodes = None
        self.num_features = None
        self.num_relations = None
        self.num_adjacency_matrices = None

        super(SpectralGraphConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (None, features_shape[1], self.output_dim)
        return output_shape

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 3
        self.input_dim = features_shape[1]
        self.num_nodes = features_shape[1]
        self.num_features = features_shape[2]
        self.num_relations = len(input_shapes) - 1

        self.num_adjacency_matrices = self.num_relations

        if self.consecutive_links:
            self.num_adjacency_matrices += 1

        if self.backward_links:
            self.num_adjacency_matrices *= 2

        if self.self_links:
            self.num_adjacency_matrices += 1

        self.W = []
        self.W_edges = []
        for i in range(self.num_adjacency_matrices):
            self.W.append(self.add_weight((self.num_features, self.output_dim), # shape: (num_features, output_dim)
                                                    initializer=self.init,
                                                    name='{}_W_rel_{}'.format(self.name, i),
                                                    regularizer=self.W_regularizer))

            if self.edge_weighting:
                self.W_edges.append(self.add_weight((self.input_dim, self.num_features), # shape: (num_features, output_dim)
                                                        initializer='ones',
                                                        name='{}_W_edge_{}'.format(self.name, i),
                                                        regularizer=self.W_regularizer))

        self.b = self.add_weight((self.input_dim, self.output_dim),
                                        initializer='random_uniform',
                                        name='{}_b'.format(self.name),
                                        regularizer=self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(SpectralGraphConvolution, self).build(input_shapes)

    def call (self, inputs, mask=None):
        features = inputs[0] # Shape: (None, num_nodes, num_features)
        A = inputs[1:]  # Shapes: (None, num_nodes, num_nodes)

        eye = A[0] * K.zeros(self.num_nodes, dtype='float32') + K.eye(self.num_nodes, dtype='float32')

        # eye = K.eye(self.num_nodes, dtype='float32')

        if self.consecutive_links:
            #shifted = tf.manip.roll(eye, shift=1, axis=0)
            #shifted = tf.roll(eye, shift=1, axis=0)
            #shifted = roll(eye, shift=1, axis=0)
            #####################################################
            eye_len = eye.get_shape().as_list()[0] 
            #shifted = tf.concat((eye, eye), axis=0)
            #shifted = tf.concat((eye[eye_len-1: , :], eye[:eye_len-1 , :]), axis=0)
            shifted = tf.concat((eye[-1: , :], eye[:-1 , :]), axis=0)

            #####################################################

            A.append(shifted)

        if self.backward_links:
            for i in range(len(A)):
                A.append(K.permute_dimensions(A[i], [0, 2, 1]))

        if self.self_links:
            A.append(eye)

        AHWs = list()
        for i in range(self.num_adjacency_matrices):
            if self.edge_weighting:
                features *= self.W_edges[i]
            HW = K.dot(features, self.W[i]) # Shape: (None, num_nodes, output_dim)
            AHW = K.batch_dot(A[i], HW) # Shape: (None, num_nodes, num_features)
            AHWs.append(AHW)
        AHWs_stacked = K.stack(AHWs, axis=1) # Shape: (None, num_supports, num_nodes, num_features)
        output = K.max(AHWs_stacked, axis=1) # Shape: (None, num_nodes, output_dim)

        if self.bias:
            output += self.b
        return self.activation(output)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'num_bases': self.num_bases,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



#-----------------------------------------------------------------------------------#

# The self-attention mechanism used
# based on: https://github.com/Lsdefine/attention-is-all-you-need-keras

class LayerNormalization(Layer):

    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super().build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention():
# =============================================================================  
#Scoring the query q vs keys k and
# Scaling the dot product attention
# =============================================================================
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
# =============================================================================
#    creating a self attention sublayer
    
#   mode 0 - big martixes, faster; mode 1 - more clear implementation
# =============================================================================
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization()
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])  
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)  
                
            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)   
                ks = self.ks_layers[i](k) 
                vs = self.vs_layers[i](v) 
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads)
            attn = Concatenate()(attns)

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn