from __future__ import print_function

from keras import activations, initializers
from keras import regularizers
from keras.engine import Layer
from keras.engine.topology import Node, InputLayer

import numpy as np
import keras.backend as K
import tensorflow as tf


def GraphInput(name=None, dtype=K.floatx(), sparse=False,
               tensor=None):
    """ TODO: Docstring """
    shape = (None, None)
    input_layer = GraphInputLayer(batch_input_shape=shape,
                                  name=name, sparse=sparse, input_dtype=dtype)
    outputs = input_layer.inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


class GraphInputLayer(InputLayer):
    """ TODO: Docstring """
    def __init__(self, input_shape=None, batch_input_shape=None,
                 input_dtype=None, sparse=False, name=None):
        self.input_spec = None
        self.supports_masking = False
        self.uses_learning_phase = False
        self.trainable = False
        self.built = True

        self.is_placeholder = True
        self.inbound_nodes = []
        self.outbound_nodes = []

        self.trainable_weights = []
        self.non_trainable_weights = []
        self.constraints = {}

        self.sparse = sparse

        if not name:
            prefix = 'input'
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        if not batch_input_shape:
            assert input_shape, 'An Input layer should be passed either a `batch_input_shape` or an `input_shape`.'
            batch_input_shape = (None,) + tuple(input_shape)
        else:
            batch_input_shape = tuple(batch_input_shape)
        if not input_dtype:
            input_dtype = K.floatx()

        self.batch_input_shape = batch_input_shape
        self.input_dtype = input_dtype

        input_tensor = K.placeholder(shape=batch_input_shape,
                                     dtype=input_dtype,
                                     sparse=self.sparse,
                                     name=self.name)

        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)
        shape = input_tensor._keras_shape
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[shape],
             output_shapes=[shape])

    def get_config(self):
        config = {'batch_input_shape': self.batch_input_shape,
                  'input_dtype': self.input_dtype,
                  'sparse': self.sparse,
                  'name': self.name}
        return config


def get_tensor_shape(G):
    return G.dense_shape


def vector_to_adjacency(inputs):
    G, e, dense_shape = inputs
    e = add_epsilon(e)
    A = tf.SparseTensor(indices=G.indices, values=e, dense_shape=dense_shape)
    return A

def vector_to_adjacency_normalized(inputs):
    G, e, dense_shape = inputs
    e = add_epsilon(e)
    A = tf.SparseTensor(indices=G.indices, values=e, dense_shape=dense_shape)
    D = tf.diag(tf.pow(tf.sparse_reduce_sum(A, 1), -1))

    A_ = tf.sparse_tensor_dense_matmul(A, D)

    return add_epsilon(A_)

def add_epsilon(A):
    return A + 0.1

def vector_to_adjacency_sym_normalized(inputs):
    G, e, dense_shape = inputs
    e = add_epsilon(e)
    A = tf.SparseTensor(indices=G.indices, values=e, dense_shape=dense_shape)
    D = tf.diag(tf.pow(tf.sparse_reduce_sum(A, 1), -0.5))

    A_ = tf.sparse_tensor_dense_matmul(A, D)
    A_ = tf.transpose(A_)
    A_ = tf.matmul(A_, D)

    return A_

def vector_to_adjacency_sym_sparse(inputs):
    G, e, dense_shape = inputs
    e = add_epsilon(e)
    A = tf.SparseTensor(indices=G.indices, values=e, dense_shape=dense_shape)
    D = tf.pow(tf.sparse_reduce_sum(A, 1), -0.5)

    #row wise normalization
    Drow = tf.gather(D, G.indices[:, 0])

    #column wise normalization, currently disabled
    #Dcol = tf.gather(D, G.indices[:, 1])

    #multiply values by D
    # e_ = tf.multiply(tf.multiply(Dcol, e), Drow)
    e_ = tf.multiply(e, Drow)
    A_ = tf.SparseTensor(indices=G.indices, values=e_, dense_shape=dense_shape)
    return A_

def vector_to_adjacency_softmax(inputs):
    G, e, dense_shape = inputs
    A = tf.SparseTensor(indices=G.indices, values=e, dense_shape=dense_shape)
    A_ = tf.sparse_softmax(A)
    return A_

def extract_from_adjs(inputs):
    As = inputs
    return tf.stack([A.values for A in As], 1)

def reshape_for_lstm(inputs):
    He = inputs
    return tf.reshape(He, [-1, 12, 20])

class GraphConvolution(Layer):
    """TODO: Docstring"""
    def __init__(self, units, support=1, kernel_initializer='glorot_uniform',
                 activation='linear', weights=None, kernel_regularizer=None,
                 bias_regularizer=None, use_bias=False, self_links=0, first_layer_one_hot=False, **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.units = units  # number of features per node
        self.support = support  # filter support / number of weights
        self.self_links = self_links # 0 for no self_links, 1 to add self links

        self.first_layer_one_hot = first_layer_one_hot

        assert support >= 1

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        self.initial_weights = weights

        # these will be defined during build()
        self.input_dim = None
        self.kernel = None
        self.bias = None

        super(GraphConvolution, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)


    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        self.input_dim = features_shape[1]

        # if no features are present, and we add one_hot as features, the input shape should change
        if self.first_layer_one_hot == True:
            self.input_dim = input_shapes[1][1]

        # use this line to remove self links from first layer
        self.kernel = self.add_weight((self.input_dim * (self.support + self.self_links - self.first_layer_one_hot), self.units),
        # self.kernel = self.add_weight((self.input_dim * (self.support + self.self_links), self.units), #uncomment this line to create weights for the self links in the first layer
                                      initializer=self.kernel_initializer,
                                      name='{}_kernel'.format(self.name),
                                      regularizer=self.kernel_regularizer)
        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        initializer='zero',
                                        name='{}_bias'.format(self.name),
                                        regularizer=self.bias_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


    def call(self, inputs, mask=None):
        features = inputs[0]
        A = inputs[1:]  # list of basis functions

        # convolve
        supports = list()
        for i in range(self.support):
            if self.first_layer_one_hot == True:
                supports.append(A[i])
            else:
                supports.append(K.dot(A[i], features))
        if self.self_links == True and self.first_layer_one_hot == False:
            supports.append(features)

        # uncomment this if you uncommented the self.kernel line above, this adds a sparse identity matrix to the input
        # if self.first_layer_one_hot == True:
        #     print('appending eye')
        #     eye = tf.SparseTensor(indices=np.array([range(self.input_dim), range(self.input_dim)]).T, values=np.ones(self.input_dim).astype('float32'), dense_shape=(self.input_dim, self.input_dim))
        #     supports.append(eye)

        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.use_bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            # 'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            # 'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            # 'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            # 'kernel_constraint': constraints.serialize(self.kernel_constraint),
            # 'bias_constraint': constraints.serialize(self.bias_constraint)
            }
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
