# %%
#! -*- coding: utf-8 -*-
# refer: https://kexue.fm/archives/5112

from keras import activations
from keras import backend as K
from keras.engine.topology import Layer
import sys

# Keeps unique .h5 file
Process = sys.argv[1]

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x


#define our own softmax function instead of K.softmax
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


#A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


# %%
# from keras.layers import K, Activation
# from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D
# from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from vendor.Capsule.Capsule_Keras import *

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28

max_features = 20000
maxlen = 4500
embed_size = 256

def get_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    x = Bidirectional(GRU(gru_len,
                          activation='relu',
                          dropout=dropout_p,
                          recurrent_dropout=dropout_p,
                          return_sequences=True))(embed_layer)
    capsule = Capsule(
        num_capsule=Num_capsule,
        dim_capsule=Dim_capsule,
        routings=Routings,
        share_weights=True)(x)

    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    capsule = LeakyReLU()(capsule)

    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


def load_imdb(maxlen=4500):
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=maxlen)
    x_train = pd.read_csv("~/capsnet/ISOTtrainFIXED.csv",encoding='latin1')
    x_test = pd.read_csv("~/capsnet/ISOTtestFIXED.csv",encoding='latin1')
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    tokenizer = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer.fit_on_texts(texts = x_train['body_text'])
    X = tokenizer.texts_to_sequences(texts = x_train['body_text'])
    X = pad_sequences(sequences = X, maxlen = maxlen)
    print(X.shape)
    Xt = tokenizer.texts_to_sequences(texts=x_test['body_text'])
    Xt = pad_sequences(sequences = Xt, maxlen = maxlen)
    # x_train = sequence.pad_sequences(x_train['body_text'], maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test['body_text'], maxlen=maxlen)
    return X, y_train, Xt, y_test


def main():
    x_train, y_train, x_test, y_test = load_imdb()

    model = get_model()

    batch_size = 64
    epochs = 5

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test))
    model.save(f"/home/jnguye30/capsnet/ISOT/ISOT{Process}.h5")


if __name__ == '__main__':
    main()


# %%
import keras.backend as K
import tensorflow as tf
from keras import initializers, layers


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
    inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
    output: shape=[dim_1, ..., dim_{n-1}]
    """

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """

    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_vector] and output shape = \
    [None, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.
    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
            initializer=self.kernel_initializer,
            name='W')

        # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_vector]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        """
        # Compute `inputs * W` by expanding the first dim of W. More time-consuming and need batch_size.
        # Now W has shape  = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
        w_tiled = K.tile(K.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])
        # Transformed vectors, inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = K.batch_dot(inputs_tiled, w_tiled, [4, 3])
        """
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0. This is faster but requires Tensorflow.
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
        """
        # Routing algorithm V1. Use tf.while_loop in a dynamic way.
        def body(i, b, outputs):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
            b = b + K.sum(inputs_hat * outputs, -1, keepdims=True)
            return [i-1, b, outputs]
        cond = lambda i, b, inputs_hat: i > 0
        loop_vars = [K.constant(self.num_routing), self.bias, K.sum(inputs_hat, 1, keepdims=True)]
        _, _, outputs = tf.while_loop(cond, body, loop_vars)
        """
        # Routing algorithm V2. Use iteration. V2 and V1 both work without much difference on performance
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                # self.bias = K.update_add(self.bias, K.sum(inputs_hat * outputs, [0, -1], keepdims=True))
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
                # tf.summary.histogram('BigBee', self.bias)  # for debugging
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])


def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding, name):
    """
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = layers.Conv1D(filters=dim_vector * n_channels, kernel_size=kernel_size, strides=strides, padding=padding, name=name)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    return layers.Lambda(squash)(outputs)

# %%
# from keras.layers import K, Activation
# from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D, concatenate,LSTM,Conv1D
# from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from vendor.Capsule.Capsule_Keras import *

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
Dim_capsule1 = 32
dropout_p = 0.25
rate_drop_dense = 0.28

max_features = 20000
maxlen = 1000
embed_size = 256

def get_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    x = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer)
    x = Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # x = SpatialDropout1D(rate_drop_dense)(x)
    x = Dropout(0.5)(x)
    primary_caps = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps")
    category_caps = CapsuleLayer(num_capsule=1, dim_vector=16, num_routing=3, name='category_caps')(primary_caps)
    out_caps = Length(name='out_caps')(category_caps)
    # # x = Bidirectional(GRU(gru_len,
    # #                       activation='relu',
    # #                       dropout=dropout_p,
    # #                       recurrent_dropout=dropout_p,
    # #                       return_sequences=True))(embed_layer)
    # x = Conv1D(filters=512, kernel_size=4, padding="valid")(x)
    # x = Dropout(0.7)(x)
    # x = Conv1D(filters=256, kernel_size=4, padding="valid")(x)
    # # x = Bidirectional(LSTM(gru_len, return_sequences=False))(x)
    # capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,activation='relu',
    #                   share_weights=True)(x)
    # # capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule1, routings=Routings,activation='relu',
    # #                   share_weights=True)(x)

    # # capsule = concatenate([capsule, capsule1], axis=-1)

    # capsule = Flatten()(capsule)
    # capsule = Dropout(dropout_p)(capsule)

    # # capsule = LeakyReLU()(capsule)

    # # output = Dense(1, activation='sigmoid')(x)
    # output = Dense(1, activation='sigmoid')(out_caps)
    model = Model(inputs=input1, outputs=out_caps)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


def load_imdb(maxlen=1000):
    x_train = pd.read_csv("~/capsnet/ISOTtrainFIXED.csv",encoding='latin1')
    x_test = pd.read_csv("~/capsnet/ISOTtestFIXED.csv",encoding='latin1')
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    tokenizer = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer.fit_on_texts(texts = x_train['body_text'])
    X = tokenizer.texts_to_sequences(texts = x_train['body_text'])
    X = pad_sequences(sequences = X, maxlen = maxlen)
    print(X.shape)
    Xt = tokenizer.texts_to_sequences(texts=x_test['body_text'])
    Xt = pad_sequences(sequences = Xt, maxlen = maxlen)
    # x_train = sequence.pad_sequences(x_train['body_text'], maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test['body_text'], maxlen=maxlen)
    return X, y_train, Xt, y_test


def main():
    x_train, y_train, x_test, y_test = load_imdb()

    model = get_model()

    batch_size = 64
    epochs = 5

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test))
    model.save(f"/home/jnguye30/capsnet/ISOT/ISOT{Process}.h5")


if __name__ == '__main__':
    main()


# %%
# from keras.layers import K, Activation
# from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D, concatenate,LSTM,Conv1D
# from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from vendor.Capsule.Capsule_Keras import *

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
Dim_capsule1 = 32
dropout_p = 0.25
rate_drop_dense = 0.28

max_features = 20000
maxlen = 1000
embed_size = 256

def get_model():
    input1 = Input(shape=(maxlen,))
    input2 = Input(shape = (maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer2 = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input2)

    x = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer)
    x1 = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer2)
    x = concatenate([x,x1], axis=-1)
    x = Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # x = SpatialDropout1D(rate_drop_dense)(x)
    x = Dropout(0.5)(x)
    primary_caps = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps")
    primary_caps2 = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps2")
    primaryconcat = concatenate([primary_caps,primary_caps2], axis=-1)
    category_caps = CapsuleLayer(num_capsule=1, dim_vector=16, num_routing=3, name='category_caps')(primaryconcat)
    out_caps = Length(name='out_caps')(category_caps)
    # # x = Bidirectional(GRU(gru_len,
    # #                       activation='relu',
    # #                       dropout=dropout_p,
    # #                       recurrent_dropout=dropout_p,
    # #                       return_sequences=True))(embed_layer)
    # x = Conv1D(filters=512, kernel_size=4, padding="valid")(x)
    # x = Dropout(0.7)(x)
    # x = Conv1D(filters=256, kernel_size=4, padding="valid")(x)
    # # x = Bidirectional(LSTM(gru_len, return_sequences=False))(x)
    # capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,activation='relu',
    #                   share_weights=True)(x)
    # # capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule1, routings=Routings,activation='relu',
    # #                   share_weights=True)(x)

    # # capsule = concatenate([capsule, capsule1], axis=-1)

    # capsule = Flatten()(capsule)
    # capsule = Dropout(dropout_p)(capsule)

    # # capsule = LeakyReLU()(capsule)

    # # output = Dense(1, activation='sigmoid')(x)
    # output = Dense(1, activation='sigmoid')(out_caps)
    model = Model(inputs=[input1,input2], outputs=out_caps)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


def load_imdb(maxlen=1000):
    x_train = pd.read_csv("~/capsnet/ISOTtrainFIXED.csv",encoding='latin1')
    x_test = pd.read_csv("~/capsnet/ISOTtestFIXED.csv",encoding='latin1')
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    tokenizer = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer.fit_on_texts(texts = x_train['body_text'])
    X = tokenizer.texts_to_sequences(texts = x_train['body_text'])
    X = pad_sequences(sequences = X, maxlen = maxlen)
    print(X.shape)
    Xt = tokenizer.texts_to_sequences(texts=x_test['body_text'])
    Xt = pad_sequences(sequences = Xt, maxlen = maxlen)
    # x_train = sequence.pad_sequences(x_train['body_text'], maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test['body_text'], maxlen=maxlen)
    tokenizer_title = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer_title.fit_on_texts(texts = x_train['title'])
    X_title = tokenizer_title.texts_to_sequences(texts = x_train['title'])
    X_title = pad_sequences(sequences = X_title, maxlen = maxlen)
    Xt_title = tokenizer_title.texts_to_sequences(texts=x_test['title'])
    Xt_title = pad_sequences(sequences = Xt_title, maxlen = maxlen)
    return X,X_title, y_train, Xt,Xt_title, y_test


def main():
    x_train,x_train1, y_train, x_test,x_test1, y_test = load_imdb()

    model = get_model()

    batch_size = 64
    epochs = 5

    model.fit([x_train,x_train1], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([x_test,x_test1], y_test))

    # huh? Never caused an error for recovery run
    model.save(f"/home/jnguye30/capsnet/ISOT/ISOT{Process}.h5")


if __name__ == '__main__':
    main()

# %%
# from keras.layers import K, Activation
# from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D, concatenate,LSTM,Conv1D
# from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from vendor.Capsule.Capsule_Keras import *

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
Dim_capsule1 = 32
dropout_p = 0.25
rate_drop_dense = 0.28

max_features = 20000
maxlen = 1000
embed_size = 256

def get_model():
    input1 = Input(shape=(maxlen,))
    input2 = Input(shape = (maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer2 = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input2)

    x = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x)
    x1 = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer2)
    x1 = Dropout(0.5)(x1)
    x1 = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x1)
    x = concatenate([x,x1], axis=-1)
    x = Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # x = SpatialDropout1D(rate_drop_dense)(x)
    x = Dropout(0.5)(x)
    primary_caps = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps")
    primary_caps2 = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps2")
    primaryconcat = concatenate([primary_caps,primary_caps2], axis=-1)
    category_caps = CapsuleLayer(num_capsule=1, dim_vector=16, num_routing=3, name='category_caps')(primaryconcat)
    out_caps = Length(name='out_caps')(category_caps)
    # # x = Bidirectional(GRU(gru_len,
    # #                       activation='relu',
    # #                       dropout=dropout_p,
    # #                       recurrent_dropout=dropout_p,
    # #                       return_sequences=True))(embed_layer)
    # x = Conv1D(filters=512, kernel_size=4, padding="valid")(x)
    # x = Dropout(0.7)(x)
    # x = Conv1D(filters=256, kernel_size=4, padding="valid")(x)
    # # x = Bidirectional(LSTM(gru_len, return_sequences=False))(x)
    # capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,activation='relu',
    #                   share_weights=True)(x)
    # # capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule1, routings=Routings,activation='relu',
    # #                   share_weights=True)(x)

    # # capsule = concatenate([capsule, capsule1], axis=-1)

    # capsule = Flatten()(capsule)
    # capsule = Dropout(dropout_p)(capsule)

    # # capsule = LeakyReLU()(capsule)

    # # output = Dense(1, activation='sigmoid')(x)
    # output = Dense(1, activation='sigmoid')(out_caps)
    model = Model(inputs=[input1,input2], outputs=out_caps)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


def load_imdb(maxlen=1000):
    x_train = pd.read_csv("~/capsnet/ISOTtrainFIXED.csv",encoding='latin1')
    x_test = pd.read_csv("~/capsnet/ISOTtestFIXED.csv",encoding='latin1')
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    tokenizer = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer.fit_on_texts(texts = x_train['body_text'])
    X = tokenizer.texts_to_sequences(texts = x_train['body_text'])
    X = pad_sequences(sequences = X, maxlen = maxlen)
    print(X.shape)
    Xt = tokenizer.texts_to_sequences(texts=x_test['body_text'])
    Xt = pad_sequences(sequences = Xt, maxlen = maxlen)
    # x_train = sequence.pad_sequences(x_train['body_text'], maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test['body_text'], maxlen=maxlen)
    tokenizer_title = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer_title.fit_on_texts(texts = x_train['title'])
    X_title = tokenizer_title.texts_to_sequences(texts = x_train['title'])
    X_title = pad_sequences(sequences = X_title, maxlen = maxlen)
    Xt_title = tokenizer_title.texts_to_sequences(texts=x_test['title'])
    Xt_title = pad_sequences(sequences = Xt_title, maxlen = maxlen)
    return X,X_title, y_train, Xt,Xt_title, y_test


def main():
    x_train,x_train1, y_train, x_test,x_test1, y_test = load_imdb()

    model = get_model()

    batch_size = 64
    epochs = 5

    model.fit([x_train,x_train1], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([x_test,x_test1], y_test))
    model.save(f"/home/jnguye30/capsnet/ISOT/ISOT{Process}.h5")
    result = model.predict([x_test,x_test1])
    y_pred = np.argmax(result, axis=-1)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))

    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))

if __name__ == '__main__':
    main()

# %%
# from keras.layers import K, Activation
# from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D, concatenate,LSTM,Conv1D
# from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from vendor.Capsule.Capsule_Keras import *

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
Dim_capsule1 = 32
dropout_p = 0.25
rate_drop_dense = 0.28

max_features = 20000
maxlen = 1000
embed_size = 256

def get_model():
    input1 = Input(shape=(maxlen,))
    input2 = Input(shape = (maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer2 = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input2)

    x = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x)
    x = Dropout(0.5)(x)
    # x = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x)
    x1 = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer2)
    x1 = Dropout(0.5)(x1)
    x1 = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x1)
    x1 = Dropout(0.5)(x1)
    # x1 = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x1)
    x = concatenate([x,x1], axis=-1)
    x = Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # x = SpatialDropout1D(rate_drop_dense)(x)
    x = Dropout(0.5)(x)
    primary_caps = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps")
    primary_caps2 = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps2")
    primaryconcat = concatenate([primary_caps,primary_caps2], axis=-1)
    category_caps = CapsuleLayer(num_capsule=1, dim_vector=16, num_routing=3, name='category_caps')(primaryconcat)
    out_caps = Length(name='out_caps')(category_caps)
    # # x = Bidirectional(GRU(gru_len,
    # #                       activation='relu',
    # #                       dropout=dropout_p,
    # #                       recurrent_dropout=dropout_p,
    # #                       return_sequences=True))(embed_layer)
    # x = Conv1D(filters=512, kernel_size=4, padding="valid")(x)
    # x = Dropout(0.7)(x)
    # x = Conv1D(filters=256, kernel_size=4, padding="valid")(x)
    # # x = Bidirectional(LSTM(gru_len, return_sequences=False))(x)
    # capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,activation='relu',
    #                   share_weights=True)(x)
    # # capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule1, routings=Routings,activation='relu',
    # #                   share_weights=True)(x)

    # # capsule = concatenate([capsule, capsule1], axis=-1)

    # capsule = Flatten()(capsule)
    # capsule = Dropout(dropout_p)(capsule)

    # # capsule = LeakyReLU()(capsule)

    # # output = Dense(1, activation='sigmoid')(x)
    # output = Dense(1, activation='sigmoid')(out_caps)
    model = Model(inputs=[input1,input2], outputs=out_caps)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


def load_imdb(maxlen=1000):
    x_train = pd.read_csv("~/capsnet/train.csv",encoding='latin1')
    x_test = pd.read_csv("~/capsnet/test.csv",encoding='latin1')
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    tokenizer = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer.fit_on_texts(texts = x_train['body_text'])
    X = tokenizer.texts_to_sequences(texts = x_train['body_text'])
    X = pad_sequences(sequences = X, maxlen = maxlen)
    print(X.shape)
    Xt = tokenizer.texts_to_sequences(texts=x_test['body_text'])
    Xt = pad_sequences(sequences = Xt, maxlen = maxlen)
    # x_train = sequence.pad_sequences(x_train['body_text'], maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test['body_text'], maxlen=maxlen)
    tokenizer_title = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer_title.fit_on_texts(texts = x_train['title'])
    X_title = tokenizer_title.texts_to_sequences(texts = x_train['title'])
    X_title = pad_sequences(sequences = X_title, maxlen = maxlen)
    Xt_title = tokenizer_title.texts_to_sequences(texts=x_test['title'])
    Xt_title = pad_sequences(sequences = Xt_title, maxlen = maxlen)
    return X,X_title, y_train, Xt,Xt_title, y_test


def main():
    x_train,x_train1, y_train, x_test,x_test1, y_test = load_imdb()

    model = get_model()

    batch_size = 64
    epochs = 5

    model.fit([x_train,x_train1], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([x_test,x_test1], y_test))
    model.save(f"/home/jnguye30/capsnet/ISOT/ISOT{Process}.h5")
    result = model.predict([x_test,x_test1])
    # print(result)
    y_pred = []
    for i in result:
      if i>=0.5:
        y_pred.append(1)
      else:
        y_pred.append(0)

    # y_pred = np.argmax(result, axis=-1)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, y_pred))

    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))

if __name__ == '__main__':
    main()

# %%
# from keras.layers import K, Activation
# from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D, concatenate,LSTM,Conv1D
# from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from vendor.Capsule.Capsule_Keras import *

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
Dim_capsule1 = 32
dropout_p = 0.25
rate_drop_dense = 0.28

max_features = 20000
maxlen = 1000
embed_size = 256

def get_model():
    input1 = Input(shape=(maxlen,))
    input2 = Input(shape = (maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer2 = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input2)

    x = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x)
    x = Dropout(0.5)(x)
    # x = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x)
    x1 = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer2)
    x1 = Dropout(0.5)(x1)
    x1 = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x1)
    x1 = Dropout(0.5)(x1)
    # x1 = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x1)
    x = concatenate([x,x1], axis=-1)
    x = Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # x = SpatialDropout1D(rate_drop_dense)(x)
    x = Dropout(0.5)(x)
    primary_caps = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps")
    primary_caps2 = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps2")
    primaryconcat = concatenate([primary_caps,primary_caps2], axis=-1)
    category_caps = CapsuleLayer(num_capsule=1, dim_vector=16, num_routing=3, name='category_caps')(primaryconcat)
    out_caps = Length(name='out_caps')(category_caps)
    # # x = Bidirectional(GRU(gru_len,
    # #                       activation='relu',
    # #                       dropout=dropout_p,
    # #                       recurrent_dropout=dropout_p,
    # #                       return_sequences=True))(embed_layer)
    # x = Conv1D(filters=512, kernel_size=4, padding="valid")(x)
    # x = Dropout(0.7)(x)
    # x = Conv1D(filters=256, kernel_size=4, padding="valid")(x)
    # # x = Bidirectional(LSTM(gru_len, return_sequences=False))(x)
    # capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,activation='relu',
    #                   share_weights=True)(x)
    # # capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule1, routings=Routings,activation='relu',
    # #                   share_weights=True)(x)

    # # capsule = concatenate([capsule, capsule1], axis=-1)

    # capsule = Flatten()(capsule)
    # capsule = Dropout(dropout_p)(capsule)

    # # capsule = LeakyReLU()(capsule)

    # # output = Dense(1, activation='sigmoid')(x)
    # output = Dense(1, activation='sigmoid')(out_caps)
    model = Model(inputs=[input1,input2], outputs=out_caps)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


def load_imdb(maxlen=1000):
    x_train = pd.read_csv("~/capsnet/ISOTtrainFIXED.csv",encoding='latin1')
    x_test = pd.read_csv("~/capsnet/ISOTtestFIXED.csv",encoding='latin1')
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    tokenizer = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer.fit_on_texts(texts = x_train['body_text'])
    X = tokenizer.texts_to_sequences(texts = x_train['body_text'])
    X = pad_sequences(sequences = X, maxlen = maxlen)
    print(X.shape)
    Xt = tokenizer.texts_to_sequences(texts=x_test['body_text'])
    Xt = pad_sequences(sequences = Xt, maxlen = maxlen)
    # x_train = sequence.pad_sequences(x_train['body_text'], maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test['body_text'], maxlen=maxlen)
    tokenizer_title = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer_title.fit_on_texts(texts = x_train['title'])
    X_title = tokenizer_title.texts_to_sequences(texts = x_train['title'])
    X_title = pad_sequences(sequences = X_title, maxlen = maxlen)
    Xt_title = tokenizer_title.texts_to_sequences(texts=x_test['title'])
    Xt_title = pad_sequences(sequences = Xt_title, maxlen = maxlen)
    return X,X_title, y_train, Xt,Xt_title, y_test


def main():
    x_train,x_train1, y_train, x_test,x_test1, y_test = load_imdb()

    model = get_model()

    batch_size = 64
    epochs = 5

    model.fit([x_train,x_train1], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([x_test,x_test1], y_test))
    model.save(f"/home/jnguye30/capsnet/ISOT/ISOT{Process}.h5")
    result = model.predict([x_test,x_test1])
    print(result)
    y_pred = []
    count1= 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    for i in result:
      if i>=0.5:
        y_pred.append(1)
        if i <=0.6:
          count1+=1
        elif i >0.6 and i<=0.7:
          count2+=1
        elif i>0.7 and i<=0.8:
          count3+=1
        elif i>0.8 and i<=0.9:
          count4+=1
        elif i>0.9 and i<=1.0:
          count5+=1
      else:
        y_pred.append(0)

    print(count1)
    print(count2)
    print(count3)
    print(count4)
    print(count5)
if __name__ == '__main__':
    main()

# %%
# from keras.layers import K, Activation
# from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D, concatenate,LSTM,Conv1D
# from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from vendor.Capsule.Capsule_Keras import *

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
Dim_capsule1 = 32
dropout_p = 0.25
rate_drop_dense = 0.28

max_features = 20000
maxlen = 1000
embed_size = 256

def get_model():
    input1 = Input(shape=(maxlen,))
    input2 = Input(shape = (maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer2 = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input2)

    x = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x)
    x = Dropout(0.5)(x)
    # x = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x)
    x1 = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer2)
    x1 = Dropout(0.5)(x1)
    x1 = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x1)
    x1 = Dropout(0.5)(x1)
    # x1 = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x1)
    x = concatenate([x,x1], axis=-1)
    x = Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # x = SpatialDropout1D(rate_drop_dense)(x)
    x = Dropout(0.5)(x)
    primary_caps = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps")
    primary_caps2 = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps2")
    primaryconcat = concatenate([primary_caps,primary_caps2], axis=-1)
    category_caps = CapsuleLayer(num_capsule=1, dim_vector=16, num_routing=3, name='category_caps')(primaryconcat)
    out_caps = Length(name='out_caps')(category_caps)
    # # x = Bidirectional(GRU(gru_len,
    # #                       activation='relu',
    # #                       dropout=dropout_p,
    # #                       recurrent_dropout=dropout_p,
    # #                       return_sequences=True))(embed_layer)
    # x = Conv1D(filters=512, kernel_size=4, padding="valid")(x)
    # x = Dropout(0.7)(x)
    # x = Conv1D(filters=256, kernel_size=4, padding="valid")(x)
    # # x = Bidirectional(LSTM(gru_len, return_sequences=False))(x)
    # capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,activation='relu',
    #                   share_weights=True)(x)
    # # capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule1, routings=Routings,activation='relu',
    # #                   share_weights=True)(x)

    # # capsule = concatenate([capsule, capsule1], axis=-1)

    # capsule = Flatten()(capsule)
    # capsule = Dropout(dropout_p)(capsule)

    # # capsule = LeakyReLU()(capsule)

    # # output = Dense(1, activation='sigmoid')(x)
    # output = Dense(1, activation='sigmoid')(out_caps)
    model = Model(inputs=[input1,input2], outputs=out_caps)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


def load_imdb(maxlen=1000):
    x_train = pd.read_csv("~/capsnet/ISOTtrainFIXED.csv",encoding='latin1')
    x_test = pd.read_csv("~/capsnet/ISOTtestFIXED.csv",encoding='latin1')
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    tokenizer = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer.fit_on_texts(texts = x_train['body_text'])
    X = tokenizer.texts_to_sequences(texts = x_train['body_text'])
    X = pad_sequences(sequences = X, maxlen = maxlen)
    print(X.shape)
    Xt = tokenizer.texts_to_sequences(texts=x_test['body_text'])
    Xt = pad_sequences(sequences = Xt, maxlen = maxlen)
    # x_train = sequence.pad_sequences(x_train['body_text'], maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test['body_text'], maxlen=maxlen)
    tokenizer_title = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer_title.fit_on_texts(texts = x_train['title'])
    X_title = tokenizer_title.texts_to_sequences(texts = x_train['title'])
    X_title = pad_sequences(sequences = X_title, maxlen = maxlen)
    Xt_title = tokenizer_title.texts_to_sequences(texts=x_test['title'])
    Xt_title = pad_sequences(sequences = Xt_title, maxlen = maxlen)
    return X,X_title, y_train, Xt,Xt_title, y_test


def main():
    x_train,x_train1, y_train, x_test,x_test1, y_test = load_imdb()

    model = get_model()

    batch_size = 64
    epochs = 5

    model.fit([x_train,x_train1], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([x_test,x_test1], y_test))
    model.save(f"/home/jnguye30/capsnet/ISOT/ISOT{Process}.h5")
    result = model.predict([x_test,x_test1])
    print(result)
    y_pred = []
    count1= 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    for i in result:
      if i>=0.5:
        y_pred.append(1)
        if i <=0.92:
          count1+=1
        elif i >0.92 and i<=0.94:
          count2+=1
        elif i>0.94 and i<=0.96:
          count3+=1
        elif i>0.96 and i<=0.98:
          count4+=1
        elif i>0.98 and i<=1.0:
          count5+=1
      else:
        y_pred.append(0)

    print(count1)
    print(count2)
    print(count3)
    print(count4)
    print(count5)
if __name__ == '__main__':
    main()

# %%
# from keras.layers import K, Activation
# from keras.engine import Layer
from keras.layers import LeakyReLU, Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D, concatenate,LSTM,Conv1D
# from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from vendor.Capsule.Capsule_Keras import *

gru_len = 256
Routings = 3
Num_capsule = 10
Dim_capsule = 16
Dim_capsule1 = 32
dropout_p = 0.25
rate_drop_dense = 0.28

max_features = 20000
maxlen = 1000
embed_size = 256

def get_model():
    input1 = Input(shape=(maxlen,))
    input2 = Input(shape = (maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer2 = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input2)

    x = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x)
    x = Dropout(0.5)(x)
    # x = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x)
    x1 = Bidirectional(LSTM(gru_len, return_sequences=True))(embed_layer2)
    x1 = Dropout(0.5)(x1)
    x1 = Bidirectional(LSTM(int(gru_len/2), return_sequences=True))(x1)
    x1 = Dropout(0.5)(x1)
    # x1 = Bidirectional(LSTM(int(gru_len/4), return_sequences=True))(x1)
    x = concatenate([x,x1], axis=-1)
    x = Conv1D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # x = SpatialDropout1D(rate_drop_dense)(x)
    x = Dropout(0.5)(x)
    primary_caps = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps")
    primary_caps2 = PrimaryCap(x, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid', name="primary_caps2")
    primaryconcat = concatenate([primary_caps,primary_caps2], axis=-1)
    category_caps = CapsuleLayer(num_capsule=1, dim_vector=16, num_routing=3, name='category_caps')(primaryconcat)
    out_caps = Length(name='out_caps')(category_caps)
    # # x = Bidirectional(GRU(gru_len,
    # #                       activation='relu',
    # #                       dropout=dropout_p,
    # #                       recurrent_dropout=dropout_p,
    # #                       return_sequences=True))(embed_layer)
    # x = Conv1D(filters=512, kernel_size=4, padding="valid")(x)
    # x = Dropout(0.7)(x)
    # x = Conv1D(filters=256, kernel_size=4, padding="valid")(x)
    # # x = Bidirectional(LSTM(gru_len, return_sequences=False))(x)
    # capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,activation='relu',
    #                   share_weights=True)(x)
    # # capsule1 = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule1, routings=Routings,activation='relu',
    # #                   share_weights=True)(x)

    # # capsule = concatenate([capsule, capsule1], axis=-1)

    # capsule = Flatten()(capsule)
    # capsule = Dropout(dropout_p)(capsule)

    # # capsule = LeakyReLU()(capsule)

    # # output = Dense(1, activation='sigmoid')(x)
    # output = Dense(1, activation='sigmoid')(out_caps)
    model = Model(inputs=[input1,input2], outputs=out_caps)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


def load_imdb(maxlen=1000):
    x_train = pd.read_csv("~/capsnet/ISOTtrainFIXED.csv",encoding='latin1')
    x_test = pd.read_csv("~/capsnet/ISOTtestFIXED.csv",encoding='latin1')
    y_train = x_train['label'].values
    y_test = x_test['label'].values
    tokenizer = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer.fit_on_texts(texts = x_train['body_text'])
    X = tokenizer.texts_to_sequences(texts = x_train['body_text'])
    X = pad_sequences(sequences = X, maxlen = maxlen)
    print(X.shape)
    Xt = tokenizer.texts_to_sequences(texts=x_test['body_text'])
    Xt = pad_sequences(sequences = Xt, maxlen = maxlen)
    # x_train = sequence.pad_sequences(x_train['body_text'], maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test['body_text'], maxlen=maxlen)
    tokenizer_title = Tokenizer(num_words = maxlen, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
    tokenizer_title.fit_on_texts(texts = x_train['title'])
    X_title = tokenizer_title.texts_to_sequences(texts = x_train['title'])
    X_title = pad_sequences(sequences = X_title, maxlen = maxlen)
    Xt_title = tokenizer_title.texts_to_sequences(texts=x_test['title'])
    Xt_title = pad_sequences(sequences = Xt_title, maxlen = maxlen)
    return X,X_title, y_train, Xt,Xt_title, y_test


def main():
    x_train,x_train1, y_train, x_test,x_test1, y_test = load_imdb()

    model = get_model()

    batch_size = 64
    epochs = 5

    model.fit([x_train,x_train1], y_train, batch_size=batch_size, epochs=epochs,
              validation_data=([x_test,x_test1], y_test))
    model.save(f"/home/jnguye30/capsnet/ISOT/ISOT{Process}.h5")
    result = model.predict([x_test,x_test1])
    print(result)
    result = np.array(result)
    np.savetxt('/home/jnguye30/capsnet/ISOToutputCode.txt', result, delimiter='\n')
    # with open("/content/drive/My Drive/Rumoroutput.txt", "w") as txt_file:
    #   for line in result:
    #     txt_file.write(line)
    #     txt_file.write("\n")
    # y_pred = []
    # count1= 0
    # count2 = 0
    # count3 = 0
    # count4 = 0
    # count5 = 0
    # for i in result:
    #   if i>=0.5:
    #     y_pred.append(1)
    #     if i <=0.92:
    #       count1+=1
    #     elif i >0.92 and i<=0.94:
    #       count2+=1
    #     elif i>0.94 and i<=0.96:
    #       count3+=1
    #     elif i>0.96 and i<=0.98:
    #       count4+=1
    #     elif i>0.98 and i<=1.0:
    #       count5+=1
    #   else:
    #     y_pred.append(0)

    # print(count1)
    # print(count2)
    # print(count3)
    # print(count4)
    # print(count5)
if __name__ == '__main__':
    main()
