import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Dense, Dropout, Embedding
from tensorflow.keras.initializers import RandomNormal

l2_reg = L2(7e-3)


class PositionalWordEmbedding(tf.keras.layers.Layer):
    """
    Transformer positional word embedding = pos embeddings + word embeddings
    :param vocab_size: vocabulary size
    :param d_model: embedding dimension
    :param maxlen: input sequence maximum length
    """
    def __init__(self,vocab_size,d_model,maxlen):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.maxlen = maxlen
        self.pos_encodings = self._positional_encoding()
        self.factor = float(d_model)**0.5
        super().__init__()

    def _positional_encoding(self):
        pos = np.zeros((self.maxlen, self.d_model), dtype=np.float32)
        for p in range(self.maxlen):
            for i in range(self.d_model):
                if i % 2 == 0:
                    angle = p / 10000 ** (2 * i / self.d_model)
                    pos[p, i] = np.sin(angle)
                else:
                    angle = p / 10000 ** (2 * i / self.d_model)
                    pos[p, i] = np.cos(angle)
        return pos

    def build(self,input_shape):
        self.word_emb = Embedding(self.vocab_size,self.d_model)
        self.pos_emb = Embedding(self.maxlen,self.d_model,weights=[self.pos_encodings],trainable=False)
        self.drop = Dropout(0.1)
        super().build(input_shape)

    def call(self,x):
        mask = x == 0
        seq_len = x.shape[1]
        pos = tf.range(0,seq_len,1)
        x = self.word_emb(x)*self.factor + self.pos_emb(pos)
        return self.drop(x),mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size':self.vocab_size,
            'd_model':self.d_model,
            'maxlen':self.maxlen,
            'pos_encodings':self.pos_encodings})
        return config


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,n_heads,look_ahead_mask=False):
        """
        Multihead attention layer class
        :param d_model: embedding dimension
        :param n_heads: num attention heads
        :param look_ahead_mask: apply look ahead mask or not (used in decoder)
        """
        self.d = d_model
        self.nh = n_heads
        assert d_model % n_heads == 0
        self.dh = d_model // n_heads
        self.look_ahead = look_ahead_mask
        self.attention_scores = None
        super().__init__()

    def build(self,input_shape):
        self.qw = Dense(self.d,kernel_regularizer=l2_reg,kernel_initializer=RandomNormal())
        self.kw = Dense(self.d,kernel_regularizer=l2_reg,kernel_initializer=RandomNormal())
        self.vw = Dense(self.d,kernel_regularizer=l2_reg,kernel_initializer=RandomNormal())
        self.fc = Dense(self.d,kernel_regularizer=l2_reg,kernel_initializer=RandomNormal())
        self.drop = Dropout(0.1)
        super().build(input_shape)

    def split_heads(self, inputs, batch_size):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.reshape(
            inputs, shape=(batch_size, inputs.shape[1], self.nh, self.dh))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def _create_look_ahead_mask(self,size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        mask = tf.cast(mask, dtype=tf.bool)
        return mask

    def call(self, x):
        """
        The input the param x should be in form -> [[q,k,v],mask]]
        q,k,v -> these refer to the query, key and value respectively (batch,maxlen,size)
        mask -> boolean mask where padding is False and rest is True (batch, maxlen)
        """
        x,mask = x
        q,k,v = x

        q = self.qw(q)  # query
        k = self.kw(k)  # key
        v = self.vw(v)  # value
        b,maxlen = mask.shape

        # create attention heads
        q = self.split_heads(q,b)
        k = self.split_heads(k,b)
        v = self.split_heads(v,b)

        # apply padding attention mask
        score = tf.matmul(q,k,transpose_b=True)/(self.d**0.5)
        mask = tf.expand_dims(tf.expand_dims(mask,1),1)
        score = tf.where(mask,-1e9,score)

        # apply look ahead attention mask (only for decoder)
        if self.look_ahead:
            mask = self._create_look_ahead_mask(maxlen)
            score = tf.where(mask,-1e9,score)

        # apply softmax
        score = tf.nn.softmax(score, axis=-1)

        # save attention score
        self.attention_scores = score

        # get attention vector
        attn = tf.matmul(score,v)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn,(-1, tf.shape(attn)[1], self.d))
        attn = self.fc(attn)
        return attn

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model':self.d,
            'n_heads':self.nh,
            'look_ahead_mask':self.look_ahead})
        return config


class FeedFwd(tf.keras.layers.Layer):
    def __init__(self,d_model,units):
        """
        Feed forward layer class
        :param d_model: embedding dimension
        :param units: feed forward units
        """
        self.d = d_model
        self.units = units
        super().__init__()

    def build(self,input_shape):
        self.fc2 = Dense(self.d,kernel_regularizer=l2_reg,kernel_initializer=RandomNormal())
        self.fc1 = Dense(self.units,activation='relu',kernel_regularizer=l2_reg,kernel_initializer=RandomNormal())
        self.drop = Dropout(0.1)
        super().build(input_shape)

    def call(self, x):
        o = self.fc1(x)
        o = self.fc2(o)
        o = self.drop(o)
        return o

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model':self.d,
            'units':self.units})
        return config