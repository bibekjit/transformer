from base_layers import MultiHeadAttention, FeedFwd
from tensorflow.keras.layers import LayerNormalization, Dropout
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self,n_heads,d_model,ff_units):
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_units = ff_units
        assert d_model % n_heads == 0
        super().__init__()

    def build(self,input_shape):
        self.ffwd = FeedFwd(self.d_model,self.ff_units)
        self.norm1 = LayerNormalization()
        self.mhsa = MultiHeadAttention(self.d_model, self.n_heads)
        self.norm2 = LayerNormalization()
        self.drop1 = Dropout(0.1)
        self.drop2 = Dropout(0.1)
        super().build(input_shape)

    def call(self,x):
        emb,mask = x
        attn = self.mhsa([[emb]*3,mask])
        attn = self.norm1(attn + emb)
        ffwd = self.ffwd(attn)
        return self.norm2(ffwd + attn)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_heads':self.n_heads,
            'd_model':self.d_model,
            'ff_units':self.ff_units
        })
        return config


class Decoder(tf.keras.layers.Layer):
    def __init__(self,d_model,ff_units,n_heads):
        self.d_model = d_model
        self.ff_units = ff_units
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        super().__init__()

    def build(self,input_shape):
        self.mhsa = MultiHeadAttention(self.d_model,self.n_heads,look_ahead_mask=True)
        self.norm1 = LayerNormalization()
        self.mhca = MultiHeadAttention(self.d_model,self.n_heads)
        self.norm2 = LayerNormalization()
        self.ffwd = FeedFwd(self.d_model,self.ff_units)
        self.norm3 = LayerNormalization()
        super().build(input_shape)

    def call(self,x):
        emb,en_out,dec_mask,en_mask = x
        mhsa = self.mhsa([[emb]*3,dec_mask])
        mhsa = self.norm1(mhsa + emb)
        mhca = self.mhca([[mhsa,en_out,en_out],en_mask])
        mhca = self.norm2(mhsa + mhca)
        ffwd = self.ffwd(mhca)
        return self.norm3(mhca + ffwd)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_heads':self.n_heads,
            'd_model':self.d_model,
            'ff_units':self.ff_units
        })
        return config













