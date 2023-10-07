from tensorflow.keras.layers import Dense
from base_layers import PositionalWordEmbedding
from transformer_layers import Encoder, Decoder
from tensorflow.keras.regularizers import L2
import tensorflow as tf


class Transformer(tf.keras.Model):
    def __init__(self,d_model,n_heads,ff_units,en_vocab_size,maxlen,
                 dec_vocab_size,en_layers=1,dec_layers=1):
        """
        :param d_model: embedding size
        :param n_heads: number of heads
        :param ff_units: num of ffwd units
        :param en_vocab_size: encoder data vocab size
        :param maxlen: input maxlen
        :param dec_vocab_size: decoder data vocab size
        :param en_layers: number encoder layers
        :param dec_layers: number of decoder layers
        """

        self.d_model = d_model
        self.heads = n_heads
        self.units = ff_units
        self.en_vocab = en_vocab_size
        self.maxlen = maxlen
        self.elayers = en_layers
        self.dec_vocab = dec_vocab_size
        self.dlayers = dec_layers
        super().__init__()

    def build(self,input_shape):
        self.en_emb = PositionalWordEmbedding(self.en_vocab,self.d_model,self.maxlen)
        self.dec_emb = PositionalWordEmbedding(self.dec_vocab,self.d_model,self.maxlen)
        self.encoder_layers = [Encoder(self.heads,self.d_model,self.units) for _ in range(self.elayers)]
        self.decoder_layers = [Decoder(self.d_model,self.units,self.heads) for _ in range(self.dlayers)]
        self.out = Dense(self.dec_vocab,kernel_regularizer=L2(0.007),activation='softmax')
        super().build(input_shape)

    def call(self,x):
        en_in,dec_in = x
        en_emb,en_mask = self.en_emb(en_in)
        dec_emb,dec_mask = self.dec_emb(dec_in)

        en_out = en_emb
        for encoder in self.encoder_layers:
            en_out = encoder([en_out,en_mask])

        dec_out = dec_emb
        for decoder in self.decoder_layers:
            dec_out = decoder([dec_out,en_out,dec_mask,en_mask])

        return self.out(dec_out)

    def summary(self,inputs):
        input_shape = [x.shape for x in inputs]
        self.build(input_shape)
        outputs = self.call(inputs)
        tf.keras.Model(inputs,outputs).summary()

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads' : self.heads,
            'ff_units' : self.units,
            'en_vocab_size':self.en_vocab,
            'maxlen':self.maxlen,
            'en_layers':self.elayers,
            'dec_vocab_size':self.dec_vocab,
            'dec_layers':self.dlayers
        })
        return config

    








