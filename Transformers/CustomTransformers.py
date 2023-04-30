import tensorflow as tf
import tensorflow_models as tfm


class CustomTransformerEncoderBlock(tfm.nlp.layers.TransformerEncoderBlock):
    def __init__(self, attention_layer, feedforward_layer, *args, **kwargs):
        self.attention_layer = attention_layer
        self.feedforward_layer = feedforward_layer
        super(CustomTransformerEncoderBlock, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(CustomTransformerEncoderBlock, self).build(input_shape)
        self._attention_layer = self.attention_layer
        self._intermediate_dense = self.feedforward_layer

    def call(self, inputs):
        return super().call(inputs)



custom_attention = tfm.nlp.layers.TalkingHeadsAttention(num_heads=8, key_dim=64)
custom_ffn = tfm.nlp.layers.GatedFeedforward(inner_dim=1024, dropout=0.1, inner_activation="gelu")

# Instantiate the custom TransformerEncoderBlock with the custom layers
custom_transformer_block = CustomTransformerEncoderBlock(
    attention_layer=custom_attention,
    feedforward_layer=custom_ffn,
    inner_activation="gelu",
    num_attention_heads = 3,
    key_dim = 64,
    inner_dim=25
)
tensor_3 = tf.random.uniform((32,24,9))
print(custom_transformer_block([tensor_3,None]))
exit()

class LearnedPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_position, d_model):
        super(LearnedPositionalEncoding, self).__init__()
        self.positional_encoding = self.add_weight("positional_encoding", shape=(max_position, d_model), initializer="zeros")

    def call(self, x):
        seq_length = tf.shape(x)[1]
        return x + self.positional_encoding[:seq_length, :]


import tensorflow as tf

class GPT(tf.keras.Model):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(GPT, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = tf.keras.layers.Add()

        self.decoder_layers = [
            tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.MultiHeadAttention(nhead, key_dim=d_model // nhead),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dense(d_model * 4, activation="gelu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(0.1)
            ])
            for _ in range(num_layers)
        ]

        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x, training=None):
        seq_length = tf.shape(x)[1]
        position = tf.range(0, seq_length, dtype=tf.float32)[:, tf.newaxis]
        x = self.embedding(x)
        x = self.positional_encoding(x + position)

        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
        look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :] * -1e9

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, attention_mask=look_ahead_mask, training=training)

        logits = self.output_layer(x)
        return logits

# Instantiate the simple GPT model
vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6

simple_gpt = GPT(vocab_size, d_model, nhead, num_layers)
