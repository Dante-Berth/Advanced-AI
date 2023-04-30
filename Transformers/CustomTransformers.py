import tensorflow as tf
import tensorflow_models as tfm


print(type(tfm.nlp.layers.TalkingHeadsAttention()))

class TransformersDecoder(tfm.nlp.layers.Transformer):
    def __int__(self, num_attention_heads: int, inner_dim: int, inner_activation: tf.keras.layers.Layer, output_range: int, activation_layer: tf.keras.layers.Layer, *args, **kwargs):
        """

        :param num_attention_heads: int number of attention heads (different relations).
        :param inner_dim: int the output dimension of the first Dense layer in a two-layer feedforward network.
        :param inner_activation: int the activation for the first Dense layer in a two-layer feedforward network.
        :param output_range: int the sequence output range, [0, output_range) for slicing the target sequence. None means the target sequence is not sliced.
        :param activation_layer: tf.keras.layers.Layer  activation layer : TalkingHeadsAttention,GatedFeedforward,KernelAttention,MultiChannelAttention,TalkingHeadsAttention
        :return: tf.keras.layers.Layer
        """
        self._num_attention_heads = num_attention_heads
        self._inner_dim = inner_dim
        self._inner_activation = inner_activation
        self._output_range = output_range
        self._activation_layer = activation_layer
        super(TransformersDecoder, self).__init__(num_attention_heads, inner_dim, inner_activation, output_range, activation_layer, *args, **kwargs)

    def build(self, input_shape):
        # Replace the MultiHeadAttention layer with TalkingHeadsAttention
        self.self.attention = self.activation_layer


tensor_4 = tf.random.uniform((32,24,9,3))
tensor_3 = tf.random.uniform((32,24,9))

attention_mask = None
encoder_block = tfm.nlp.layers.TransformerEncoderBlock(
        num_attention_heads=2, inner_dim=10, inner_activation='relu')
outputs = encoder_block([tensor_3, attention_mask])
print(outputs)

TransformersDecoder(num_attention_heads=2, inner_dim=10, inner_activation='relu',)