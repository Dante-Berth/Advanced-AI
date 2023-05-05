import tensorflow as tf
import tensorflow_models as tfm
from official.modeling import tf_utils
from official.nlp.modeling.layers import util


class LearnedPositionalEncoding(tf.keras.layers.Layer):
    """
       Custom Keras layer for learned positional encoding.

       Args:
           None

       Attributes:
           positional_encoding: A dense layer to learn the positional encoding.

       Methods:
           build(input_shape): Builds the layer with the given input shape.
           call(x): Passes the input through the layer.

       Returns:
           The input sequence with learned positional encoding added.
       """
    def __init__(self):
        super(LearnedPositionalEncoding, self).__init__()

    def build(self, input_shape):
        """
        Builds the layer with the given input shape.

        Args:
            input_shape: A tensor shape object for the input.

        Returns:
            None
        """
        self.positional_encoding = tf.keras.layers.Dense(units=input_shape[-1])


    def call(self, x):
        """
        Passes the input through the layer.

        Args:
            x: The input sequence.

        Returns:
            The input sequence with learned positional encoding added.
        """
        return tf.concat([x, self.positional_encoding(x)], axis=-1)


class CustomTransformerEncoderBlock(tfm.nlp.layers.TransformerEncoderBlock):
    """
    A custom implementation of the Transformer encoder block in TensorFlow NLP library.

    Args:
        attention_layer: str, the type of attention layer to use. One of ["MultiHeadAttention", "TalkingHeadsAttention",
            "MultiChannelAttention", "KernelAttention"]. Default is "MultiHeadAttention".
        feedforward_layer: str, the type of feedforward layer to use. One of ["Dense", "GatedFeedforward"]. Default is
            "Dense".
        num_random_features: int, the number of random Fourier features to use for the KernelAttention layer. Default is
            256.
        feature_transform: str, the feature transformation function to use for the KernelAttention layer. One of
            ["identity", "sqrt", "erf", "relu", "leaky_relu", "elu", "gelu", "sin"]. Default is "exp".
        num_blocks_intermediate: int, the number of blocks in the intermediate feedforward layer. Default is 2.
        num_blocks_output: int, the number of blocks in the output feedforward layer. Default is 2.
        *args: additional positional arguments to pass to the base TransformerEncoderBlock class.
        **kwargs: additional keyword arguments to pass to the base TransformerEncoderBlock class.
        """
    def __init__(self, attention_layer: str, feedforward_layer: str, num_random_features=256, feature_transform="exp",
                 num_blocks_intermediate=2, num_blocks_output=2, *args, **kwargs):
        """
        Initializes the CustomTransformerEncoderBlock.

        Args:
            attention_layer: str, the type of attention layer to use. One of ["MultiHeadAttention", "TalkingHeadsAttention",
                "MultiChannelAttention", "KernelAttention"]. Default is "MultiHeadAttention".
            feedforward_layer: str, the type of feedforward layer to use. One of ["Dense", "GatedFeedforward"]. Default is
                "Dense".
            num_random_features: int, the number of random Fourier features to use for the KernelAttention layer. Default is
                256.
            feature_transform: str, the feature transformation function to use for the KernelAttention layer. One of
                ["identity", "sqrt", "erf", "relu", "leaky_relu", "elu", "gelu", "sin"]. Default is "exp".
            num_blocks_intermediate: int, the number of blocks in the intermediate feedforward layer. Default is 2.
            num_blocks_output: int, the number of blocks in the output feedforward layer. Default is 2.
            *args: additional positional arguments to pass to the base TransformerEncoderBlock class.
            **kwargs: additional keyword arguments to pass to the base TransformerEncoderBlock class.
        """

        self._attention_layer = None
        self.attention_layer = attention_layer
        self.feedforward_layer = feedforward_layer
        self.num_random_features = num_random_features
        self.feature_transform = feature_transform
        self.num_blocks_intermediate = num_blocks_intermediate
        self.num_blocks_output = num_blocks_output
        self.hyperparameter = {
            "hyperparameter_attention_layer": ["MultiHeadAttention", "TalkingHeadsAttention", "MultiChannelAttention", "KernelAttention"],
            "hyperparameter_feedforward_layer": ["GatedFeedforward","None"],
            "hyperparameter_feature_transform" : ["identity", "sqrt", "erf", "relu", "leaky_relu", "elu", "gelu", "sin"],
            "hyperparameter_num_blocks_intermediate" : ["1","10","1"],
            "hyperparameter_num_random_features" : ["8","256","8"],
            "hyperparameter_num_blocks_output" : ["1","10","1"]
        }
        super(CustomTransformerEncoderBlock, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        """
        Creates the variables of the layer.

            This method should be called at the end of the `__init__` method, after
            calling the super constructor with `super().__init__(**kwargs)`.

            Args:
                input_shape: A shape tuple (integer), not including the batch size. For
                    instance, `(32,)` indicates that the expected input will be batches
                    of 32-dimensional vectors.

            Returns:
                None.
        """
        super(CustomTransformerEncoderBlock, self).build(input_shape)
        if self.attention_layer in ["MultiHeadAttention", "TalkingHeadsAttention", "MultiChannelAttention"]:
            self._attention_layer = getattr(tfm.nlp.layers, self.attention_layer)(
                num_heads=self._num_heads,
                key_dim=self._key_dim,
                value_dim=self._value_dim,
                dropout=self._attention_dropout_rate,
                use_bias=self._use_bias,
                kernel_initializer=self._attention_initializer,
                bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
                attention_axes=self._attention_axes,
                output_shape=self._output_last_dim,
                name=f"{self.attention_layer}_attention")

        elif self.attention_layer == "KernelAttention":
            self._attention_layer = tfm.nlp.layers.KernelAttention(
                num_heads=self._num_heads,
                key_dim=self._key_dim,
                value_dim=self._value_dim,
                dropout=self._attention_dropout_rate,
                use_bias=self._use_bias,
                kernel_initializer=self._attention_initializer,
                bias_initializer=tf_utils.clone_initializer(self._bias_initializer),
                attention_axes=self._attention_axes,
                output_shape=self._output_last_dim,
                name=f"{self.attention_layer}_attention",
                feature_transform=self.feature_transform,
                num_random_features=self.num_random_features,
                seed=0,
                redraw=False,
                is_short_seq=False,
                begin_kernel=0,
                scale=None,
                scale_by_length=False,
                use_causal_windowed=False
            )
        else:
            self.attention_layer = self.attention_layer

        if isinstance(input_shape, tf.TensorShape):
            input_tensor_shape = input_shape
        elif isinstance(input_shape, (list, tuple)):
            input_tensor_shape = tf.TensorShape(input_shape[0])
        else:
            raise ValueError(
                "The type of input shape argument is not supported, got: %s" %
                type(input_shape))
        einsum_equation = "abc,cd->abd"
        if len(input_tensor_shape.as_list()) > 3:
            einsum_equation = "...bc,cd->...bd"
        hidden_size = input_tensor_shape[-1]

        if self._key_dim is None:
            self._key_dim = int(hidden_size // self._num_heads)
        if self._output_last_dim is None:
            last_output_shape = hidden_size
        else:
            last_output_shape = self._output_last_dim

        if self.feedforward_layer == "GatedFeedforward":
            self._intermediate_dense = getattr(tfm.nlp.layers, self.feedforward_layer)(
                inner_dim=self._inner_dim,
                inner_activation=self._inner_activation,
                dropout=self._output_dropout_rate,
                num_blocks=self.num_blocks_intermediate
            )

            self._output_dense = getattr(tfm.nlp.layers, self.feedforward_layer)(
                inner_dim=last_output_shape,
                inner_activation=self._inner_activation,
                dropout=self._output_dropout_rate,
                num_blocks=self.num_blocks_output
            )

    def call(self, inputs):
        """
        Applies the layer to the input tensor.

            Args:
                inputs: The input tensor.

            Returns:
                The output tensor.
        """
        return tf.keras.layers.LayerNormalization()(super().call(inputs))
    def get_hyperparameters(self):
        return self.hyperparameter

# Instantiate the custom TransformerEncoderBlock with the custom layers
custom_transformer_block = CustomTransformerEncoderBlock(
    attention_layer="KernelAttention",
    feedforward_layer="GatedFeedforward",
    inner_activation="gelu",
    num_attention_heads=3,
    key_dim=64,
    inner_dim=25
)
tensor_3 = tf.ones((32, 24, 24))
tensor_34 = tf.ones((32, 12, 24))
print(custom_transformer_block([tensor_3, tensor_34, None]))

custom_transformer_block = CustomTransformerEncoderBlock(
    attention_layer="MultiHeadAttention",
    feedforward_layer=None,
    inner_activation="gelu",
    num_attention_heads=3,
    key_dim=24,
    inner_dim=12
)
tensor_3 = tf.ones((55, 24, 24))
tensor_34 = tf.ones((55, 12, 24))
print(custom_transformer_block([tensor_3, tensor_34, None]))

exit()
