import tensorflow as tf
import tensorflow_models as tfm
from official.modeling import tf_utils
from Activation.CustomActivationLayers import MetaActivationLayer
from official.nlp.modeling.layers import util


class Reshape_Layer_3D(tf.keras.layers.Layer):
    def __init__(self):
        super(Reshape_Layer_3D, self).__init__()
        self.reshape = None

    def build(self, input_shape):
        if len(input_shape) > 3:
            self.reshape = tf.keras.layers.Reshape((input_shape[1], tf.reduce_prod(input_shape[2:]).numpy()[()]))

    def call(self, inputs, *args, **kwargs):
        if self.reshape:
            return self.reshape(inputs)
        else:
            return inputs


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


class TransformerEncoderBlock_layer(tfm.nlp.layers.TransformerEncoderBlock):
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

    def __init__(self, attention_layer: str, feedforward_layer: str, num_random_features=256,
                 num_blocks_intermediate=2, num_heads=8, inner_dim=42, inner_activation="gelu", key_dim=32):
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
        self.num_blocks_intermediate = num_blocks_intermediate
        self.num_blocks_output = 2
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.inner_activation = inner_activation
        self.key_dim = key_dim
        self.feature_transform = "exp"
        super(TransformerEncoderBlock_layer, self).__init__(num_attention_heads=self.num_heads,
                                                            inner_dim=self.inner_dim,
                                                            inner_activation=self.inner_activation,
                                                            key_dim=self.key_dim)

    @staticmethod
    def get_name():
        return "transformers_encoder_block"

    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_attention_layer": ["MultiHeadAttention", "TalkingHeadsAttention",
                                               "KernelAttention"],
            "hyperparameter_feedforward_layer": ["GatedFeedforward", "None"],
            "hyperparameter_num_blocks_intermediate": [1, 10, 1],
            "hyperparameter_num_random_features": [8, 80],
            "hyperparameter_num_heads": [2, 12, 1],
            "hyperparameter_inner_dim": [8, 80],
            "hyperparameter_inner_activation": ["sigmoid", "tanh", MetaActivationLayer(), "relu"],
            "hyperparameter_key_dim": [8, 80]

        }

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
        super(TransformerEncoderBlock_layer, self).build(input_shape)
        if self.attention_layer in ["MultiHeadAttention", "TalkingHeadsAttention"]:
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


class R_TransformerEncoderBlock_layer(tf.keras.layers.Layer):
    def __init__(self, attention_layer, feedforward_layer, num_random_features,
                 num_blocks_intermediate, num_heads, inner_dim, inner_activation, key_dim):
        super(R_TransformerEncoderBlock_layer, self).__init__()
        self.transformer = TransformerEncoderBlock_layer(attention_layer, feedforward_layer, num_random_features,
                                                         num_blocks_intermediate, num_heads, inner_dim,
                                                         inner_activation, key_dim)

    @staticmethod
    def get_name():
        return "transformers_encoder_block"

    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_attention_layer": ["MultiHeadAttention", "TalkingHeadsAttention",
                                               "KernelAttention"],
            "hyperparameter_feedforward_layer": ["GatedFeedforward", "None"],
            "hyperparameter_num_blocks_intermediate": [1, 10, 1],
            "hyperparameter_num_random_features": [8, 80],
            "hyperparameter_num_heads": [2, 12, 1],
            "hyperparameter_inner_dim": [8, 80],
            "hyperparameter_inner_activation": ["sigmoid", "tanh", MetaActivationLayer(), "relu"],
            "hyperparameter_key_dim": [8, 80]

        }

    def build(self, input_shape):

        input_tensor_shape = input_shape[0]
        if input_shape[1]:
            input_tensor_shape_minor = input_shape[1]
            self.reshape_layer_minor = Reshape_Layer_3D()
            self.reshape_layer_minor.build(input_tensor_shape_minor)

        self.reshape_layer_main = Reshape_Layer_3D()
        self.reshape_layer_main.build(input_tensor_shape)
        reshaped_shape = self.reshape_layer_main.compute_output_shape(input_tensor_shape)
        self.transformer.build(reshaped_shape)


    def call(self, inputs):

        if isinstance(inputs[1], tf.Tensor):
            x = self.reshape_layer_main.call(inputs[0])
            y = self.reshape_layer_minor.call(inputs[1])
            output = self.transformer.call([x, y, None])
        else:
            x = self.reshape_layer_main.call(inputs[0])
            output = self.transformer.call([x, x, None])
        return output


if __name__ == "__main__":
    custom_transformer_block = TransformerEncoderBlock_layer(
        attention_layer="MultiHeadAttention",
        feedforward_layer="GatedFeedforward",
        inner_activation=MetaActivationLayer(),
        num_heads=4,
        key_dim=12,
        inner_dim=8
    )

    tensor_123 = tf.ones((4, 5, 6, 7))
    tensor_456 = tf.ones((4, 5, 6, 7))
    print(R_TransformerEncoderBlock_layer(attention_layer="MultiHeadAttention",
        feedforward_layer="GatedFeedforward", num_random_features=256,
                 num_blocks_intermediate=2, num_heads=8, inner_dim=42, inner_activation="gelu", key_dim=32)([tensor_123, None]))
    exit()

    print("second layer")
    print(custom_transformer_block(transform_tensors([tensor_123, None, None])))

    # Instantiate the custom TransformerEncoderBlock with the custom layers
    custom_transformer_block = TransformerEncoderBlock_layer(
        attention_layer="KernelAttention",
        feedforward_layer="GatedFeedforward"
    )
    tensor_3 = tf.ones((48, 23, 24))
    tensor_34 = tf.ones((48, 12, 38))
    print(custom_transformer_block(transform_tensors([tensor_3, tensor_34, None])))
    print("success")
