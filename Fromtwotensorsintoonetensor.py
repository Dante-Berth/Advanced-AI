import tensorflow as tf
class R_ListTensor(tf.keras.layers.Layer):
    """
    Custom Keras layer that performs element-wise addition of tensors with optional padding.
    """
    def __init__(self):
        """
        Initializes a new instance of the R_ListTensor layer.
        """
        super(R_ListTensor, self).__init__()
        self.instance_padding = True

    @staticmethod
    def get_output_shape(input_shape):
        """
        Computes the output shape of the layer based on the input shapes.

        Args:
            input_shape (tuple or list): Input shape of the layer.

        Returns:
            tf.TensorShape: Output shape of the layer.
        """

        if isinstance(input_shape[1],tf.TensorShape) and isinstance(input_shape[0],tf.TensorShape):
            shape_1 = input_shape[0].as_list()
            shape_2 = input_shape[1].as_list()
            if len(shape_1) > len(shape_2):
                shape_2.extend([1] * (len(shape_1) - len(shape_2)))
            elif len(shape_1) < len(shape_2):
                shape_1.extend([1] * (len(shape_2) - len(shape_1)))
            combined_shape = [max(dim1, dim2) if dim1 is not None and dim2 is not None else None for dim1, dim2 in zip(shape_1, shape_2)]

            return tf.TensorShape(combined_shape)
        elif isinstance(input_shape[0], tf.TensorShape):
            return input_shape[0]
        elif isinstance(input_shape[1], tf.TensorShape):
            return input_shape[1]
        else:
            return "error"
    def call(self, x):
        """
        Forward pass of the layer.

        Args:
            x (list): Input tensors.

        Returns:
            tf.Tensor: Output tensor.
        """
        if len(x)>1 and isinstance(x[1], tf.Tensor):
            print(x[1])
            x[0], x[1] = self.creator_pad(x[0],x[1])
            y = x[0] + x[1]
        else:
            y = x[0]
        return y
    def creator_pad(self,tensor_1,tensor_2):
        """
        Pads the input tensors to match their shapes.

        Args:
            tensor_1 (tf.Tensor): First input tensor.
            tensor_2 (tf.Tensor): Second input tensor.

        Returns:
            tuple: Padded tensors.
        """
        if self.instance_padding:
            self.padding(tensor_1,tensor_2)

        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape
        if len(shape_1) > len(shape_2):
            tensor_2 = tf.expand_dims(tensor_2,axis=-1)
        elif len(shape_1)<len(shape_2):
            tensor_1 = tf.expand_dims(tensor_1, axis=-1)



        return tf.pad(tensor_1, list(self.padding_1)), tf.pad(tensor_2, list(self.padding_2))

    def padding(self,tensor_1,tensor_2):
        """
        Determines the padding required for the input tensors.

        Args:
            tensor_1 (tf.Tensor): First input tensor.
            tensor_2 (tf.Tensor): Second input tensor.
        """
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape
        if len(shape_1) > len(shape_2):
            tensor_2 = tf.expand_dims(tensor_2,axis=-1)
            shape_2 = tensor_2.shape

        elif len(shape_1)<len(shape_2):
            tensor_1 = tf.expand_dims(tensor_1, axis=-1)
            shape_1 = tensor_1.shape

        padding_1 = []
        padding_2 = []

        for i, j in zip(shape_1, shape_2):
            if i is None and j is None:
                padding_1.append([0, 0])
                padding_2.append([0, 0])
            else:
                padding_1.append([0, max(0, j - i)])
                padding_2.append([0, max(0, i - j)])


        self.padding_1 = padding_1
        self.padding_2 = padding_2
        self.instance_padding = False





if __name__=="__main__":

    vector_1 = tf.ones(shape=(12, 5, 6,2))
    vector_2 = tf.ones(shape=(10,8,14))
    print(R_ListTensor()([vector_1, vector_2]))
    print(R_ListTensor()([vector_1]))

