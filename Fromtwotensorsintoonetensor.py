import tensorflow as tf
class R_ListTensor(tf.keras.layers.Layer):
    def __init__(self):
        super(R_ListTensor, self).__init__()
        self.instance_padding = True

    @staticmethod
    def get_output_shape(input_shape):
        if isinstance(input_shape[1],tf.TensorShape):
            shape_1 = input_shape[0].as_list()
            shape_2 = input_shape[1].as_list()
            if len(shape_1) > len(shape_2):
                shape_2.extend([1] * (len(shape_1) - len(shape_2)))
            elif len(shape_1) < len(shape_2):
                shape_1.extend([1] * (len(shape_2) - len(shape_1)))
            combined_shape = [max(dim1, dim2) for dim1, dim2 in zip(shape_1, shape_2)]
            return tf.TensorShape(combined_shape)
        else:
            return input_shape
    def call(self, x):
        if isinstance(x[1], tf.Tensor):
            x[0], x[1] = self.creator_pad(x[0],x[1])
            y = x[0] + x[1]
        else:
            y = x[0]
        return y
    def creator_pad(self,tensor_1,tensor_2):
        if self.instance_padding:
            self.padding(tensor_1,tensor_2)

        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape
        if len(shape_1) > len(shape_2):
            tensor_2 = tf.expand_dims(tensor_2,axis=-1)
        elif len(shape_1)<len(shape_2):
            tensor_1 = tf.expand_dims(tensor_1, axis=-1)


        return tf.pad(tensor_1, self.padding_1.numpy().tolist()), tf.pad(tensor_2, self.padding_2.numpy().tolist())

    def padding(self,tensor_1,tensor_2):
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
            padding_1.append([0, max(0,j - i)])
            padding_2.append([0, max(0,i - j)])

        self.padding_1 = tf.constant(padding_1)
        self.padding_2 = tf.constant(padding_2)
        self.instance_padding = False





if __name__=="__main__":

    vector_1 = tf.ones(shape=(12, 5, 6,2))
    vector_2 = tf.ones(shape=(10,8,14))
    print(R_ListTensor()([vector_1, vector_2]))

