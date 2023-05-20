import tensorflow as tf
class tensor_list_tensor(tf.keras.layers.Layer):
    def __init(self):
        super(tensor_list_tensor).__init()
        self.bool_padding = True
    def call(self, x):
        if isinstance(x[1], tf.Tensor):
            x[0], x[1] = self.creator_pad(x[0],x[1])
            y = x[0] + x[1]
    def creator_pad(self,tensor_1,tensor_2):
        if self.instance_padding:
            self.padding(tensor_1,tensor_2)
        return tf.pad(tensor_1, self.padding_1), tf.pad(tensor_2, self.padding_2)

    def padding(self,tensor_1,tensor_2):
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape
        if shape_1 > shape_2:
            tensor_2 = tf.expand_dims(tensor_2,axis=-1)
            shape_2 = tensor_2.shape

        elif shape_1<shape_2:
            tensor_1 = tf.expand_dims(tensor_2, axis=-1)
            shape_1 = tensor_1.shape

        padding_1 = []
        padding_2 = []
        for i, j in zip(shape_1, shape_2):
            padding_1.append([0, max(j - i)])
            padding_2.append([0, max(i - j)])

        self.padding_1 = padding_1
        self.padding_2 = padding_2
        self.instance_padding = False





# Input vector with size (4, 5, 6)
vector_1 = tf.ones(shape=(12, 5, 6))
vector_2 = tf.ones(shape=(10,8,14))
shape_1 = vector_1.shape
shape_2 = vector_2.shape

pad_first = shape_1[0] - shape_2[0]
pad_second = shape_1[1] - shape_2[1]
pad_three = shape_1[2] - shape_2[2]
# Pad the input vector
padded_vector = tf.pad(vector_1, [[0, max(0,-pad_first)], [0, max(0,-pad_second)], [0, max(0,-pad_three)]])
padded_vector_2 = tf.pad(vector_2, [[0, max(0,pad_first)], [0, max(0,pad_second)], [0, max(0,pad_three)]])

print("Padded Vector Shape:", padded_vector.shape )
print("Padded Vector Shape:", padded_vector_2.shape)
tf.keras.layers.Concatenate(axis=-1)([padded_vector,padded_vector_2])
print("success")
exit()

import tensorflow as tf

# Input 2D matrix
input_matrix = tf.constant([[1, 2, 3],
                            [4, 5, 6]])

# Pad the matrix with 1 zero around each edge
padded_matrix = tf.pad(input_matrix, [[1, 1], [1, 1]])

print("Padded Matrix:")
print(padded_matrix)

exit()
print()

