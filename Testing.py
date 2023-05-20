import random
b,c,d = 4,12,9
print(b,c,d)
tensor_3_1 = tf.ones((12, b, c))
tensor_4_1 = tf.ones((12, c, b, d))
tensor_3_2 = tf.ones((12, d, c))
tensor_4_2 = tf.ones((12, b, b, d))

# Input vector with size (4, 5, 6)
vector_1 = tf.ones(shape=(b, d, c))
vector_2 = tf.ones(shape=(c,b,d))
shape_1 = vector_1.shape
shape_2 = vector_2.shape

pad_first = shape_1[0] - shape_2[0]
pad_second = shape_1[1] - shape_2[1]
pad_three = shape_1[2] - shape_2[2]


# Pad the input vector
padded_vector = tf.pad(vector_1, [[0, int(tf.keras.activations.relu(0,pad_first).numpy())], [0, int(tf.keras.activations.relu(0,pad_second).numpy())], [0, int(tf.keras.activations.relu(0,pad_three).numpy())]])
print(shape_1)
padded_vector_2 = tf.pad(vector_2, [[0, int(tf.keras.activations.relu(-pad_first,0).numpy())], [0, int(tf.keras.activations.relu(0,-pad_second).numpy())], [0, int(tf.keras.activations.relu(0,-pad_three).numpy())]])
print(shape_2)
print("Padded Vector:")
print(padded_vector)
print("Padded Vector Shape:", padded_vector.shape)
print("Padded Vector:")
print(padded_vector_2)
print("Padded Vector Shape:", padded_vector_2.shape)


import tensorflow as tf

# Input 2D matrix
input_matrix = tf.constant([[1, 2, 3],
                            [4, 5, 6]])

# Pad the matrix with 1 zero around each edge
padded_matrix = tf.pad(input_matrix, [[1, 1], [1, 1]])

print("Padded Matrix:")
print(padded_matrix)