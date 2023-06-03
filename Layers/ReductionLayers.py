import tensorflow as tf

class DimensionalityReductionLayer(tf.keras.layers.Layer):
    def __init__(self, energy_threshold=0.95):
        super(DimensionalityReductionLayer, self).__init__()
        self.energy_threshold = energy_threshold

    def build(self, input_shape):
        super(DimensionalityReductionLayer, self).build(input_shape)

    def call(self, inputs):
        # Perform SVD on the input tensor
        s, u, v = tf.linalg.svd(inputs)

        total_energy = tf.reduce_sum(s ** 2)

        singular_value_energy = s ** 2 / total_energy

        num_singular_values = tf.reduce_sum(tf.cast(tf.cumsum(singular_value_energy) < self.energy_threshold, tf.int32)) + 1

        s_truncated = s[:, :num_singular_values]
        u_truncated = u[:, :num_singular_values]
        v_truncated = v[:, :num_singular_values]

        reconstructed_tensor = tf.matmul(tf.matmul(u_truncated, tf.linalg.diag(s_truncated)), tf.transpose(v_truncated))

        return reconstructed_tensor
# See in Custom CNN for reducing layers