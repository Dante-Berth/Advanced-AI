import tensorflow as tf

class DimensionalityReductionLayer(tf.keras.layers.Layer):
    def __init__(self, r):
        super(DimensionalityReductionLayer, self).__init__()
        self.r = r
    @staticmethod
    def rank_r_approx(s, U, V, r):
        s_r, U_r, V_r = s[..., :r], U[..., :, :r], V[..., :, :r]
        A_r = tf.einsum('...s,...us,...vs->...uv', s_r, U_r, V_r)
        return A_r

    def call(self, inputs):

        # Perform SVD on the input tensor
        s, U, V = tf.linalg.svd(inputs)
        return self.rank_r_approx(s,U,V,self.r)

# See in Custom CNN for reducing layers


if __name__ == "__main__":



    tensor_3 = tf.random.uniform((12, 24, 36))
    tensor_4 = tf.random.uniform((12, 24, 36, 48))



    linalgmonolayer = DimensionalityReductionLayer(50)

    # Pass the input tensor through the layer
    output = linalgmonolayer(tensor_4)
    print(output)
    print(tensor_4)