import tensorflow as tf
from Fromtwotensorsintoonetensor import R_ListTensor
from tensorflow import keras
@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention_Layer(tf.keras.layers.Layer):
    """
    MultiHeadAttention_Layer is the layer reffered to Multi Head Attention
    """

    def __init__(self, num_heads: int, key_dim: int, value_dim: int, dropout: float, self_attention: str, **kwargs):
        super(MultiHeadAttention_Layer,self).__init__()
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout // 10 * 0.1
        if self_attention == "True":
            self_attention = True
        else:
            self_attention = False
            self.same_dim_two_tensors = R_ListTensor(two_vectors=True)
        self.self_attention = self_attention
        self.multiheadattention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout,
                                                       value_dim=self.value_dim)



    @staticmethod
    def get_name():
        return "multiheadattention"
    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_num_heads": [1, 10, 1],
            "hyperparameter_key_dim": [8, 64,],
            "hyperparameter_value_dim": [8, 64],
            "hyperparameter_dropout": [0, 50, 10],
            "hyperparameter_self_attention": ["True","False"]
        }



    def call(self, input, **kwargs):
        if isinstance(input,list):
            if self.self_attention:
                x = self.multiheadattention.call(input[0],input[0])
            else:
                x = self.same_dim_two_tensors.call(input)
                x = self.multiheadattention.call(*x)
        else:
            x = self.multiheadattention.call(input, input)
        return x
if __name__=="__main__":
    tensor_3 = tf.ones((12,24,36))
    tensor_4 = tf.ones((12,24,36,48))
    a = MultiHeadAttention_Layer(num_heads=10, value_dim=10, key_dim=20,dropout=25,self_attention="False")
    b = MultiHeadAttention_Layer(num_heads=10, value_dim=10, key_dim=20, dropout=25, self_attention="False")
    c = MultiHeadAttention_Layer(num_heads=10, value_dim=10, key_dim=20, dropout=25, self_attention="False")
    d = MultiHeadAttention_Layer(num_heads=10, value_dim=10, key_dim=20, dropout=25, self_attention="False")
    e = MultiHeadAttention_Layer(num_heads=10, value_dim=10, key_dim=20, dropout=25, self_attention="True")
    print(a(tensor_3))
    print(b(tensor_4))
    print(c([tensor_4,tensor_3]))
    print(d([tensor_3, tensor_3]))
    print(e([tensor_4, tensor_3]))