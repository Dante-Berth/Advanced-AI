import tensorflow as tf
from Fromtwotensorsintoonetensor import RListTensor
from tensorflow import keras
@tf.keras.utils.register_keras_serializable()
class MultiHeadAttention_Layer(tf.keras.layers.Layer):
    """
    MultiHeadAttention_Layer is the layer reffered to Multi Head Attention
    """

    def __init__(self, num_heads: int, key_dim: int, value_dim: int, dropout: float, self_attention="False", **kwargs):
        super(MultiHeadAttention_Layer,self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout // 10 * 0.1
        self_attention = self_attention # putting to false
        if self_attention == "True":
            self_attention = True
        else:
            self_attention = False
            self.same_dim_two_tensors = RListTensor(two_vectors=True)
        self.self_attention = self_attention
        self.multiheadattention = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout,
                                                       value_dim=self.value_dim)

    def get_config(self):
        config = super(MultiHeadAttention_Layer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim':self.key_dim,
            'value_dim': self.value_dim,
            'dropout': self.dropout,
            'self_attention': self.self_attention

        })
        return config

    @staticmethod
    def get_name():
        return "multiheadattention"
    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_num_heads": [1, 5, 1],
            "hyperparameter_key_dim": [8, 32],
            "hyperparameter_value_dim": [8, 32],
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
    tensor_3 = tf.ones((28,6,31))
    tensor_4 = tf.ones((1,1,8))
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

    f = MultiHeadAttention_Layer(num_heads=10, value_dim=10, key_dim=20, dropout=25, self_attention="True")
    vector_1 = tf.keras.layers.Input(shape=(5, 12, 2))
    vector_2 = tf.keras.layers.Input(shape=(5, 12))
    vector_3 = [vector_1, vector_2]
    ouputs = f(vector_3)

    model = tf.keras.models.Model(inputs=[vector_1, vector_2], outputs=ouputs)
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print("Model before the loading")
    model.summary()

    PATH = 'testing_model_custom_activation_layer.h5'
    model.save(PATH)
    model_2 = tf.keras.models.load_model(PATH)
    print("Model loaded")
    print(model_2.summary())

    vector_1 = tf.ones(shape=(12, 5, 12, 2))
    vector_2 = tf.ones(shape=(12, 5, 12))
    vector_3 = [vector_1, vector_2]
    print(model_2.predict(vector_3))