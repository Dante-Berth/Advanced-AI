import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, k, **kwargs):
        self.k = k
        super(CustomLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["k"] = self.k
        return config

    def call(self, input):
        return tf.multiply(input, 2)

@tf.keras.utils.register_keras_serializable()
class Expcos(tf.keras.layers.Layer):
    """
    Activation function X -> sign(X)*\exp(w_{2}\cos(w_{1}X))
    """
    def __init__(self,**kwargs):
        self.weight_1 = self.add_weight(
            name='weight_1',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-3.1415, maxval=3.1415),
            trainable=True,
            dtype=self.dtype
        )
        self.weight_2 = self.add_weight(
            name='weight_2',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-3.1415, maxval=3.1415),
            trainable=True,
            dtype=self.dtype
        )
        super(Expcos, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        config["weight_1"] = self.weight_1
        config["weight_2"] = self.weight_2
        return config
    def call(self, inputs):
        return tf.math.sign(inputs)*tf.math.exp(self.weight_2*tf.math.cos(self.weight_1*inputs))
def main():
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name='input_layer', shape=(10,)),
            tf.keras.layers.Dense(10),
            Expcos(),
            tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        ]
    )
    print("SUMMARY OF THE MODEL CREATED")
    print("-" * 60)
    print(model.summary())
    model.save('model.h5')

    del model

    print()
    print()

    model = tf.keras.models.load_model('model.h5')
    print("SUMMARY OF THE MODEL LOADED")
    print("-" * 60)
    print(model.summary())

if __name__ == "__main__":
    main()