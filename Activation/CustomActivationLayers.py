import tensorflow as tf
"""
MetaActivationLayer généralise le concept de couche d'activation
"""



@tf.keras.utils.register_keras_serializable()
class Expcos(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Expcos, self).__init__(**kwargs)
        self.weight_1 = self.add_weight(
            name='weight_1',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-3.1415, maxval=3.1415),
            trainable=True,
        )
        self.weight_2 = self.add_weight(
            name='weight_2',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-3.1415, maxval=3.1415),
            trainable=True,
        )

    def get_config(self,**kwargs):
        config = super(Expcos, self).get_config(**kwargs)
        return config

    def call(self, inputs,**kwargs):
        return tf.math.sign(inputs) * tf.math.exp(self.weight_2 * tf.math.cos(self.weight_1 * inputs))

@tf.keras.utils.register_keras_serializable()
class Signlog(tf.keras.layers.Layer):
    """
    Activation function from a paper Dreamer but adding a weight for increasing or reducing the input importance
    """
    def __init__(self,**kwargs):
        super(Signlog, self).__init__(**kwargs)
        self.weight = self.add_weight(
            name='weights',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-10, maxval=10),
            trainable=True
        )

    def get_config(self):
        config = super(Signlog, self).get_config()

        return config

    def call(self, inputs):
        return tf.math.sign(inputs) * tf.math.log(tf.keras.activations.relu(self.weight)*tf.math.abs(inputs) + 1)

@tf.keras.utils.register_keras_serializable()
class Signlog_Expcos(tf.keras.layers.Layer):

    def __init__(self,signlog,expcos,**kwargs):
        super(Signlog_Expcos, self).__init__(**kwargs)
        self.signlog = signlog
        self.expcos = expcos
    def get_config(self):
        config = super(Signlog_Expcos, self).get_config()
        config.update({
            'signlog': tf.keras.layers.serialize(self.signlog),
            'expcos': tf.keras.layers.serialize(self.expcos)
        })
        return config

    @classmethod
    def from_config(cls, config):
        signlog = tf.keras.layers.deserialize(config.pop('signlog'))
        expcos = tf.keras.layers.deserialize(config.pop('expcos'))
        return cls(signlog=signlog, expcos=expcos, **config)
    def call(self, inputs):
        print("entergint in  the layer")
        return self.signlog(inputs) + self.expcos(inputs)


@tf.keras.utils.register_keras_serializable()
class MetaActivationLayer(tf.keras.layers.Layer):
    """
    Idea taken from AutoML Springer P.66
    """
    def __init__(self, **kwargs):
        super(MetaActivationLayer, self).__init__(**kwargs)
        self.activation_gelu = tf.keras.activations.gelu
        self.activation_softmax = tf.keras.activations.softmax
        self.activation_softsign = tf.keras.activations.softsign
        self.activation_Signlog = Signlog(**kwargs)
        self.activation_Expcos = Expcos(**kwargs)

        self.num_weights = 5
        self.weights_list = self.add_weight(
            name='weights',
            shape=(self.num_weights, 1),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True
        )
        self.activation_list = None
    def get_weights(self):
        return self.weights_list

    def get_config(self,**kwargs):
        config = super(MetaActivationLayer, self).get_config(**kwargs)
        return config

    def call(self, inputs, **kwargs):
        print(self.activation_list,"activation_list")
        if self.activation_list is None:
            print("there")
            self.activation_list = [
                self.activation_gelu,
                self.activation_softmax,
                self.activation_softsign,
                self.activation_Signlog,
                self.activation_Expcos
            ]
            print(self.activation_list)
        weighted_activations = tf.zeros_like(inputs)
        for weight, activation in zip(tf.unstack(self.weights_list, axis=0), self.activation_list):
            weighted_activations += weight * activation(inputs)
        return weighted_activations
if __name__=="__main__":
    tensor_123 = tf.ones((4, 5, 6, 7))
    print(MetaActivationLayer()(tensor_123))
    input_layer = tf.keras.layers.Input(shape=(4, 5, 6, 7))
    output = Signlog_Expcos(signlog=Signlog(),expcos=Expcos())(input_layer)



    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.summary()

    PATH = 'testing_model_custom_activation_layer.h5'
    model.save(PATH)
    model = tf.keras.models.load_model(PATH)
    """
    try it
        def get_config(self):
        config = super(OuterCustomLayer, self).get_config()
        config['inner_layer'] = tf.keras.utils.serialize_keras_object(self.inner_layer)
        return config

    """












