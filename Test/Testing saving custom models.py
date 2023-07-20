from MLP.CustomPerceptron import Perceptron_Layer
from Layers.CustomLayers import LinalgMonolayer
import tensorflow as tf
if __name__ == "__main__":
    vector_1 = tf.keras.layers.Input(shape=(5, 6, 2))
    signallayer = LinalgMonolayer("svd")
    perceptron_layer = Perceptron_Layer(units=20, activation="MetaActivationLayer")
    vector_2 = perceptron_layer(vector_1)
    ouputs = signallayer(vector_2)

    model = tf.keras.models.Model(inputs=vector_1, outputs=ouputs)
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print("Model before the loading")
    model.summary()

    PATH = 'testing_model_custom_activation_layer.h5'
    model.save(PATH)
    model = tf.keras.models.load_model(PATH)
    print("Model loaded")
    print(model.summary())
    vector_12 = tf.ones(shape=(12, 5, 6, 2))
    print(model.predict(vector_12))
