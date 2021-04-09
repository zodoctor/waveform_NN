import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def residual_block(X,n_neurons=[256, 500]):
    X_shortcut = X
    for i in range(len(n_neurons)):
        X = layers.Dense(n_neurons[i])(X)
        X = layers.LeakyReLU()(X)
    X = keras.layers.Concatenate()([X,X_shortcut])
    return X
