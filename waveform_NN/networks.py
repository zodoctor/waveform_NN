import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def residual_block(X,n_neurons=[256, 500]):
    X_shortcut = X
    for i in range(len(n_neurons)):
        X = layers.Dense(n_neurons[i])(X)
        X = layers.LeakyReLU()(X)
    X = layers.Concatenate()([X,X_shortcut])
    return X

def identity_block(X):
    X_shortcut = X
    print('xshape',X.shape)
    X = layers.Conv2D(1,kernel_size=(4,20),padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv2D(1,kernel_size=(4,20),padding='same')(X)
    X = layers.Add()([X,X_shortcut])
    X = layers.LeakyReLU()(X)
    return X
   
def dense_shortcut_model(n_input,n_output,n_copies=1):
    inputs = keras.Input(shape=(n_input,))
    X = layers.Dense(n_output)(inputs)
    X_shortcut = X
    X = layers.LeakyReLU()(X)
    for i in range(n_copies):
        X = layers.Dense(n_output)(X)
        X = layers.LeakyReLU()(X)
        X = layers.Dense(n_output)(X)
        X = layers.Add()([X,X_shortcut])
        X = layers.LeakyReLU()(X)
    X = layers.Dense(n_output)(X)
    model = keras.Model(inputs=inputs,outputs=X)
    return model

def loosely_connected_model(n_input,n_output,n_extra_layers=0):
    inputs = keras.Input(shape=(n_input,))
    Xs = []
    for i in range(n_output):
        X = layers.Dense(n_input)(inputs)
        X = layers.LeakyReLU()(X)
        for j in range(n_extra_layers):
            X = layers.Dense(n_input)(X)
            X = layers.LeakyReLU()(X)
        X = layers.Dense(1)(X)
        Xs.append(X)
    X = layers.Concatenate()(Xs)
    model = keras.Model(inputs=inputs,outputs=X)
    return model

def higher_order_block(X,order=2):
    Xs = []
    for i in range(order):
        Xs.append(layers.Lambda(lambda x: x**(i+1))(X))
    X = layers.Concatenate()(Xs)
    return X 
