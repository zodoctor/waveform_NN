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
    
