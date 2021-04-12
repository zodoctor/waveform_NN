#!/usr/bin/env python3

from argparse import ArgumentParser
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
import h5py
from waveform_NN.normalization import Normalizer
from waveform_NN.networks import residual_block,identity_block
from waveform_NN.losses import MismatchLoss 
import glob 
import pickle
import numpy as np
import sys

parser = ArgumentParser()
parser.add_argument("outdir",type=str)
parser.add_argument("data_path",type=str)
parser.add_argument("n_epochs",type=int)
parser.add_argument("--neurons_per_layer",nargs='*',type=int,default=[])
parser.add_argument("--activations",nargs='*',type=str,default=[])
parser.add_argument("--batch_size",default=32,type=int)
parser.add_argument("--loadmodel",default=False,action='store_true')
parser.add_argument("--architecture",default='residual',type=str)
args = parser.parse_args()

nlayers = len(args.neurons_per_layer)

data_file_names = glob.glob(f'{args.data_path}/*.h5')
from pathlib import Path
Path(args.outdir).mkdir(parents=True,exist_ok=True)

histories = [] # collect training histories
for i,data_file_name in enumerate(data_file_names):

    # get data
    print(f'opening {data_file_name}')
    with h5py.File(data_file_name,'r') as f:
        X = f['X'][...]
        Y = f['Y'][...]
    
    print('size of Y is',sys.getsizeof(Y),'bytes')
    nsamples = X.shape[0]
    nfeatures = Y.shape[1]
    nfreqs = nfeatures//4

    Xtrain,Ytrain = X,Y
    # split training and test sets
    #train_idx = np.random.choice(nsamples,replace=False,size=int(nsamples*0.8))
    #train_sel = np.zeros_like(X[:,0],dtype=bool)
    #train_sel[train_idx] = True
    #Xtrain,Ytrain = X[train_sel,:], Y[train_sel,:]
    #Xtest,Ytest = X[~train_sel,:], Y[~train_sel,:]
    
    #normalize using just the first data set 
    if i==0:
        normalizer = Normalizer(Ytrain)
        with open(f'{args.outdir}/normalizer.pkl','wb') as n:
            pickle.dump(normalizer,n)
        mm_loss = MismatchLoss(normalizer,freqs=np.logspace(np.log10(10.),np.log10(1000.),500)) 
        inputs = keras.Input(shape=(Xtrain.shape[1],))
        model_block = Dense(nfeatures)(inputs)
        model_block = LeakyReLU()(X)
        model_block = layers.Reshape((4,nfreqs))(model_block)
        model_block = identity_block(model_block)
        model_block = layers.Reshape(-1)(model_block)
        model_block = Dense(nfeatures)(model_block)
        model = keras.Model(inputs=inputs,outputs=model_block)
        model.compile(loss=mm_loss,optimizer='adam')
        keras.utils.plot_model(model,f'{args.outdir}/model.png') 
    
        

    Ytrain = normalizer.whiten(Ytrain)
    history = model.fit(Xtrain,Ytrain,epochs=args.n_epochs,batch_size=args.batch_size,validation_split=0.1)
    histories.append(history.history)
    
    # checkpoint 
    model.save(f'{args.outdir}/model')
    with open(f'{args.outdir}/history.pkl','wb') as f:
        pickle.dump(histories,f)

