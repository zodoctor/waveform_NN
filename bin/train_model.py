#!/usr/bin/env python3

from argparse import ArgumentParser
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import h5py
from waveform_NN.normalization import Normalizer
from waveform_NN.networks import residual_block
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
parser.add_argument("--use_residual_block",default=False,action='store_true')
args = parser.parse_args()

nlayers = len(args.neurons_per_layer)

data_file_names = glob.glob(f'{args.data_path}/*.h5')
from pathlib import Path
Path(args.outdir).mkdir(parents=True,exist_ok=True)

history_amps,history_phases = [], [] # collect training histories
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
    Ytrain = normalizer.whiten(Ytrain)
    
    if i==0:
        model_phase = Sequential()
        model_amp = Sequential()

        for j in range(nlayers):
            model_phase.add(Dense(args.neurons_per_layer[j],activation=args.activations[j]))
            model_amp.add(Dense(args.neurons_per_layer[j],activation=args.activations[j]))
        
        #model_amp.add(Dropout(0.2))
        #model_phase.add(Dropout(0.2))
        model_amp.add(Dense(nfeatures//2, activation='linear'))
        model_phase.add(Dense(nfeatures//2, activation='linear'))
        model_phase.compile(loss="mean_squared_error", optimizer='adam',metrics=["mean_squared_error"])
        model_amp.compile(loss="mean_squared_error", optimizer='adam',metrics=["mean_squared_error"])

    Ytrain_amp = Ytrain[:,:nfeatures//2]
    Ytrain_phase = Ytrain[:,nfeatures//2:]

    history_amp = model_amp.fit(Xtrain,Ytrain_amp,epochs=args.n_epochs,batch_size=args.batch_size,validation_split=0.1)
    history_phase = model_phase.fit(Xtrain,Ytrain_phase,epochs=args.n_epochs,batch_size=args.batch_size,validation_split=0.1)
    history_amps.append(history_amp.history)
    history_phases.append(history_phase.history)
    
    # checkpoint 
    model_amp.save(f'{args.outdir}/model_amp')
    model_phase.save(f'{args.outdir}/model_phase')
    with open(f'{args.outdir}/history_amp.pkl','wb') as f:
        pickle.dump(history_amps,f)
    with open(f'{args.outdir}/history_phase.pkl','wb') as f:
        pickle.dump(history_phases,f)

