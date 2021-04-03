#!/usr/bin/env python3

from argparse import ArgumentParser
from keras.models import Sequential
from keras.layers import Dense
import h5py
from waveform_NN.normalization import Normalizer

parser = ArgumentParser()
parser.add_argument("neurons_per_layer",nargs='*',type=int)
parser.add_argument("activations",nargs='*',type=str)
parser.add_argument("--batch_size",default=32,type=int)
parser.add_argument("run_name",type=str)
parser.add_argument("data_path",type=str)
parser.add_argument("n_epochs",type=int)
args = parser.parse_args()

nlayers = len(args.neurons_per_layer)

with h5py.File(args.data_path,'r') as f:
    X = f['X'][...]
    Y = f['Y'][...]

nsamples = X.shape[0]
nfeatures = Y.shape[1]
nfreqs = nfeatures//4

train_idx = np.random.choice(nsamples,replace=False,size=int(nsamples*0.8))
train_sel = np.zeros_like(X[:,0],dtype=bool)
train_sel[train_idx] = True
Xtrain,Ytrain = X[train_sel,:], Y[train_sel,:]
Xtest,Ytest = X[~train_sel,:], Y[~train_sel,:]

normalizer = Normalizer(Ytrain)
Ytrain = normalizer.whiten(Ytrain)

model_phase = Sequential()
model_amp = Sequential()

for i in range(nlayers):
    model_phase.add(Dense(args.neurons_per_layer[i],activation=args.activations[i]))
    model_amp.add(Dense(args.neurons_per_layer[i],activation=args.activations[i]))
model_phase.compile(loss="mean_squared_error", optimizer='adam',metrics=["mean_squared_error"])
model_amp.compile(loss="mean_squared_error", optimizer='adam',metrics=["mean_squared_error"])

Ytrain_amp = Ytrain[:,:nfeatures//2]
Ytrain_phase = Ytrain[:,nfeatures//2:]

history_amp = model_amp.fit(Xtrain,Ytrain_amp,epochs=args.n_epochs,batch_size=args.batch_size)
history_phase = model_phase.fit(Xtrain,Ytrain_phase,epochs=args.n_epochs,batch_size=args.batch_size)

model_amp.save(f'model_amp_{args.run_name}.h5')
model_phase.save(f'model_phase_{args.run_name}.h5')
