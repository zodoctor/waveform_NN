#!/usr/bin/env python3

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("data_path",type=str,help='path h5 file with test data')
parser.add_argument("model_dir",type=str,help='path to folder containing model')

args = parser.parse_args()

import h5py 
import keras
from keras.models import load_model
import numpy as np
from waveform_NN.benchmarking import calc_mismatch
from waveform_NN.losses import MismatchLoss
import pickle
import os

interp_grid = np.logspace(np.log10(10.),np.log10(1000.),500)

# get test data
with h5py.File(args.data_path,'r') as f:
    X = f['X'][...]
    Y = f['Y'][...]

# get normalizer
with open(f'{args.model_dir}/normalizer.pkl','rb') as f:
    normalizer = pickle.load(f)
Ywht = normalizer.whiten(Y)

# get models
if os.path.exists(f'{args.model_dir}/model_amp'):
    split=True 
    model_amp = load_model(f'{args.model_dir}/model_amp')
    model_phase = load_model(f'{args.model_dir}/model_phase')
    y_amp_pred = model_amp.predict(X)
    y_phase_pred = model_phase.predict(X)
    y_pred = np.concatenate((y_amp_pred,y_phase_pred),axis=1) 
    with open(f'{args.model_dir}/history_amp.pkl','rb') as f:
        history_amp = pickle.load(f)
    with open(f'{args.model_dir}/history_phase.pkl','rb') as f:
        history_phase = pickle.load(f)
else:
    split=False
    mmloss = MismatchLoss(normalizer,interp_grid,approx=True)
    model = load_model(f'{args.model_dir}/model',custom_objects={'__call__':mmloss})
    with open(f'{args.model_dir}/history.pkl','rb') as f:
        history = pickle.load(f)
    y_pred = model.predict(X)

Yreconst = normalizer.color(y_pred)

# figure out indices of each amplitude / phase segment
n_features = Yreconst.shape[1]
amp1_max_idx = n_features//4
amp2_max_idx = n_features//2
phase1_max_idx = amp1_max_idx + amp2_max_idx 

hpMMs = []
hxMMs = []

delta_f = 1./32
f_final = 1000.
for i in range(Yreconst.shape[0]):
    hpMMs.append(calc_mismatch(Yreconst[i,:amp1_max_idx],Yreconst[i,amp2_max_idx:phase1_max_idx],Y[i,:amp1_max_idx],Y[i,amp2_max_idx:phase1_max_idx],delta_f,f_final,interp_grid))
    hxMMs.append(calc_mismatch(Yreconst[i,amp1_max_idx:amp2_max_idx],Yreconst[i,phase1_max_idx:],Y[i,amp1_max_idx:amp2_max_idx],Y[i,phase1_max_idx:],delta_f,f_final,interp_grid))

with h5py.File(f'{args.model_dir}/evaluation_data.h5','w') as f:
    f.create_dataset("Yreconst",data = Yreconst)
    f.create_dataset("hpMMs",data = np.array(hpMMs))
    f.create_dataset("hxMMs",data = np.array(hxMMs))
