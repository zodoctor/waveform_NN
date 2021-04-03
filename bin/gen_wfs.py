#!/usr/bin/env python3

from waveform_NN.generate import generate_waveforms
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("log2_n_samples",type=int)
parser.add_argument("--ncores",default=1,type=int)
parser.add_argument("--sample_type",default="random",type=str)
args = parser.parse_args()

generate_waveforms(nsamples = 2**args.log2_n_samples,ncores=args.ncores,sample_type=args.sample_type)
