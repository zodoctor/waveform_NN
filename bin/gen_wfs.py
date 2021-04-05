#!/usr/bin/env python3

from waveform_NN.generate import generate_waveforms
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("outdir",type=str,help='directory in which to save output')
parser.add_argument("runname",type=str,help='name of run')
parser.add_argument("log2_n_samples",type=int)
parser.add_argument("--ncores",default=1,type=int)
parser.add_argument("--sample_type",default="random",type=str)
parser.add_argument("--seed",default=1234,type=int)
args = parser.parse_args()


Path(args.outdir).mkdir(parents=True,exist_ok=True)
generate_waveforms(outstr=f'{args.outdir}/{args.runname}',nsamples = 2**args.log2_n_samples,ncores=args.ncores,sample_type=args.sample_type,seed=args.seed)
