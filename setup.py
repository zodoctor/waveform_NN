#!/usr/bin/env python
__author__ = 'Zoheyr Doctor'
__description__ = 'hi'
from distutils.core import setup

setup(
        name='Waveform_NN',
        version='0.1',
        url='https://github.com/zodoctor/waveform_NN',
        author = __author__,
        author_email = 'zoheyr@gmail.com',
        description = __description__,
        scripts = [
            'bin/gen_wfs.py',
            ],
        packages = [
            'waveform_NN',
            ],
        py_modules = [
            'waveform_NN',
            'waveform_NN.generate',
            'waveform_NN.benchmarking',
            'waveform_NN.normalization',
            ],
        data_files = [],
        install_requires = ['tensorflow','keras','numpy','pycbc','matplotlib','scipy','h5py'],
)
