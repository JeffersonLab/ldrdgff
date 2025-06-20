#!/usr/bin/env python
# coding: utf-8

## 1. Intro

import gepard as g
import numpy as np

print(g.__version__)

# This just gets my current git revision hash, in case version number above is not reliable.
# You should comment this out if your Gepard installation is not git cloned from github repository.
import os
import subprocess

branch = subprocess.check_output(['git', '-C', os.path.dirname(g.__file__),
                                  'branch', '--show-current']).decode('ascii').strip()
revhash = subprocess.check_output(['git', '-C', os.path.dirname(g.__file__),
                                   'rev-parse', 'HEAD']).decode('ascii').strip()
print('Current git branch is "{}" and revision hash is {}'.format(branch, revhash))

import torch

# 
HallA15_XLUw = [g.dset[117]]
HallA15w = [g.dset[117], g.dset[116][:15], g.dset[116][15:]]
HallA17w = [g.dset[135], g.dset[136][:22], g.dset[136][22:]]


class NN(g.model.NeuralModel, g.eff.KellyEFF, g.dvcs.BM10tw2):

    def build_net(self):
            '''Overriding the default architecture and optimizer'''
            nn_model = torch.nn.Sequential(
                    torch.nn.Linear(3, 23),
                    torch.nn.ReLU(),
                    torch.nn.Linear(23, 37),
                    torch.nn.ReLU(),
                    torch.nn.Linear(37, 19),
                    torch.nn.ReLU(),
                    torch.nn.Linear(19, len(self.output_layer))
                )
            optimizer = torch.optim.Rprop(nn_model.parameters(), lr=0.01)
            return nn_model, optimizer

## Fits

old_map = {'XLUw': HallA15_XLUw, 'Xw': HallA15w}
new_map = {'Xw': HallA17w}
out_map = {'2C': ['ImH', 'ReH'], '4C': ['ImH', 'ReH', 'ImHt', 'ReEt'], 
           '4Ce': ['ImH', 'ReH', 'ImE', 'ReE'],
           '6C': ['ImH', 'ReH', 'ImE', 'ReE', 'ImHt', 'ImEt']}
xpow_map = {'zero': 0, 'half': -0.5, 'one': -1., 'three': -1.5}

def fit(outs='4Ce', xpows='zero', old='Xw', new=None, nnets=10):

    OUTPUT_LAYER = out_map[outs]
    XPOW = xpow_map[xpows]
    
    fitsets = old_map[old]
    setnames = 'old{}'.format(old)
    if new:
        setnames += '_new{}'.format(new)
        for dset in new_map[new]:
            fitsets.append(dset)

    FILENAME = 'nets_{}_{}_{}.tar'.format(outs, xpows, setnames)   # where do we save

    th = NN(output_layer=OUTPUT_LAYER, q2in=True, xpow=XPOW)

    f = g.fitter.NeuralFitter(fitsets, th, nnets=nnets, batchlen=10, regularization='L2', lx_lambda=0.002)

    print("---------------------------------------------")
    print("Training {} nets with model {}".format(f.nnets, th.output_layer))
    print("CFFs are modelled as x^({})*NNet".format(XPOW))
    print("We'll do {} epochs with {} regularization (lx_lambda={})".format(f.nbatch*f.batchlen,
                           f.regularization, f.lx_lambda))
    print("Data used is:")
    g.describe_data(f.datapoints)
    if new:
        print("Where also new 2017 Hall A data is used.")
    print("Nets will be saved in {}".format(FILENAME))
    print("---------------------------------------------")

    f.fit()

    torch.save(th.nets, FILENAME)
    print("Nets saved in {}".format(FILENAME))

    for fitset in fitsets:
        print('\n {} {}'.format(fitset[0].collaboration, fitset[0].y1name), end='')
        print(' {:.1f}/{}  '.format(th.chisq(fitset)[0], len(fitset)), end='')

