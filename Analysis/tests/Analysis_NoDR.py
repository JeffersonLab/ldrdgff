#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from datetime import timedelta

import gepard as g  # NeuralModel with Dispersion relation added
#from gepard import model, fitter, plots
import gepard.plots as gplot
from gepard.fits import GLO15new, AUTIpts, ALUIpts, ACpts, AULpts, ALLpts
from gepard import data
#from gepard.data import DataSet
#from gepard import theory
#from gepard.fitter import Fitter

#import gmaster as gm
from gmaster.fits import th_KM15 #, th_KM10b  # need KM15 for simulated data
th15 = th_KM15
from gmaster.constants import Mp2, toTeX

import torch
import torch.nn as nn
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, set_start_method

#import logging, copy, pandas as pd
#from scipy.integrate import quad
#from scipy.interpolate import InterpolatedUnivariateSpline
#from scipy.stats import norm
#from math import sqrt
#from typing import List, Union
#import random


## Time 
start_time = time.time()



############### Configurations ################
# Configure matplotlib to use LaTeX fonts
plt.rc('text', usetex=True)
params = {'text.latex.preamble' : '\n'.join([r'\usepackage{amssymb}', r'\usepackage{amsmath}'])}
plt.rcParams.update(params)

# Define file paths
RESULTS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/Results/Test'
FITS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/fits_models/Test'

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FITS_DIR, exist_ok=True)

# Add necessary paths
sys.path.append('/Users/higuera-admin/Documents/Programs/ldrdgff/gepard/src/gepard')



################ Load datasets ################
import mydatafiles
from mydatafiles import ep2epgamma

# Load my datasets
mydset = g.data.loaddata(mydatafiles)
mydset.update(g.data.loaddata(ep2epgamma))


fitpoints = (
    g.dset.get(101, []) + g.dset.get(102, []) + g.dset.get(8, []) + 
    g.dset.get(81, []) + g.dset.get(94, []) + g.dset.get(95, []) + g.dset.get(96, []) 
    + mydset.get(182, []) + mydset.get(192, []) 
    + ACpts+ AULpts + ALLpts 
    #+ mydset.get(251, [])
)   
g.describe_data(fitpoints)

# Plot xi vs -t for dataset 101 with bin lines
Data = g.dset[101]
BSD = Data.df()  # CLAS 2015
tmlims = [0, 0.13, 0.18, 0.22, 0.3, 0.4, 0.5, 0.6, 0.7]
xilims = [0, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.26]

fig, ax = plt.subplots(figsize=[7, 6])
ax.scatter(BSD.xi, BSD.tm, s=10)
for tm in tmlims:
    ax.axhline(tm, color='g', linewidth=1, alpha=0.4)
for xi in xilims:
    ax.axvline(xi, color='g', linewidth=1, alpha=0.4)
ax.set_xlabel(r'$\xi$', fontsize=14)
ax.set_ylabel(r'$-t\quad[\rm{GeV}^2]$', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "xi_vs_t_CLAS2015.png"))
plt.close()




# Set up and run the ensemble fit
ensembleSize = 20

###################  Fit without Dispersion Relation -> Previous NeuralModel fit   ###################
class NNTest(g.model.NeuralModel, g.eff.DipoleEFF, g.dvcs.BM10):
    def build_net(self):
        '''Overriding the default architecture and optimizer'''
        nn_model = torch.nn.Sequential(
            torch.nn.Linear(2, 17),
            torch.nn.ReLU(),
            torch.nn.Linear(17, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 17),
            torch.nn.ReLU(),
            torch.nn.Linear(17, len(self.output_layer))
        )
        optimizer = torch.optim.Rprop(nn_model.parameters(), lr=0.01)
        return nn_model, optimizer

# Describe data and initialize model
th = NNTest(output_layer=['ImH', 'ReH', 'ReE', 'ImE'])
th.name = "Fit No-DR"


def train_one(i):

    print(f"Training non-DR model {i}/{ensembleSize}...")
    f = g.fitter.NeuralFitter(fitpoints, th, nnets=20, nbatch=10, batchlen=2, regularization='L2', lx_lambda=0.0001)
    f.fit()
    print("saving model", i)

    torch.save({
        'nets': f.theory.nets,                      # needed for plots
        'output_layer': f.theory.output_layer,      # helpful for inspecting or checking
        'history': f.history,                       # for training performance plots
        'test_history': f.test_history              # for ensemble evaluation
    }, os.path.join(FITS_DIR, f'nets_NoDR_{i}.pt'))

    print("done saving model", i)
    print("Output layer:", f.theory.output_layer)   
    return f.history, f.test_history

if __name__ == "__main__":
    set_start_method("spawn", force=True)

    with Pool(processes=5) as pool:
        results = pool.map(train_one, range(1, ensembleSize + 1))

    # Unpack results
    history, test_history = zip(*results)

    # Plot results
    for i in range(ensembleSize):
        plt.figure(figsize=(8, 5))
        plt.plot(history[i], label="Training Loss", linewidth=2.0)
        plt.plot(test_history[i], label="Test Loss", linestyle="dashed", linewidth=2.0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss vs. Epochs (Model {i+1})")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(RESULTS_DIR, f"lossNoDR_model_{i+1}.png"))
        plt.close()  

    print('done...')



end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = str(timedelta(seconds=int(elapsed_time)))

print(f"\n⏱️ Total execution time: {formatted_time} (hh:mm:ss)")