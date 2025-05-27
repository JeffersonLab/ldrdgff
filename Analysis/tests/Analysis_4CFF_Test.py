#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")  # Prevent X server issues

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool, set_start_method

import gepard as g
from gepard import data
import gepard.plots as gplot

from gmaster.fits import th_KM15
from gmaster.constants import Mp2, toTeX


# Global paths and config
ensembleSize = 4
RESULTS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/Results/Test_borrar'
FITS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/fits_models/Test_borrar'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FITS_DIR, exist_ok=True)

plt.rc('text', usetex=True)
plt.rcParams.update({
    'text.latex.preamble': '\n'.join([r'\usepackage{amssymb}', r'\usepackage{amsmath}'])
})

# Add necessary paths
sys.path.append('/Users/higuera-admin/Documents/Programs/ldrdgff/gepard/src/gepard')

import mydatafiles
from mydatafiles import ep2epgamma
# Load data
mydset = g.data.loaddata(mydatafiles)
mydset.update(g.data.loaddata(mydatafiles.ep2epgamma))
fitpoints = (
    g.dset.get(101, []) + g.dset.get(102, []) + g.dset.get(8, []) + g.dset.get(81, []) +
    g.dset.get(94, []) + g.dset.get(95, []) + g.dset.get(96, []) +
    mydset.get(182, []) + mydset.get(192, [])
)
g.describe_data(fitpoints)

# Bin plot
BSD = g.dset[101].df()
tmlims = [0, 0.13, 0.18, 0.22, 0.3, 0.4, 0.5]
xilims = [0, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.26]
fig, ax = plt.subplots(figsize=[7, 6])
ax.scatter(BSD.xi, BSD.tm, s=10)
for tm in tmlims:
    ax.axhline(tm, color='g', linewidth=1, alpha=0.4)
for xi in xilims:
    ax.axvline(xi, color='g', linewidth=1, alpha=0.4)
ax.set_xlabel(r'$\xi$', fontsize=14)
ax.set_ylabel(r'$-t\\quad[\rm{GeV}^2]$', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "xi_vs_t_CLAS2015.png"))
plt.close()


# Network and Model Definitions
class CustomNetwork_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.Linear(2, 20)
        self.n2 = nn.Linear(20, 25)
        self.n3 = nn.Linear(25, 2)
        self.n1p = nn.Linear(1, 7)
        self.n2p = nn.Linear(7, 5)
        self.n3p = nn.Linear(5, 1)

    def forward(self, x):
        x0 = x
        x = torch.relu(self.n1(x0))
        x = torch.relu(self.n2(x))
        output1 = self.n3(x)
        x0_1 = x0[:, 1].unsqueeze(1)
        x2 = torch.relu(self.n1p(x0_1))
        x2 = torch.relu(self.n2p(x2))
        output2 = self.n3p(x2)
        return torch.cat((output1, output2), dim=1)


class NNTest_DR_4(g.model.NeuralModel_DR, g.eff.DipoleEFF, g.dvcs.BM10, g.cff.DispersionCFF):
    def build_net(self):
        model = CustomNetwork_4()
        optimizer = torch.optim.Rprop(model.parameters(), lr=0.05)
        return model, optimizer

    def subtraction(self, pt): return self.cffs(2, pt, xi)
    def ImH(self, pt, xi=0): return self.cffs(0, pt, xi)
    def ImE(self, pt, xi=0): return self.cffs(1, pt, xi)
    def ImHt(self, pt, xi=0): return self.zero(pt)
    def ImEt(self, pt, xi=0): return self.zero(pt)


def train_one(i):
    print(f"[{os.getpid()}] Training model {i}/{ensembleSize}")
    th = NNTest_DR_4(output_layer=['ImH', 'ImE', 'D'])
    th.name = f"4CFFs DR Model {i}"
    f = g.fitter.NeuralFitter(fitpoints, th, nnets=10, nbatch=10, batchlen=2, regularization='L2', lx_lambda=0.001)
    f.fit()
    model_path = os.path.join(FITS_DIR, f'nets_4CFF_DR_{i}.pt')
    torch.save({
        'nets': f.theory.nets,
        'output_layer': f.theory.output_layer,
        'history': f.history,
        'test_history': f.test_history
    }, model_path)
    print(f"[{os.getpid()}] Saved model {i} to {model_path}")
    return f.history, f.test_history


if __name__ == "__main__":
    start_time = time.time()
    #set_start_method("spawn", force=True)
    set_start_method("fork") 

    with Pool(processes=ensembleSize) as pool:
        results = pool.map(train_one, range(1, ensembleSize + 1))

    history, test_history = zip(*results)
    for i in range(ensembleSize):
        plt.figure(figsize=(8, 5))
        plt.plot(history[i], label="Training Loss", linewidth=2.0)
        plt.plot(test_history[i], label="Test Loss", linestyle="dashed", linewidth=2.0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss vs. Epochs (Model {i+1})")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(RESULTS_DIR, f"loss4CFF_model_{i+1}.png"))
        plt.close()

    elapsed = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"\n✅ DONE. Execution time: {elapsed}")