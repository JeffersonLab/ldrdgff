#!/usr/bin/env python
# coding: utf-8

import time
from datetime import timedelta
import os
import sys
import gepard as g  # NeuralModel with Dispersion relation added
from gepard import model, fitter, plots
import gepard.plots as gplot
from gepard.fits import GLO15new, AUTIpts, ALUIpts
from gepard import theory
from gepard.fitter import Fitter
from gepard import data
from gepard.data import DataSet

import gmaster as gm
from gmaster.fits import th_KM15, th_KM10b  # need KM15 for simulated data
th15 = th_KM15
from gmaster.constants import Mp2, toTeX

import torch
import torch.nn as nn
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import logging, copy, pandas as pd
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from math import sqrt
from typing import List, Union
import time
import random
from multiprocessing import Pool, set_start_method

## Time 
start_time = time.time()


# Configure matplotlib to use LaTeX fonts
plt.rc('text', usetex=True)
params = {'text.latex.preamble' : '\n'.join([r'\usepackage{amssymb}', r'\usepackage{amsmath}'])}
plt.rcParams.update(params)

# Define file paths
RESULTS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/Results/Test_borrar'
FITS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/fits_models/Test_borrar'

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FITS_DIR, exist_ok=True)

# Add necessary paths
sys.path.append('/Users/higuera-admin/Documents/Programs/ldrdgff/gepard/src/gepard')

import mydatafiles
from mydatafiles import ep2epgamma

# Load datasets
mydset = g.data.loaddata(mydatafiles)
mydset.update(g.data.loaddata(ep2epgamma))

# Ensure g.dset is initialized correctly before using it
if hasattr(g, 'dset'):
    fitpoints = g.dset.get(101, []) + g.dset.get(102, []) + g.dset.get(8, []) + mydset.get(182, []) + mydset.get(192, []) #+ g.dset.get(81, []) + g.dset.get(94, []) + g.dset.get(95, []) + g.dset.get(96, [])
    g.describe_data(fitpoints)
else:
    print("Warning: dset is not defined correctly.")



# Plot xi vs -t for dataset 101 with bin lines
Data = g.dset[101]
BSD = Data.df()  # CLAS 2015
tmlims = [0, 0.13, 0.18, 0.22, 0.3, 0.4, 0.5]
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



"""

###################  Fit without Dispersion Relation - NeuralModel test   ###################

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
th = NNTest(output_layer=['ImH', 'ReH'])
th.name = "Fit No-DR"

# Set up and run the ensemble fit
ensembleSize = 15
for i in range(1, 1 + ensembleSize):
    print(f"Training non-DR model {i}/{ensembleSize}...")
    f = g.fitter.NeuralFitter(fitpoints, th, nnets=10, nbatch=20, batchlen=5, regularization='L2', lx_lambda=0.01)
    f.fit()
    torch.save(th.nn_model.state_dict(), os.path.join(FITS_DIR, f'Test_NoDR_nets_{i}.pt'))

# Plot results of the last model
fig = gplot.jbod(points=fitpoints, lines=[th], bands=[th])
fig.savefig(os.path.join(RESULTS_DIR, "fit_result_NoDR.png"))
plt.close(fig)
"""



############### Fit with Dispersion Relation using customized network #################

class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()

        # Define the layers for the network
        self.n1 = nn.Linear(2, 20)
        self.n2 = nn.Linear(20, 25)
        self.n3 = nn.Linear(25, 1)

        self.n1p = nn.Linear(1, 12)
        self.n2p = nn.Linear(12, 17)
        self.n3p = nn.Linear(17, 1)

    def forward(self, x):
        x0 = x

        # Build Im network
        x = torch.relu(self.n1(x0))
        x = torch.relu(self.n2(x))
        output1 = self.n3(x)

        # Build D network
        # build the input tensor
        x0_1 = torch.cat((x0[:,1].unsqueeze(1),), dim=0)
        
        x2 = torch.relu(self.n1p(x0_1))
        x2 = torch.relu(self.n2p(x2))
        output2 = self.n3p(x2)

        #concatenate outputs
        output = torch.cat((output1, output2), dim=1)
        return output

class NNTest_DR(g.model.NeuralModel_DR, g.eff.KellyEFF, g.dvcs.BM10, g.cff.DispersionCFF):
    def build_net(self):
        nn_model = CustomNetwork()
        optimizer = torch.optim.Rprop(nn_model.parameters(), lr=0.01)
        return nn_model, optimizer

    def subtraction(self, pt): #This should use the NN, the real part will be calculated by the CFF_Dispersion methods
        """Subtraction constant."""
        #refer to the location of D in output layer (defined in the next cell)
        return self.cffs(1, pt, xi)

    def ImH(self, pt, xi=0) -> float:
        """Return Im(CFF H) for kinematic point."""
        #refer to the location of Im H in output layer (defined in the next cell)
        return self.cffs(0, pt, xi)

    def ImE(self, pt, xi=0):
        """Return Im(CFF E) for kinematic point."""
        return self.zero(pt)

    def ImHt(self, pt, xi=0):
        """Return Im(CFF Ht) for kinematic point."""
        return self.zero(pt)

    def ImEt(self, pt, xi=0):
        """Return Im(CFF Et) for kinematic point."""
        return self.zero(pt)


th_dr = NNTest_DR(output_layer=['ImH', 'D'])
th_dr.name = "Fit DR"

'''
ensembleSize = 2
for i in range(1, 1 + ensembleSize):
    print(f"Training DR model {i}/{ensembleSize}...")
    f_dr = g.fitter.NeuralFitter(fitpoints, th_dr, nnets=2, nbatch=5, batchlen=1, regularization='L2', lx_lambda=0.001)
    f_dr.fit()
    torch.save(th_dr.nn_model.state_dict(), os.path.join(FITS_DIR, f'Test_DR_nets_{i}.pt'))



########## Ensemble for 2CFFs with DR
Data_points = [[dp.xB, dp.t] for dp in fitpoints]
Data_pointsy = np.array([[dp.val, dp.err] for dp in fitpoints])

# Collect ensemble predictions for DR model
ensemble_predictions = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_model = th_dr.nn_model.to(device)

for i in range(1, 1 + ensembleSize):

    model_path = os.path.join(FITS_DIR, f'Test_DR_nets_{i}.pt')
    current_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    current_model.load_state_dict(current_state_dict, strict=False)
    # Set to evaluation mode
    current_model.eval() 

    current_predictions = current_model(torch.as_tensor(Data_points, dtype=torch.float32).to(device))
    ensemble_predictions.append(current_predictions.detach().cpu().numpy()) # Convert to NumPy for analysis

ensemble_predictions = np.stack(ensemble_predictions)
print("Ensemble prediction shape:", ensemble_predictions.shape) # Shape: (num_models, num_data_points, num_outputs)

# Utility functions for evaluation

# Function to remove entries from array:
def remove_entries(a, n):
    if n < a.shape[0]:
        removed_entries = np.random.choice(a.shape[0], n, replace=False)
        return np.delete(a, removed_entries, axis=0), True
    return a, False

# Get residuals from ensemble predictions:
def get_residuals(ensemble_predictions, y_true, current_idx):
    residual = y_true - np.mean(ensemble_predictions[current_idx], axis=0)
    r_mean = np.mean(residual, axis=0)
    r_std = np.std(residual, axis=0)
    return np.mean(r_mean), np.mean(r_std)

# Evaluate ensemble performance:
def evaluate_ensemble(ensemble_predictions, y_true, n_remove, n_trials):
    ensemble_size = ensemble_predictions.shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    for _ in range(n_trials):
        residuals_mean = []
        residuals_std = []
        sizes = [ensemble_size]

        keep_removing = True
        idx = np.arange(ensemble_size)
        res = get_residuals(ensemble_predictions, y_true, idx)
        residuals_mean.append(res[0])
        residuals_std.append(res[1])

        while keep_removing:
            new_idx, keep_removing = remove_entries(idx, n_remove)
            if keep_removing:
                res = get_residuals(ensemble_predictions, y_true, new_idx)
                residuals_mean.append(res[0])
                residuals_std.append(res[1])
                sizes.append(new_idx.shape[0])

            idx = new_idx

        ax[0].plot(sizes, residuals_mean, '-o', linewidth=3.0, markersize=10)
        ax[1].plot(sizes, residuals_std, '-o', linewidth=3.0, markersize=10)
    ax[0].set_title("Mean Residual vs Ensemble Size")
    ax[1].set_title("Residual Std vs Ensemble Size")
    ax[0].grid(True)
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ensemble_eval.png"))
    plt.close()

evaluate_ensemble(ensemble_predictions, Data_pointsy, n_remove=2, n_trials=10)

# Chi2 and final CFF comparisons for DR
print("Chi2 (DR):", th_dr.chisq(fitpoints))

# Final CFF band plots for DR
fig = gplot.CFF3log(cffs=['ImH', 'ReH'], mesh=th_dr, bands=[th_dr], tval=-0.2)
fig.savefig(os.path.join(RESULTS_DIR, "CFF3log_DR.png"))
plt.close(fig)

fig = gplot.CFF3(cffs=['ImH', 'ReH'], mesh=th_dr, bands=[th_dr])
fig.savefig(os.path.join(RESULTS_DIR, "CFF3_mesh_DR.png"))
plt.close(fig)

fig = gplot.CFF3(cffs=['ImH', 'ReH'], lines=[th,th_dr], bands=[th,th_dr])
fig.savefig(os.path.join(RESULTS_DIR, "CFF3_lines_DR.png"))
plt.close(fig)

fig = gplot.CFFt(cffs=['ImH', 'ReH'], lines=[th,th_dr], bands=[th,th_dr])
fig.savefig(os.path.join(RESULTS_DIR, "CFFt_DR.png"))
plt.close(fig)


'''



def train_one(i):
    f = g.fitter.NeuralFitter(fitpoints, th_dr, nnets=10, nbatch=10, batchlen=2, regularization='L2', lx_lambda=0.001)
    f.fit()
    print("saving model", i)
    torch.save(f.theory.nn_model.state_dict(), os.path.join(FITS_DIR, f'Test_DR_nets_{i}.pt'))
    print("done saving model", i)
    return f.history, f.test_history

if __name__ == "__main__":
    ensembleSize = 4  # or however many

    with Pool(processes=8) as pool:
        set_start_method("spawn", force=True)
        results = pool.map(train_one, range(1, ensembleSize + 1))
 
    # Plot results for first model
    # Unpack results
    history, test_history = zip(*results)

    # Plot results\
    for i in range(ensembleSize):
        plt.figure(figsize=(8, 5))
        plt.plot(history[i], label="Training Loss", linewidth=2.0)
        plt.plot(test_history[i], label="Test Loss", linestyle="dashed", linewidth=2.0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss vs. Epochs (Model {i+1})")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(RESULTS_DIR, f"loss_plot_model_{i+1}.png"))
        plt.close()

    print("Done ...")


'''

################# Customization for 4-CFF DR model ###################
class CustomNetwork_4(nn.Module):
    def __init__(self):
        super(CustomNetwork_4, self).__init__()

        # Define the layers for the network
        self.n1 = nn.Linear(2, 20)
        self.n2 = nn.Linear(20, 25)
        self.n3 = nn.Linear(25, 2)

        self.n1p = nn.Linear(1, 7)
        self.n2p = nn.Linear(7, 5)
        self.n3p = nn.Linear(5, 1)

    def forward(self, x):

        x0 = x

        # Build Im network
        x = torch.relu(self.n1(x0))
        x = torch.relu(self.n2(x))
        output1 = self.n3(x)

        # Build D network
        # build the input tensor
        x0_1 = torch.cat((x0[:,1].unsqueeze(1),), dim=0)

        x2 = torch.relu(self.n1p(x0_1))
        x2 = torch.relu(self.n2p(x2))
        output2 = self.n3p(x2)

        #concatenate outputs
        output = torch.cat((output1, output2), dim=1)
        return output



# Define the 4-CFF DR model
class NNTest_DR_4(g.model.NeuralModel_DR, g.eff.DipoleEFF, g.dvcs.BM10, g.cff.DispersionCFF):
    def build_net(self):
        """Overriding the default architecture and optimizer"""
        nn_model = CustomNetwork_4()
        optimizer = torch.optim.Rprop(nn_model.parameters(), lr=0.05)
        return nn_model, optimizer

    def subtraction(self, pt): #This should use the NN, the real part will be calculated by the CFF_Dispersion methods
        """Subtraction constant."""
        #print("I am here in D")
        #refer to the location of D in output layer (defined in the next cell)
        return self.cffs(2, pt, xi)

    def ImH(self, pt, xi=0) -> float:
        """Return Im(CFF H) for kinematic point."""
        #refer to the location of Im H in output layer (defined in the next cell)
        return self.cffs(0, pt, xi)

    def ImE(self, pt, xi=0):
        """Return Im(CFF E) for kinematic point."""
        return self.cffs(1, pt, xi)

    def ImHt(self, pt, xi=0):
        """Return Im(CFF Ht) for kinematic point."""
        return self.zero(pt)

    def ImEt(self, pt, xi=0):
        """Return Im(CFF Et) for kinematic point."""
        return self.zero(pt)


th_dr_4 = NNTest_DR_4(output_layer=['ImH', 'ImE', 'D'])
th_dr_4.name = "4CFFs with DR"

for i in range(1, 1 + ensembleSize):
    print(f"Training DR 4-CFF model {i}/{ensembleSize}...")
    f3 = g.fitter.NeuralFitter(fitpoints, th_dr_4, nnets=1, nbatch=5, batchlen=5, regularization='L2', lx_lambda=0.001)
    f3.fit()
    torch.save(th_dr_4.nn_model.state_dict(), os.path.join(FITS_DIR, f'Test_DR4CFF_nets_{i}.pt'))




# Evaluate chi2
print("Chi2 (4-CFF DR):", th_dr_4.chisq(fitpoints))



# Assign names for clarity in plots
th.name = "No DR"
th_dr.name = "2CFFs with DR"

# Generate CFF plots comparing all three models
fig = gplot.CFF3log(cffs=['ImH', 'ReH', 'ImE', 'ReE'], mesh=th_dr_4, bands=[th_dr_4], tval=-0.2)
fig.savefig(os.path.join(RESULTS_DIR, "CFF3log.png"))
plt.close(fig)

fig = gplot.CFF3(cffs=['ImH', 'ReH', 'ImE', 'ReE'], mesh=th_dr_4, bands=[th_dr_4])  
fig.savefig(os.path.join(RESULTS_DIR, "CFF3.png"))
plt.close(fig)

fig = gplot.CFF3(cffs=['ImH', 'ReH', 'ImE', 'ReE'], lines=[th, th_dr, th_dr_4], bands=[th, th_dr, th_dr_4])
fig.savefig(os.path.join(RESULTS_DIR, "CFF3_lines_Compare_3models.png"))
plt.close(fig)

fig = gplot.CFFt(cffs=['ImH', 'ReH', 'ImE', 'ReE'], lines=[th, th_dr, th_dr_4], bands=[th, th_dr, th_dr_4])
fig.savefig(os.path.join(RESULTS_DIR, "CFFt_Compare_3models.png"))
plt.close(fig)

fig = gplot.CFF3(cffs=['ImH', 'ReH', 'ImE', 'ReE'], lines=[th_dr_4], bands=[th_dr_4])
fig.savefig(os.path.join(RESULTS_DIR, "4CFF_xi.png"))
plt.close(fig)

fig = gplot.CFFt(cffs=['ImH', 'ReH', 'ImE', 'ReE'], lines=[th_dr_4], bands=[th_dr_4])
fig.savefig(os.path.join(RESULTS_DIR, "4CFF_t.png"))
plt.close(fig)




####### D-term estimate from DR fit with 2 and 4 CFFs ##########


ptts = []
tm_list = np.linspace(0.1, 0.5, 10)

for tm in tm_list:
    ptb = g.dset[102][0].copy()
    ptb.t = -tm
    ptb.tm = tm
    th15.prepare(ptb)
    ptts.append(ptb)


D = []
std_values = []
net_indices = []

D = []
std_values = []
net_indices = []
for th_model in [th_dr, th_dr_4]:
    print("\n---- [{} - {}] ----".format(th_model.name, th_model.description))
    A = []
    for pt in ptts[::-1]:
        Ds = []
        for i, net in enumerate(th_model.nets):
            th_model.nn_model, th_model.nn_mean, th_model.nn_std = net
            Ds.append(float(th_model.m.subtraction(pt).detach().numpy() * 18. / 25.))
            th_model.cffs_evaluated = False
        net_indices.append(i)
        Ds = np.array(Ds)
        D_std = Ds.std()
        std_values.append(D_std)
        print("{:.3f}, {:.3f}: {:.3f} +- {:.3f}".format(pt.xB, pt.t, Ds.mean(), Ds.std()))
        A.append((pt.tm, Ds.mean(), Ds.std()))
    th_model.m.parameters['nnet'] = 'ALL'
    D.append(A)


# Comparison with KM15 model
# Comparison plot for D-terms from DR and DR_4 models
NPTS = 20
ts = np.linspace(0, 0.6, NPTS)
KMS = np.array([-(18./25.)*2.768/(1.+tm/1.204**2)**2 for tm in ts])
CDS = np.array(D[0])
CADS = np.array(D[1])

fig, ax = plt.subplots(1, 1, figsize=[6, 6])
ax.axhline(y=0, linewidth=0.5, color='k')
ax.plot(ts, KMS, 'r-.', label='KM15 global fit')
ax.errorbar(CDS[:, 0], CDS[:, 1], CDS[:, 2], linestyle='None', color='navy',
            elinewidth=2, capsize=3, capthick=2, marker='o', label='H with DR')
ax.errorbar(CADS[:, 0] + 0.01, CADS[:, 1], CADS[:, 2], linestyle='None',
            elinewidth=2, capsize=3, capthick=2, marker='o', color='indianred', label='H and E with DR')

ax.set_xlim(0.0, 0.6)
ax.set_ylim(-4.5, 1)
ax.legend(loc=4, handlelength=2.5,
          prop=matplotlib.font_manager.FontProperties(size="large")).set_frame_on(False)
ax.set_xlabel(r'$-t \quad [{\rm GeV}^2]$', fontsize=16)
ax.set_ylabel(r'$D^{Q}(t)$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.02))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.2))

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "Dterm_Compare_DR_DR4.png"))
plt.close()

'''

end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = str(timedelta(seconds=int(elapsed_time)))

print(f"\n⏱️ Total execution time: {formatted_time} (hh:mm:ss)")