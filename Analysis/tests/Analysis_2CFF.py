#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from datetime import timedelta

import gepard as g  # NeuralModel with Dispersion relation added
import gepard.plots as gplot
from gepard.fits import GLO15new, AUTIpts, ALUIpts, ACpts, AULpts, ALLpts, H_AULpts, H1ZEUS
from gepard import data, dvcs, cff, model, fitter, theory

from gmaster.fits import th_KM15 #, th_KM10b  # need KM15 for simulated data
th15 = th_KM15
from gmaster.constants import Mp2, toTeX

import torch
import torch.nn as nn
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, set_start_method
import random

os.environ["OMP_NUM_THREADS"] = "2"           # OpenMP threads
os.environ["OPENBLAS_NUM_THREADS"] = "2"      # OpenBLAS (used by NumPy)
os.environ["MKL_NUM_THREADS"] = "2"           # Intel MKL (used by PyTorch/NumPy)
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"    # Apple vecLib (macOS-specific, not needed on JLab cluster)
os.environ["NUMEXPR_NUM_THREADS"] = "2"       # NumExpr (if used)

## Time 
start_time = time.time()

########### Global paths and config ##############
# Configure matplotlib to use LaTeX fonts
plt.rc('text', usetex=True)
params = {'text.latex.preamble' : '\n'.join([r'\usepackage{amssymb}', r'\usepackage{amsmath}'])}
plt.rcParams.update(params)

# Define file paths
RESULTS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/Results/NoDR2CFF_CLAStest_wQ2'
FITS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/fits_models/NoDR2CFF_CLAStest_wQ2'

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FITS_DIR, exist_ok=True)

# Add other necessary paths
sys.path.append('/Users/higuera-admin/Documents/Programs/ldrdgff/gepard/src/gepard')



################ Load datasets ################
import mydatafiles
from mydatafiles import ep2epgamma
# Load datasets
mydset = g.data.loaddata(mydatafiles)
mydset.update(g.data.loaddata(ep2epgamma))


############ Datasets ############
# HALL A 
HallA15_XLUw = g.dset.get(117,[])
HallA15w = g.dset.get(117,[]) + g.dset.get(116,[])[:15] + g.dset.get(116,[])[15:] # 2015(BSDw+BSSw)
HallA17w = g.dset.get(135,[]) + g.dset.get(136,[])[:22] + g.dset.get(136,[])[22:] # 2017(BSDw+BSSw)
HallA6w = g.dset.get(50, []) + g.dset.get(51, []) + g.dset.get(105, []) # 2006(BSDw_byDM+BSSw+BSDovBSS)
HALLAg = HallA15_XLUw + HallA15w + HallA17w
# CLAS 
CLASold = g.dset.get(101, []) + g.dset.get(102, []) + g.dset.get(8, []) + g.dset.get(81, []) + g.dset.get(94, []) + g.dset.get(95, []) + g.dset.get(96, [])
CLAS23 = mydset.get(150, [])
CLAS18XUUw = g.select(mydset.get(162, []), criteria=['FTn == 0']) + g.select(mydset.get(162, []), criteria=['FTn == 1']) + g.select(mydset.get(162, []), criteria=['FTn == 2'])
CLAS18XLU = mydset.get(165, [])
CLAS18 = CLAS18XUUw + CLAS18XLU
CLAS25XUU = g.select(mydset.get(167, []), criteria=['FTn == 0']) #+ g.select(mydset.get(167, []), criteria=['FTn == 1']) #+ g.select(mydset.get(167, []), criteria=['FTn == 2'])
# HERMES
HERMES = ALUIpts + ACpts + H_AULpts + ALLpts + AUTIpts

fitpoints = (
	g.dset.get(101, []) + g.dset.get(102, []) + g.dset.get(8, [])
    #+ CLASold + CLAS18 #+ CLAS23 # + mydset.get(182, []) + mydset.get(192, []) #cFT
    #+ CLAS25XUU
    #+ HERMES
	#+ mydset.get(251, [])  # CLAS_TSA
	#+ HALLAg 
	#+ H1ZEUS	# HERA
)
g.describe_data(fitpoints)


###################  Fit without Dispersion Relation -> Previous NeuralModel fit   ###################
class NNTest(g.model.NeuralModel, g.eff.DipoleEFF, g.dvcs.BM10):
    def build_net(self):
        '''Overriding the default architecture and optimizer'''
        nn_model = torch.nn.Sequential(
            torch.nn.Linear(3, 17),
            torch.nn.ReLU(),
            torch.nn.Linear(17, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 17),
            torch.nn.ReLU(),
            torch.nn.Linear(17, len(self.output_layer))
        )
        optimizer = torch.optim.Rprop(nn_model.parameters(), lr=0.01)
        return nn_model, optimizer


############### Fit with Dispersion Relation using customized network #################

class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()

        # Define the layers for the network
        self.n1 = nn.Linear(3, 25)
        self.n2 = nn.Linear(25, 30)
        self.n3 = nn.Linear(30, 1)

        self.n1p = nn.Linear(1, 12)
        self.n2p = nn.Linear(12, 10)
        self.n3p = nn.Linear(10, 1)

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
        xi = pt.xB / (2 - pt.xB)
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


############# Set up and run the ensemble fit ################
ensembleSize = 10

def train_one(i):
	print(f"Training DR model {i}/{ensembleSize}...")
	start_1 = time.perf_counter()

    #th = NNTest(output_layer=['ImH', 'ReH'])
	#th.name = "Fit No-DR_{i}"

	th = NNTest(output_layer=['ImH', 'ReH'], q2in=True)
	th.name = f"Fit No-DR_{i}"
     
    #th = NNTest_DR(output_layer=['ImH', 'D'], q2in=True)
	#th.name = f"Fit DR_{i}" 
    
	f = g.fitter.NeuralFitter(fitpoints, th, nnets=10, nbatch=10, batchlen=3, regularization='L2', lx_lambda=0.001) 
    #f = g.fitter.NeuralFitter(fitpoints, th, nnets=10, batchlen=10, regularization='L2', lx_lambda=0.01)  #HERMES
	#f = g.fitter.NeuralFitter(fitpoints, th, nnets=10, batchlen=10, regularization='L2', lx_lambda=0.002) #HALLA
	f.fit()
	print("saving model", i)

	torch.save({
		'nets': f.theory.nets,                         # needed for D-term and plots
		'output_layer': f.theory.output_layer,         # helpful for inspecting or checking
		'q2in': True,
		'history': f.history,                          # for training performance plots
		'test_history': f.test_history                 # for ensemble evaluation
	}, os.path.join(FITS_DIR, f'nets_4CFFDR_{i}.pt'))

	end_1 = time.perf_counter()
	elapsed_1 = end_1 - start_1
	print(f"Done saving model {i}. Elapsed time: {elapsed_1:.2f} seconds")
	print("Output layer:", f.theory.output_layer)    
	return f.history, f.test_history


if __name__ == "__main__":
	set_start_method("spawn", force=True)

	with Pool(processes=ensembleSize) as pool:
		results = pool.map(train_one, range(1, ensembleSize + 1))
 
	# Plot results for first model
	# Unpack results
	history, test_history = zip(*results)

	# Plot results
	for i in range(ensembleSize):
		plt.figure(figsize=(8, 5))
		plt.plot(history[i], label="Training Loss", linewidth=2.0)
		plt.plot(test_history[i], label="Test Loss", linestyle="dashed", linewidth=2.0)
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.title(f"Loss vs. Epochs (Model {i})")
		plt.legend()
		plt.grid()
		plt.savefig(os.path.join(RESULTS_DIR, f"loss4CFF_DRmodel_{i}.png"))
		plt.close()    


	print('done...')


end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = str(timedelta(seconds=int(elapsed_time)))

print(f"\n⏱️ Total execution time: {formatted_time} (hh:mm:ss)")