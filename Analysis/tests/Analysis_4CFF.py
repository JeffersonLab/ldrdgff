#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from datetime import timedelta

import gepard as g  # NeuralModel with Dispersion relation added
import gepard.plots as gplot
from gepard.fits import GLO15new, AUTIpts, ALUIpts, ACpts, AULpts, ALLpts
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
RESULTS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/Results/DR4CFF_CLAS+HERMES+TSA_BM10'
FITS_DIR = '/Users/higuera-admin/Documents/Programs/ldrdgff/Analysis/tests/fits_models/DR4CFF_CLAS+HERMES+TSA_BM10'

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


fitpoints = (
	g.dset.get(101, []) + g.dset.get(102, []) + g.dset.get(8, []) 
	+ g.dset.get(81, []) + g.dset.get(94, []) + g.dset.get(95, []) + g.dset.get(96, [])
	+ mydset.get(182, []) + mydset.get(192, []) # + mydset.get(150, []) #cFT
	+ ACpts+ AULpts + ALLpts 
	+ mydset.get(251, [])
	#+ g.dset.get(117, []) + g.dset.get(135, []) + g.dset.get(136, []) #HallA
)
g.describe_data(fitpoints)



# Bin plot

#Data = g.dset[101]
#BSD = Data.df()  # CLAS 2015
#tmlims = [0, 0.13, 0.18, 0.22, 0.3, 0.4, 0.5,0.6,0.7]
#xilims = [0, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.26]
#fig, ax = plt.subplots(figsize=[7, 6])
#ax.scatter(BSD.xi, BSD.tm, s=10)
#for tm in tmlims:
#	ax.axhline(tm, color='g', linewidth=1, alpha=0.4)
#for xi in xilims:
#	ax.axvline(xi, color='g', linewidth=1, alpha=0.4)
#ax.set_xlabel(r'$\xi$', fontsize=14)
#ax.set_ylabel(r'$-t\quad[\rm{GeV}^2]$', fontsize=14)
#plt.tight_layout()
#plt.savefig(os.path.join(RESULTS_DIR, "xi_vs_t_CLAS2015.png"))
#plt.close()




################# Customization for 4-CFF DR model ###################
class CustomNetwork(nn.Module):
	def __init__(self):
		super(CustomNetwork, self).__init__()

		# Define the layers for the network
		self.n1 = nn.Linear(2, 20)  # 2 input features, 20 hidden units
		self.n2 = nn.Linear(20, 25) # 20 hidden units, 25 hidden units
		self.n3 = nn.Linear(25, 2)  # 25 hidden units, 2 output units

		self.n1p = nn.Linear(1, 7)  # 1 input feature, 7 hidden units
		self.n2p = nn.Linear(7, 5)  # 7 hidden units, 5 hidden units
		self.n3p = nn.Linear(5, 1)  # 5 hidden units, 1 output unit

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
class NNTest_DR(g.model.NeuralModel_DR, g.eff.DipoleEFF, g.dvcs.BM10, g.cff.DispersionCFF):
	def build_net(self):
		"""Overriding the default architecture and optimizer"""
		nn_model = CustomNetwork()
		optimizer = torch.optim.Rprop(nn_model.parameters(), lr=0.05)
		return nn_model, optimizer

	def subtraction(self, pt): #This should use the NN, the real part will be calculated by the CFF_Dispersion methods
		"""Subtraction constant."""
		xi = pt.xB / (2 - pt.xB)
		#print("I am here in D")
		#refer to the location of D in output layer (defined in the next cell)
		return self.cffs(2, pt, xi)

	def ImH(self, pt, xi=0) -> float:
		"""Return Im(CFF H) for kinematic point."""
		#refer to the location of Im H in output layer (defined in the next cell)
		"""Return Im(CFF H). If xi is None, infer from pt.xB."""
		#if xi is None:
		#    xi = pt.xB / (2 - pt.xB)
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




############# Set up and run the ensemble fit ################
ensembleSize = 2

def train_one(i):
	print(f"Training DR model {i}/{ensembleSize}...")
	start_1 = time.perf_counter()

	th = NNTest_DR(output_layer=['ImH', 'ImE', 'D'])
	th.name = f"Fit DR_{i}"

	# Set unique random seed per model
	#seed = np.random.randint(1, 1000)
	#torch.manual_seed(seed)
	# 0r
	#seed = 42 + i  # or use time + i for variability
	#torch.manual_seed(seed)
	#np.random.seed(seed)
	#random.seed(seed)

	f = g.fitter.NeuralFitter(fitpoints, th, nnets=5, nbatch=15, batchlen=5, regularization='L2', lx_lambda=0.0001)
	f.fit()
	print("saving model", i)


	torch.save({
		'nets': f.theory.nets,                         # needed for D-term and plots
		'output_layer': f.theory.output_layer,         # helpful for inspecting or checking
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