"""Base classes for models of "soft" hadronic structure functions.

    * Elastic electromagnetic form factors
    * Generalized parton distribution functions (GPDs)
    * Compton form factors (for DVCS)

Actual implementations are in other files. Here only generic
treatment of generalized parametrized model is coded.

Todo:
    * Model defined by grid of numbers
    * Flavored models

"""
from __future__ import annotations

import torch

from math import sqrt
from typing import List, Union
import numpy as np

from . import data, theory, cff
from .constants import tolerance2


class Model(theory.Theory):
    """Base class for all models.

    Instance of this class specifies structure of relevant hadrons.
    Methods provided are typically GPDs, CFFs, elastic FFs,
    DVMP transition TFFs, DAs, etc.
    Main subclasses are:

     - ParameterModel which depends on real parameters (some of which can be
       provided by minimization routine in the fitting procedure).
     - NeuralModel where structure functions are represented as neural nets
       (not yet implemented)
     - GridModel where values of structure functions are represented as
       grids of numbers, which may be interpolated
       (not yet implemented)

    These are then further subclassed to model actual structure functions.

    """
    def __init__(self, **kwargs) -> None:
        self.cffs_evaluated = False  # Controls that NNet CFFs eval happens only once per predict()
        super().__init__(**kwargs)


class ParameterModel(Model):
    """Base class for all parametrized models.

    Attributes:
        parameters: dict {'par0': float, ...}. Actual value of parameter.
        parameters_fixed: dict {'par0': bool, ...}. Is parameter value
            fixed? Considered False if non-existent. Once Fitter object
            is created, user should manipulate this dict only via Fitter methods.
        parameters_limits: dict {'par0: (float, float), ...}. Allowed range
            for fitting. Considered (-inf, inf) if non-existent. Once Fitter object
            is created, user should manipulate this dict only via Fitter methods.

    """
    def __init__(self, **kwargs) -> None:
        # If subclases don't create these dicts, this is the
        # last chance. They have to exist.
        for d in ['parameters', 'parameters_fixed', 'parameters_limits']:
            if not hasattr(self, d):
                setattr(self, d, {})
        # By default, all params are fixed
        self._fix_parameters('ALL')
        # print('ParameterModel init done')
        super().__init__(**kwargs)

    def add_parameters(self, newpars: dict):
        """Append newpars to parameters."""
        # Create parameters dict if it doesn't exist yet
        try:
            self.parameters
        except AttributeError:
            self.parameters = {}
        self.parameters.update(newpars)

    def add_parameters_limits(self, newlimits: dict):
        """Append newlimits to parameters_limits."""
        # Create parameters_limits dict if it doesn't exist yet
        try:
            self.parameters_limits
        except AttributeError:
            self.parameters_limits = {}
        self.parameters_limits.update(newlimits)

    def _release_parameters(self, *pars: str):
        """Release parameters for fitting.

        Args:
            *pars: Names of parameters to be released

        Notes:
            User should relase and fix parameters using methods of Fitter instance.
            This then calls this private ParameterModel method.

        """
        for par in pars:
            if par not in self.parameters:
                raise ValueError('Parameter {} is not defined in model {}'.format(
                        par, self))
            self.parameters_fixed[par] = False

    def _fix_parameters(self, *pars: str):
        """Fix parameters so they are not fitting variables.

        Args:
            *pars: Names of parameters to be fixed. If first name is 'ALL'
                then fix all parameters.

        Notes:
            User should relase and fix parameters using methods of Fitter instance.
            This then calls this private ParameterModel method.

        """
        if pars[0] == 'ALL':
            # fix 'em all
            for par in self.parameters:
                self.parameters_fixed[par] = True
        else:
            for par in pars:
                if par not in self.parameters:
                    raise ValueError('Parameter {} is not defined in model {}'.format(
                            par, self))
                self.parameters_fixed[par] = True

    def free_parameters(self) -> List[str]:
        """Return list of names of free fitting parameters."""
        return [p for p in self.parameters if p not in self.parameters_fixed
                or not self.parameters_fixed[p]]

    def print_parameters(self):
        """Print fitting parameters and their errors."""
        for p in self.free_parameters():
            val = self.parameters[p]
            err = sqrt(tolerance2)*self.parameters_errors[p]
            print('{:5s} = {:8.3f} +- {:5.3f}'.format(p, val, err))


class NeuralModel(Model):
    """Model where CFFs are represented by set of neural nets.

    Args:
        fitpoints (data.DataSet): points to fit to
        theory (theory.Theory): theory/model to be fitted
        smear_replicas (Bool): Should we smear the replica according the uncertainties
        patience (Int): How many batches to wait for improvement before early stopping
        q2in (Bool): Should input layer be [xB, t, Q2] or only [xB, t]? Default: False.
        xpow: CFF will be modelled as CFF = xB**(xpow) * NNet. Default: 0.

    Note:
        There will be separate model where GPDs are represented by neural net.

    """

    def __init__(self, output_layer=['ReH', 'ImH'], **kwargs) -> None:
        self.nets = []  # here come PyTorch models [(mean, std, net), ...]
        self.allCFFs = {'ReH', 'ImH', 'ReE', 'ImE', 'ReHt', 'ImHt', 'ReEt', 'ImEt',
                        'ReHeff', 'ImHeff', 'ReEeff', 'ImEeff',
                        'ReHteff', 'ImHteff', 'ReEteff', 'ImEteff'}
        self.output_layer = output_layer
        self.cffs_map = {cff: k for k, cff in  enumerate(self.output_layer)}
        self.in_training = False
        self.q2in = kwargs.setdefault('q2in', False)
        self.xpow = kwargs.setdefault('xpow', 0)
		
        if self.q2in:
            self.indim = 3
        else:
            self.indim = 2
        super().__init__(**kwargs)

    def get_standard(self, x, leave=[]):
        mean = x.mean(0, keepdim=True)
        std = x.std(0, unbiased=False, keepdim=True)
        for dim in leave:
            # we don't standardize indices e.g.
            # FIXME: if we give up on y-standardization remove this
            mean[0, dim] = 0
            std[0, dim] = 1 - 1e-7
        return mean, std

    def standardize(self, x, mean, std):
        y = x - mean
        y /= (std + 1e-7)
        return y

    def unstandardize(self, x, mean, std):
        y = x * (std + 1e-7)
        y += mean
        return y

    def build_net(self):
        '''Builds net architecture. For user to override.'''
        nn_model = torch.nn.Sequential(
                torch.nn.Linear(self.indim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, len(self.output_layer))
            )
        optimizer = torch.optim.Rprop(nn_model.parameters(),
                lr=0.01)
        return nn_model, optimizer

    def __getattr__(self, name):
        if name in self.output_layer:
            self.cff_index = self.cffs_map[name]
            return self.cffs
        elif name in self.allCFFs:
            return self.zero

    def zero(self, pt):
        return 0
    
    def all_cffs(self, pt):
        
        xB = pt[:,0].view(-1, 1) #xB is store in the first column of the data. See Fitter.py datasets_replica function
        t = pt[:,1].view(-1, 1)
        #print(self.cffs_evaluated)
        
        if not self.cffs_evaluated or self.in_training:
            if self.q2in:
                print("Not supported yet")
            else:
                input_layer = torch.hstack((xB, t))
            
            x = self.standardize(torch.tensor(input_layer, dtype=torch.float32),
                                    self.nn_mean, self.nn_std)
            
            self.all_cffs_val = self.nn_model(x)
            
            if not self.in_training:
                self.cffs_evaluated = True

            return self.all_cffs_val * xB**(self.xpow)
        
    def cffs(self, pt):
        if not self.in_training and not self.cffs_evaluated:
            if self.q2in:
                input_layer = [pt.xB, pt.t, pt.Q2]
            else:
                input_layer = [pt.xB, pt.t]        
            x = self.standardize(torch.tensor(input_layer, dtype=torch.float32),
                                    self.nn_mean, self.nn_std)
            self._cffs = self.nn_model(x)[0]
            self.cffs_evaluated = True
        return self._cffs[self.cff_index] * pt.xB**(self.xpow)

    def prune(self, pts: data.DataSet, min_prob: float = None,
                    max_chisq: float = None, max_chisq_npt: float = None):
        """Remove bad nets.

        If you specify more criteria, violating any is enough to be pruned.

        Args:
            pts: List of datapoints that will be used to evaluate net performance.
            min_prob: Probability that data occurs if model is true has to be larger
                than this if the model is to be considered good (i.e. "p-value")
            max_chisq: Chi-squared has to be smaller than this for the model to be considered good.
            max_chisq_npt: Chi-squared per datapoint has to be smaller than this for the model
                to be considered good.

        Returns:
            None. Changes Theory instance in place by deleting bad nets.

        """
        chis, npts, probs = self.chisq(pts, mesh=True)
        bad_indices = []
        for k, (chi, prob) in enumerate(zip(chis, probs)):
            if ( (min_prob and (prob < min_prob)) or
                    (max_chisq and (chi > max_chisq)) or
                    (max_chisq_npt and (chi/npts > max_chisq_npt)) ):
                bad_indices.append(k)
        for k in bad_indices[::-1]:   # must delete backwards
            del self.nets[k]
            
            
class NeuralModel_DR(Model):
    """Model where CFFs are represented by set of neural nets. Real parts of CFFs are modeled using a dispersion relation.

    Args:
        fitpoints (data.DataSet): points to fit to
        theory (theory.Theory): theory/model to be fitted
        smear_replicas (Bool): Should we smear the replica according the uncertainties
        patience (Int): How many batches to wait for improvement before early stopping
        q2in (Bool): Should input layer be [xB, t, Q2] or only [xB, t]? Default: False.
        xpow: CFF will be modelled as CFF = xB**(xpow) * NNet. Default: 0.

    Note:
        There will be separate model where GPDs are represented by neural net.

    """

    def __init__(self, output_layer=['ImH', 'D'], **kwargs) -> None:
        self.nets = []  # here come PyTorch models [(mean, std, net), ...]
        self.allCFFs = {'ReH', 'ImH', 'ReE', 'ImE', 'ReHt', 'ImHt', 'ReEt', 'ImEt',
                        'ReHeff', 'ImHeff', 'ReEeff', 'ImEeff',
                        'ReHteff', 'ImHteff', 'ReEteff', 'ImEteff'}
        self.output_layer = output_layer
        #self.modeled_cffs = modeled_cffs #Store the CFFs which can be derived from the NN. Could be done better maybe. The Imaginary must be in the same order and the real part after
        self.cffs_map = {cff: k for k, cff in enumerate(self.output_layer)}
        self.in_training = False
        self.q2in = kwargs.setdefault('q2in', False)
        self.xpow = kwargs.setdefault('xpow', 0)
        
        
        #This is just to accomodate the DispersionCFF attributes
        self.parameters = {'dummy':0.0}
        self.parameters_fixed = {'dummy': True}
		
        if self.q2in:
            self.indim = 3
        else:
            self.indim = 2
        super().__init__(**kwargs)

    def get_standard(self, x, leave=[]):
        mean = x.mean(0, keepdim=True)
        std = x.std(0, unbiased=False, keepdim=True)
        for dim in leave:
            # we don't standardize indices e.g.
            # FIXME: if we give up on y-standardization remove this
            mean[0, dim] = 0
            std[0, dim] = 1 - 1e-7
        return mean, std

    def standardize(self, x, mean, std):
        #print("in standardize")
        #print("x ",x)
        #print("mean ",mean)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        y = x - mean
        y /= (std + 1e-7)
        return y

    def unstandardize(self, x, mean, std):
        y = x * (std + 1e-7)
        y += mean
        return y

    def build_net(self):
        '''Builds net architecture. For user to override.'''
        nn_model = torch.nn.Sequential(
                torch.nn.Linear(self.indim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, len(self.output_layer))
            )
        optimizer = torch.optim.Rprop(nn_model.parameters(),
                lr=0.01)
        return nn_model, optimizer
        
    def __getattr__(self, name):
        if name in self.output_layer:
            self.cff_index = self.cffs_map[name]
            return self.cffs
        elif name in self.allCFFs:
            return self.zero

    def zero(self, pt):
        return 0
        
    def all_cffs(self, pt, xi = 0):
        if isinstance(xi, np.ndarray): #Copy from cff.py
            # function was called with third argument that is xi nd array
            
            #This allows to use the dispersion relation in cff.py
            xB = 2.0*xi/(1+xi)
        else:
            # xi should be taken from pt object
            xB = pt[:,0].view(-1, 1) #xB is store in the first column of the data. See Fitter.py datasets_replica function
        
        t = pt[:,1].view(-1, 1)
        #print(self.cffs_evaluated)
        
        if not self.cffs_evaluated or self.in_training:
            if self.q2in:
                print("Not supported yet")
                #input_layer = torch.vstack(xB, t, Q2)
            else:
                input_layer = torch.hstack((xB, t))
            
            #print("input layer ",input_layer)
            #print(self.ImH)
            
            x = self.standardize(torch.tensor(input_layer, dtype=torch.float32),
                                    self.nn_mean, self.nn_std)
            
            # Make sure the CFF have three inputs as in DispersionFixedPoleCFF for example
            self.all_cffs_val = self.nn_model(x) #Make sure IM are at the start of the output and modeled list
            
            if not self.in_training:
                self.cffs_evaluated = True

            return self.all_cffs_val * xB**(self.xpow)
    
    def cffs(self, index, pt, xi: Union[float, torch.Tensor] = 0):
                
        self.curname = self.output_layer[index]
        #print("In CFFs ",self.curname)
        #print("Xi ",xi)
        #print(type(xi))
        if isinstance(xi, torch.Tensor): #Copy from cff.py
            #print("Here in tensor case")
            # function was called with third argument that is xi nd array
            #This allows to use the dispersion relation in cff.py
            xB = 2.0*xi/(1+xi)
        elif isinstance(xi, float) and xi != 0:
            # function was called with third argument that is xi number
            xB = 2.0*xi/(1+xi)
        else:
            #print("Here in the scalar case")
            # xi should be taken from pt object
            xB = pt.xB
            
        #print(xB)
        
        if not self.cffs_evaluated:
            if isinstance(xi, torch.Tensor): #Copy from cff.py
                #create a tensor with x and repeated values of t
                input_layer = torch.hstack((xB.view(-1, 1), torch.full_like(xB.view(-1, 1), pt.t)))
            else:
                input_layer = [xB, pt.t]
            
            #print(input_layer)
            
            #x = self.standardize(torch.tensor(input_layer, dtype=torch.float32),
            #                        self.nn_mean, self.nn_std)
            
            x = self.standardize(input_layer,
                                    self.nn_mean, self.nn_std)
            
            #print("after standard stuff ",x)
        
            # Make sure the CFF have three inputs as in DispersionFixedPoleCFF for example
            self._cffs = self.nn_model(x.float()) #Make sure IM are at the start of the output and modeled list
            
            #print(" self._cffs ",self._cffs)
            
            #self.cffs_evaluated = True
            #print(index)
            #print(self._cffs)
        return self._cffs[:,index] * pt.xB**(self.xpow)

    def prune(self, pts: data.DataSet, min_prob: float = None,
                    max_chisq: float = None, max_chisq_npt: float = None):
        """Remove bad nets.

        If you specify more criteria, violating any is enough to be pruned.

        Args:
            pts: List of datapoints that will be used to evaluate net performance.
            min_prob: Probability that data occurs if model is true has to be larger
                than this if the model is to be considered good (i.e. "p-value")
            max_chisq: Chi-squared has to be smaller than this for the model to be considered good.
            max_chisq_npt: Chi-squared per datapoint has to be smaller than this for the model
                to be considered good.

        Returns:
            None. Changes Theory instance in place by deleting bad nets.

        """
        chis, npts, probs = self.chisq(pts, mesh=True)
        bad_indices = []
        for k, (chi, prob) in enumerate(zip(chis, probs)):
            if ( (min_prob and (prob < min_prob)) or
                    (max_chisq and (chi > max_chisq)) or
                    (max_chisq_npt and (chi/npts > max_chisq_npt)) ):
                bad_indices.append(k)
        for k in bad_indices[::-1]:   # must delete backwards
            del self.nets[k]
    

    #def subtraction(self, pt): #This should use the NN, the real part will be calculated by the CFF_Dispersion methods
    #    """Subtraction constant."""
    #    return 0  # default


class FlavoredNeuralModel(NeuralModel):

    def __init__(self, output_layer=['ReHu', 'ReHd', 'ImHu', 'ImHd'], **kwargs):
         '''Flavored CFFs are defined by 'u' and 'd' name endings and they have to come in pairs.'''
         super().__init__(output_layer, **kwargs)

    def __getattr__(self, name):
        if name+'u' in self.output_layer:
            self.cff_flavored_indices = self.cffs_map[name+'u'], self.cffs_map[name+'d']
            return self.flavored_cffs
        else:  # relegate to normal non-flavor-separated CFFs of parent class
            return super().__getattr__(name)

    def flavored_cffs(self, pt):
        u_ind, d_ind = self.cff_flavored_indices
        if not self.cffs_evaluated:
            if self.q2in:
                input_layer = [pt.xB, pt.t, pt.Q2]
            else:
                input_layer = [pt.xB, pt.t]
            x = self.standardize(torch.tensor(input_layer, dtype=torch.float32),
                                    self.nn_mean, self.nn_std)
            self._cffs = self.nn_model(x)[0]
            self.cffs_evaluated = True
        if hasattr(pt, 'in2particle') and pt.in2particle == 'n':
            res = (4/9)*self._cffs[d_ind] + (1/9)*self._cffs[u_ind]
        else:  # proton
            res = (4/9)*self._cffs[u_ind] + (1/9)*self._cffs[d_ind]
        return res * pt.xB**(self.xpow)

