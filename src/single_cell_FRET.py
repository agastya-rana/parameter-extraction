"""
Single-cell FRET class that holds stimulus, model and inference parameters.

Created by Nirag Kadakia and Agastya Rana, 10-21-2021.

Cleaned
"""

import scipy as sp
import numpy as np
import sys
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import scipy.linalg as LA
from src.load_data import load_stim_file, load_meas_file
import src.models as models
import collections
from src.load_data import load_FRET_recording
from src.save_data import save_stim, save_meas_data

MODEL_DEP_PARAMS = ['nD', 'nP', 'L_idxs', 'P_idxs', 'state_bounds', 'param_bounds', 'x0']
INT_PARAMS = ['nT', 'est_beg_T', 'est_end_T', 'pred_end_T', 'step_stim_density', 'step_stim_seed']
LIST_PARAMS = ['x0', 'beta_array', 'meas_noise', 'params_set', 'state_bounds', 'param_bounds', 'stim_params', 'step_stim_vals']
FLOAT_PARAMS = ['stim_l1', 'stim_ts']
STR_PARAMS = ['stim_file', 'stim_type', 'stim_smooth_type', 'meas_file']
DICT_PARAMS = ['stim_protocol']

class single_cell_FRET():
    """
    Class to hold single cell FRET functions and modules.
    """
    def __init__(self, **kwargs):
        ## Timetrace Variables
        self.nD = 2  ## Number of data dimensions
        self.dt = 0.1  ## Model timestep
        self.nT = 0  ## Number of model timepoints
        self.Tt = np.arange(0, self.dt * self.nT, self.dt)  ## List of timepoints (timetrace)
        self.L_idxs = [1]  ## Indices of the observable (non-hidden) components of the state vector

        # Stimulus Variables - TODO: make this into a dictionary
        self.stim = None  ## List of stimulus values at times given by Tt
        self.stim_file = None  ## Filename of stimulus file within the /stim/ directory
        self.stim_type = 'step'  ## Type of stimulus to be generated
        self.stim_smooth_dt = None  ## TODO: ??
        self.stim_smooth_type = 'gaussian'  ## TODO: ??
        self.stim_random_seed = 1  ## Seed for random generation of stimulus
        self.stim_no_changes = 30  ## Number of stim changes in timetrace
        self.stim_vals = [0.085, 0.1, 0.115]  ## Stimulus values used
        self.stim_bg = self.stim_vals[1]
        self.stim_params = None
        self.stim_l1 = 0  ## For block stimulus
        self.stim_l2 = None
        self.stim_ts = 10  ## For block stimulus

        # Variables for generating/loading measured data
        self.meas_data = None ## measured data of size (len(L_idxs), nT); can include NaN (if data measured sparsely)
        self.meas_file = None ## measured data file; default: forward integrating the model with the stimulus file
        self.meas_data_seed = 0 ## seed for measurement noise generation
        self.meas_noise = [0.01] ## Measurement noise for each observable component of state

        # Variables for integrating model
        self.model = models.MWC_MM()  ## Dynamical model proposed to explain data
        self.nP = self.model.nP ## Number of parameters (both fixed and fitted) that the model involves
        self.params_set = self.model.params['default']  ## Parameter set used for forward integration of the model
        self.true_states = None  ## Stores state of integrated system over the complete timetrace TODO: better name?
        self.x0 = self.model.x0 ## Initialized value of state for model to be integrated

        # Variables for estimation and prediction windows
        self.est_beg_T = None ## Time when estimation begins (default: 0)
        self.est_end_T = None ## Time when estimation ends (default: end of time series)
        self.pred_beg_T = None ## Time when prediction begins (default: when estimation ends)
        self.pred_end_T = None ## Time when prediction ends (default: end of time series)
        self.est_wind_idxs = None ## Indices of data that correspond to the estimation window
        self.pred_wind_idxs = None ## Indices of data that correspond to the prediction window

        # Variables for optimization of single annealing step
        self.state_bounds = self.model.state_bounds ## Bounds on the states of model, list of [lower bound, upper bound] elements
        self.param_bounds = self.model.param_bounds ## Bounds on the parameters of model, list of [lower bound, upper bound] elements
        self.bounds = None ## List with all bounds combined; used for input to annealing
        self.init_seed = 0 ## Seed to randomly initialize true states and parameters that are optimized in VA
        self.x_init = None ## True state initialization for optimization
        self.p_init = None ## Parameter initialization for optimization

        # Variables for annealing run
        self.alpha = 2.0 ## Multiplier by which Rf is increased in each annealing step
        self.beta_increment = 1 ## Rf increases by a factor of alpha to this value at each step
        self.beta_array = range(0, 61, self.beta_increment) ## Exponents used to calculate Rf
        self.Rf0 = 1e-6 ## Initial Rf

        # Overwrite variables with passed arguments
        if 'model' in kwargs:
            model_name = kwargs['model']
            assert hasattr(models, '%s' % model_name), 'Model class "%s" not in models module' % model_name
            exec('self.model = models.%s()' % model_name)
            for attr in MODEL_DEP_PARAMS:
                exec('self.%s = self.model.%s' % (attr, attr))

        for key in kwargs:
            val = kwargs[key]
            if key in INT_PARAMS:
                exec('self.%s = int(val)' % key)
            elif key in STR_PARAMS:
                exec('self.%s = str(val)' % key)
            elif key in FLOAT_PARAMS:
                exec('self.%s = float(val)' % key)
            elif key in LIST_PARAMS:
                exec('self.%s = %s' % (key, val))
            else:
                print("Parameter not recognized.")
                sys.exit(1)

        self.set_stim()
        self.set_meas_data()
        self.set_est_pred_windows()
    #############################################
    ##### 		Stimulus functions			#####
    #############################################
    def set_stim(self):
        """
        Set the stimulus vector, either from file or generated from a function.
        """
        if self.stim_file is not None:
            self.import_stim_data()
            print('Stimulus data imported from %s.stim.' % self.stim_file)
        else:
            self.generate_stim()
            print('Stimulus data generated from inputs.')

    def import_stim_data(self):
        """
        Import stimulus from file.
        """
        Tt_stim = load_stim_file(self.stim_file)
        self.Tt = Tt_stim[:, 0]
        self.dt = self.Tt[1] - self.Tt[0]
        self.nT = len(self.Tt)
        self.stim = Tt_stim[:, 1]

    def generate_stim(self): ## TODO: check this
        """
        Generate a stimulus of type stim_type.
        'random_switch': switch stimulus to new random level a set number of times at random points
        """
        ## Stochastic
        stim_type = self.stim_type
        if stim_type == 'random_switch':
            assert self.stim_no_changes < self.nT, "Number of stimulus changes must be " \
                                                   "less than number of time points, but # changes = %s, # time " \
                                                   "points = %s" \
                                                   % (self.nT, self.stim_no_changes)
            self.stim = np.zeros(self.nT)
            # Get points at which to switch the step stimulus
            np.random.seed(self.stim_random_seed)
            switch_pts = np.sort(np.random.choice(self.nT, self.stim_no_changes))
            # Set values in each inter-switch interval from step_stim_vals array
            for i in range(self.stim_no_changes - 1):
                stim_val = np.random.choice(self.stim_vals)
                self.stim[switch_pts[i]: switch_pts[i + 1]] = stim_val
            # Fill in ends with background
            self.stim[:switch_pts[0]] = self.stim_bg
            self.stim[switch_pts[-1]:] = self.stim_bg

        elif stim_type == 'stochastic':
            ## Create transition matrix
            levels = len(self.stim_vals)
            transtemp = np.zeros((levels, levels))
            p = self.stim_no_changes / self.nT
            sensible = collections.deque([1 - p] + [p / (levels - 1)] * (levels - 1))
            transition = np.copy(transtemp)
            for i in range(levels):
                transition[i] = list(sensible)
                sensible.rotate(1)
            self.stim = np.zeros(self.nT)
            self.stim[0] = self.stim_bg
            for i in range(1, self.nT):
                self.stim[i] = np.random.choice(self.stim_vals, p=transition[self.stim[i - 1]])

        elif stim_type == 'block':  ## TODO: Finish other stimulus inputs; clean this up too
            ## t_s, dl_1, dl_2 must be kwargs
            self.stim = np.zeros(self.nT)
            adapt_time = 30  ## time given for adaptation
            pre_stim = 10
            t_s, l1 = self.stim_params
            if self.stim_l2 == None:
                l2 = -self.stim_l1
            else:
                l2 = self.stim_l2
            block = np.concatenate((np.ones(int(pre_stim / self.dt)) * self.stim_bg,
                                    np.ones(int(t_s / self.dt)) * (self.stim_bg + l1),
                                    np.ones(int(adapt_time / self.dt)) * self.stim_bg))
            repeats = self.nT // len(block)
            for i in range(repeats):
                self.stim[i * len(block):(i + 1) * len(block)] = block
            print(len(self.Tt), len(self.stim))

    def eval_stim(self, t):
        """
        Evaluate the stimulus at arbitrary times by interpolated self.stim
        """
        assert self.stim is not None, 'Set stim before smoothing w/ set_stim()'
        return interp1d(self.Tt, self.stim, fill_value='extrapolate')(t)

    #############################################
    #####          Measured Data            #####
    #############################################

    def set_meas_data(self):
        """
        Set the meas data from file (if meas_file set) or generate from true parameters.
        """
        if self.meas_file is not None:
            self.import_meas_data()
            print('Measured data imported from %s.meas.' % self.meas_file)
        else:
            assert self.stim is not None, "No stimulus file specified to forward integrate model with."
            self.forward_integrate()
            self.meas_data = np.zeros((self.nT, len(self.L_idxs)))
            np.random.seed(self.meas_data_seed)
            assert len(self.meas_noise) == len(self.L_idxs), "meas_noise must " \
                                                             "be a list of length L_idxs = %s" % len(self.L_idxs)
            ## TODO: allow measurement noise to vary over time
            for iL_idx, iL in enumerate(self.L_idxs):
                self.meas_data[:, iL_idx] = self.true_states[:, iL] + np.random.normal(0, self.meas_noise[iL_idx], self.nT)

    def import_meas_data(self):
        """
		Import measured data from file.
		"""
        Tt_meas_data = load_meas_file(self.meas_file)
        self.meas_data = Tt_meas_data[:, 1:] ## to remove time column
        ## TODO: allow the observed timetrace to be different from the input stimulus timetrace.
        assert (np.all(Tt_meas_data[:, 0] == self.Tt)), "Tt vector in " \
                                                        "measured data file must be same as Tt"
        assert self.meas_data.shape[1] == len(self.L_idxs), "Dimension of " \
                                                            "imported measurement vectors must be same as length of " \
                                                            "L_idxs "

    #############################################
    #####			VA parameters			#####
    #############################################

    def set_init_est(self):
        print('Initializing estimate with seed %s' % self.init_seed)
        assert (self.nD == self.model.nD), 'self.nD != %s' % self.model.nD
        assert (self.nP == self.model.nP), 'self.nP != %s' % self.model.nP
        self.bounds = np.vstack((self.state_bounds, self.param_bounds))
        self.x_init = np.zeros((self.nT, self.nD))
        self.p_init = np.zeros(self.nP)

        ## Generate random initial states within state bounds over timetrace
        np.random.seed(self.init_seed)
        for iD in range(self.nD):
            self.x_init[:, iD] = np.random.uniform(self.state_bounds[iD][0],
                                                   self.state_bounds[iD][1], self.nT)
        ## Generate random initial parameters in parameter bounds
        for iP in range(self.nP):
            self.p_init[iP] = np.random.uniform(self.param_bounds[iP][0],
                                                self.param_bounds[iP][1])

    def df_estimation(self, t, x, inputs):
        """
		Function to return value of vector field in format used for varanneal
		"""
        p, stim = inputs
        return self.model.df(t, x, (p, stim))

    def set_est_pred_windows(self):
        if self.est_beg_T == None:
            print("Assuming estimation begins at first timepoint.")
            self.est_beg_T = 0
        if self.est_end_T == None:
            print("Assuming estimation ends at last timepoint.")
            self.est_end_T = (self.nT-1) * self.dt
        if self.pred_beg_T == None:
            self.pred_beg_T = self.est_end_T
        if self.pred_end_T == None:
            print("Assuming prediction ends at last timepoint.")
            self.pred_end_T = (self.nT-1) * self.dt

        est_beg_idx = int(self.est_beg_T / self.dt)
        est_end_idx = int(self.est_end_T / self.dt)
        pred_beg_idx = int(self.pred_beg_T / self.dt)
        pred_end_idx = int(self.pred_end_T / self.dt)
        self.est_wind_idxs = np.arange(est_beg_idx, est_end_idx)
        self.pred_wind_idxs = np.arange(pred_beg_idx, pred_end_idx)

        if len(self.est_wind_idxs) == 0:
            print('WARNING: Estimation window has len 0; change est_beg_T and/or est_end_T')
        if len(self.pred_wind_idxs) == 0:
            print('WARNING: Prediction window has len 0; change est_end_T and/or pred_end_T')

    #############################################
    #####	Numerical Integration of Model	#####
    #############################################

    def df_data_generation(self, x, t, p):
        """
        Function to return value of vector field in format used for sp.odeint
        """
        return self.model.df(t, x, (p, self.eval_stim(t)))

    def forward_integrate(self):
        """
        Forward integrate the model given true parameters and x0.
        """
        assert len(self.x0) == self.nD, "Initial state has dimension %s != " \
                                        "model dimension %s" % (len(self.x0), self.nD)
        self.true_states = odeint(self.df_data_generation, self.x0, self.Tt,
                                  args=(self.params_set,))

def create_cell_from_mat(dir, mat_file, cell):
    """Save stimulus and measurement files from FRET recording"""
    data = load_FRET_recording(dir, mat_file, cell)
    a = single_cell_FRET()
    a.stim = data['stim']
    a.Tt = data['Tt']
    a.meas_data = data['FRET']
    spec_name = '%s_cell_%s' % (dir.replace('/', '_'), cell)
    save_stim(a, spec_name)
    save_meas_data(a, spec_name)