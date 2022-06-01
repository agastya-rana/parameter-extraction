"""
Single-cell FRET class that holds stimulus, model and inference parameters.

Created by Nirag Kadakia and Agastya Rana, 10-21-2021.

Cleaned
"""

import numpy as np
import sys
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from src.load_data import load_stim_file, load_meas_file
import src.models as models
import collections
from src.load_data import load_FRET_recording
from src.save_data import save_stim, save_meas_data

MODEL_DEP_PARAMS = ['nD', 'nP', 'L_idxs', 'P_idxs', 'state_bounds', 'param_bounds', 'params_set', 'x0', 'dt']
INT_PARAMS = ['nT', 'est_beg_T', 'est_end_T', 'pred_end_T', 'data_skip']
FLOAT_PARAMS = ['dt'] ## TODO: enforce an input with this to change the model parameter
LIST_PARAMS = ['x0', 'beta_array', 'params_set', 'state_bounds', 'param_bounds',
               'Tt_data', 'stim_protocol']
STR_PARAMS = ['stim_file', 'meas_file']
NP_PARAMS = ['meas_noise']
MODEL_INFL_PARAMS = ['dt', 'x0', 'param_bounds']


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
        self.Tt_data = None
        self.L_idxs = [1]  ## Indices of the observable (non-hidden) components of the state vector

        # Stimulus Variables
        self.stim = None  ## List of stimulus values at times given by Tt
        self.stim_file = None  ## Filename of stimulus file within the /stim/ directory

        ## Stimulus generation variables
        self.stim_protocol = {
            'type': 'step',  ## Type of stimulus to be generated
            'random_seed': 1,  ## Seed for random generation of stimulus
            'no_changes': 30,  ## Number of stim changes in timetrace
            'vals': [0.085, 0.1, 0.115],  ## Stimulus values used
            'bg': [0.1],
        }

        # Variables for generating/loading measured data
        self.meas_data = None  ## measured data of size (nT, len(L_idxs)); can include NaN (if data measured sparsely)
        self.meas_file = None  ## measured data file; default: forward integrating the model with the stimulus file
        self.meas_data_seed = 0  ## seed for measurement noise generation
        self.meas_noise = 0.01 * np.ones(
            (len(self.L_idxs),))  ## Measurement noise for each observable component of state

        # Variables for integrating model
        self.model = models.MWC_MM()  ## Dynamical model proposed to explain data
        self.nP = self.model.nP  ## Number of parameters (both fixed and fitted) that the model involves
        self.P_idxs = self.model.P_idxs
        self.params_set = self.model.params_set  ## Parameter set used for forward integration of the model
        self.true_states = None  ## Stores state of integrated system over the complete timetrace
        self.x0 = self.model.x0  ## Initialized value of state for model to be integrated

        # Variables for estimation and prediction windows
        self.est_beg_T = None  ## Time when estimation begins (default: 0)
        self.est_end_T = None  ## Time when estimation ends (default: end of time series)
        self.pred_beg_T = None  ## Time when prediction begins (default: when estimation ends)
        self.pred_end_T = None  ## Time when prediction ends (default: end of time series)
        self.est_wind_idxs = None  ## Indices of stimulus that correspond to the estimation window
        self.pred_wind_idxs = None  ## Indices of stimulus that correspond to the prediction window
        self.est_data_wind_idxs = None  ## Indices of data that correspond to the estimation window
        self.pred_data_wind_idxs = None  ## Indices of data that correspond to the prediction window

        # Variables for optimization of single annealing step
        self.state_bounds = self.model.state_bounds  ## Bounds on the states of model, list of [lower bound, upper bound] elements
        self.param_bounds = self.model.param_bounds  ## Bounds on the parameters of model, list of [lower bound, upper bound] elements
        self.bounds = None  ## List with all bounds combined; used for input to annealing
        self.init_seed = 0  ## Seed to randomly initialize true states and parameters that are optimized in VA
        self.x_init = None  ## True state initialization for optimization
        self.p_init = None  ## Parameter initialization for optimization

        # Variables for annealing run
        self.alpha = 2.0  ## Multiplier by which Rf is increased in each annealing step
        self.beta_increment = 1  ## Rf increases by a factor of alpha to this value at each step
        self.beta_array = range(0, 61, self.beta_increment)  ## Exponents used to calculate Rf
        self.Rf0 = 1e-6  ## Initial Rf ## TODO: imagine array of Rf to differentiate between error in a and m

        # Overwrite variables with passed arguments
        if 'model' in kwargs:
            model_name = kwargs['model']
            assert hasattr(models, '%s' % model_name), 'Model class "%s" not in models module' % model_name
            exec('self.model = models.%s()' % model_name)
            for param in MODEL_DEP_PARAMS:
                if param in kwargs:
                    exec('self.model.%s = %s' % (param, kwargs[param]))
            for attr in MODEL_DEP_PARAMS:
                exec('self.%s = self.model.%s' % (attr, attr))
        else:
            print("Need to include model used for integration.")
            sys.exit(1)

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
            elif key in NP_PARAMS:
                exec('self.%s = np.asarray(%s)' % (key, val))
            elif key == 'model':
                pass
            else:
                print("Parameter %s not recognized." % key)
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
            ## Need to ensure that self.dt and self.nT well defined
            assert self.nT > 0, "Number of timepoints needs to be specified."
            self.Tt = np.arange(0, self.dt * self.nT, self.dt)
            if self.Tt_data is None:
                self.Tt_data = self.Tt
            self.generate_stim()
            print('Stimulus data generated from inputs.')

    def import_stim_data(self):
        """
        Import stimulus from file.
        """
        Tt_stim = load_stim_file(self.stim_file)
        self.Tt = Tt_stim[:, 0]
        if self.Tt_data is None:
            self.Tt_data = self.Tt
        self.dt = self.Tt[1] - self.Tt[0]
        self.nT = len(self.Tt)
        self.stim = Tt_stim[:, 1]

    def generate_stim(self):
        """
        Generate a stimulus of type stim_protocol[stim_type].
        'random_switch': switch stimulus to new random level a set number of times at random points
        'stochastic': switch stimulus stochastically at each time point such that approx stim_no_changes happen
        'block': blocks starting at bg, going up to bg+l1, down to bg+l2, where l1,l2 can vary for each block
        """
        try:
            type = self.stim_protocol['stim_type']
        except:
            print("Stimulus type not specified in stim dictionary")
            sys.exit(1)

        if type == 'random_switch':
            try:
                no_changes = self.stim_protocol['no_changes']
                vals = self.stim_protocol['vals']
                bg = self.stim_protocol['bg']
                seed = self.stim_protocol['seed']
            except:
                print("Stimulus parameters not correctly supplied for stim type %s" % type)
                sys.exit(1)

            assert no_changes < self.nT, "Number of stimulus changes must be " \
                                         "less than number of time points, but # changes = %s, # time " \
                                         "points = %s" \
                                         % (self.nT, no_changes)
            self.stim = np.zeros(self.nT)
            # Get points at which to switch the step stimulus
            np.random.seed(seed)
            switch_pts = np.sort(np.random.choice(self.nT, no_changes))
            # Set values in each inter-switch interval from step_stim_vals array
            for i in range(no_changes - 1):
                stim_val = np.random.choice(vals)
                self.stim[switch_pts[i]: switch_pts[i + 1]] = stim_val
            # Fill in ends with background
            self.stim[:switch_pts[0]] = bg
            self.stim[switch_pts[-1]:] = bg

        elif type == 'stochastic':
            try:
                no_changes = self.stim_protocol['no_changes']
                vals = self.stim_protocol['vals']
                bg = self.stim_protocol['bg']
            except:
                print("Stimulus parameters not correctly supplied for stim type %s" % type)
                sys.exit(1)
            ## Create transition matrix
            levels = len(vals)
            transtemp = np.zeros((levels, levels))
            p = no_changes / self.nT
            sensible = collections.deque([1 - p] + [p / (levels - 1)] * (levels - 1))
            transition = np.copy(transtemp)
            for i in range(levels):
                transition[i] = list(sensible)
                sensible.rotate(1)
            self.stim = np.zeros(self.nT)
            self.stim[0] = bg
            for i in range(1, self.nT):
                self.stim[i] = np.random.choice(vals, p=transition[self.stim[i - 1]])

        elif type == 'block':
            try:
                bg = self.stim_protocol['stim_bg']
            except:
                print("Stimulus parameters not correctly supplied for stim type %s" % type)
                sys.exit(1)
            ## t_s, dl_1, dl_2 must be kwargs
            self.stim = np.zeros(self.nT)
            try:
                adapt_time = self.stim_protocol['adapt_time']
                pre_stim = self.stim_protocol['pre_stim']
                t_s = self.stim_protocol['t_s']
            except:
                adapt_time = 20  ## time given for adaptation
                pre_stim = 20
                t_s = 5
            block_time = adapt_time + pre_stim + t_s
            blocks = self.nT // (block_time / self.dt)
            try:
                l1 = self.stim_protocol['l1']
                l2 = self.stim_protocol['l2']
            except:
                assert self.stim_protocol['vals'], 'Stimulus values to generate blocks do not exist'
                vals = self.stim_protocol['vals']
                l1 = np.random.choice(vals, blocks) - bg
                l2 = np.random.choice(vals, blocks) - bg
            for i in range(blocks):
                block = np.concatenate((np.ones(int(pre_stim / self.dt)) * bg,
                                        np.ones(int(t_s / self.dt)) * (bg + l1[i]),
                                        np.ones(int(adapt_time / self.dt)) * (bg + l2[i])))
                self.stim[i * len(block):(i + 1) * len(block)] = block
            print(len(self.Tt), len(self.stim))
        if self.Tt_data is None:
            self.Tt_data = self.Tt

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
            if self.meas_noise.shape == (len(self.L_idxs),):
                self.meas_noise = np.resize(self.meas_noise, (len(self.Tt_data), len(self.L_idxs)))
            print('Measured data imported from %s.meas.' % self.meas_file)
        else:
            assert self.stim is not None, "No stimulus file specified to forward integrate model with."
            self.forward_integrate()
            assert self.Tt_data[-1] == self.Tt[-1], "Last element of timetrace should have a measurable value"
            self.meas_data = np.zeros((len(self.Tt_data), len(self.L_idxs)))
            self.data_idxs = np.searchsorted(self.Tt, self.Tt_data)
            assert np.all(self.Tt[self.data_idxs] == self.Tt_data), "Tt_data not compatible with Tt"
            np.random.seed(self.meas_data_seed)
            if self.meas_noise.shape == (len(self.L_idxs),):
                self.meas_noise = np.resize(self.meas_noise, (len(self.Tt_data), len(self.L_idxs)))
            for iL_idx, iL in enumerate(self.L_idxs):
                self.meas_data[:, iL_idx] = self.true_states[self.data_idxs, iL] + \
                                            np.random.normal(0, self.meas_noise[:, iL_idx], len(self.Tt_data))

    def import_meas_data(self):
        """
        Import measured data from file.
        """
        Tt_meas_data = load_meas_file(self.meas_file)
        self.Tt_data = Tt_meas_data[:, 0]
        self.data_idxs = np.searchsorted(self.Tt, self.Tt_data)
        self.meas_data = Tt_meas_data[:, 1:]  ## to remove time column
        if not np.all(self.Tt_data == self.Tt):
            print("Usecase: sparse data")
        assert self.meas_data.shape[1] == len(self.L_idxs), "Measured data imported not of sufficient dimension."

    #############################################
    #####			VA parameters			#####
    #############################################

    def set_init_est(self):
        print('Initializing estimate with seed %s' % self.init_seed)
        self.bounds = np.vstack((self.state_bounds, self.param_bounds)) ## only for variable params
        self.x_init = np.zeros((self.nT, self.nD))
        self.p_init = np.zeros(len(self.P_idxs))

        ## Generate random initial states within state bounds over timetrace
        np.random.seed(self.init_seed)
        for iD in range(self.nD):
            self.x_init[:, iD] = np.random.uniform(self.state_bounds[iD][0], self.state_bounds[iD][1], self.nT)
        for iP in range(len(self.P_idxs)):
            self.p_init[iP] = np.random.uniform(self.param_bounds[iP][0], self.param_bounds[iP][1])

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
            self.est_end_T = (self.nT - 1) * self.dt
        if self.pred_beg_T == None:
            self.pred_beg_T = self.est_end_T
        if self.pred_end_T == None:
            print("Assuming prediction ends at last timepoint.")
            self.pred_end_T = (self.nT - 1) * self.dt

        assert self.est_beg_T in self.Tt_data, "Estimation must begin at data timepoint."
        assert self.est_end_T in self.Tt_data, "Estimation must end at data timepoint."
        assert self.pred_beg_T in self.Tt_data, "Prediction must begin at data timepoint."
        assert self.pred_end_T in self.Tt_data, "Prediction must end at data timepoint."

        est_beg_idx = np.searchsorted(self.Tt, self.est_beg_T)
        est_end_idx = np.searchsorted(self.Tt, self.est_end_T)
        pred_beg_idx = np.searchsorted(self.Tt, self.pred_beg_T)
        pred_end_idx = np.searchsorted(self.Tt, self.pred_end_T)
        self.est_wind_idxs = np.arange(est_beg_idx, est_end_idx+1)
        self.pred_wind_idxs = np.arange(pred_beg_idx, pred_end_idx+1)

        est_data_beg_idx = np.searchsorted(self.Tt_data, self.est_beg_T)
        est_data_end_idx = np.searchsorted(self.Tt_data, self.est_end_T)
        pred_data_beg_idx = np.searchsorted(self.Tt_data, self.pred_beg_T)
        pred_data_end_idx = np.searchsorted(self.Tt_data, self.pred_end_T)
        self.est_data_wind_idxs = np.arange(est_data_beg_idx, est_data_end_idx+1)
        self.pred_data_wind_idxs = np.arange(pred_data_beg_idx, pred_data_end_idx+1)

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
        est_params = [self.params_set[i] for i in self.P_idxs]
        self.true_states = odeint(self.df_data_generation, self.x0, self.Tt,
                                  args=(est_params,))


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
