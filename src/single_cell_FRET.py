"""
Single-cell FRET class that holds stimulus, model and inference parameters.

Created by Nirag Kadakia and Agastya Rana, 10-21-2021.
"""

import scipy as sp
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import scipy.linalg as LA
from load_data import load_stim_file, load_meas_file
import models
import collections
from utils import smooth_vec

INT_PARAMS = ['nT', 'nD', 'nP', 'step_stim_density', 'step_stim_seed']
LIST_PARAMS = ['L_idxs', 'x0', 'step_stim_vals', 'P_idxs', 'beta_array', 'meas_noise', 'params_set', 'stim_params']
MODEL_PARAMS = ['model']
FLOAT_PARAMS = ['stim_l1', 'stim_ts']
STR_PARAMS = ['stim_file', 'stim_type', 'stim_smooth_type', 'bounds_set', 'meas_file']


class single_cell_FRET():
    """
    Class to hold single cell FRET functions and modules.
    """

    def __init__(self, **kwargs):
        ## Timetrace Variables
        self.nD = 2  ## Number of dimensions
        self.dt = 0.1  ## Timestep
        self.nT = 767  ## Number of timepoints
        self.Tt = np.arange(0, self.dt * self.nT, self.dt)  ## List of timepoints (timetrace)
        self.L_idxs = [1]  ## Index of the observable (non-hidden) components of the state vector

        # Stimulus Variables
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
        self.stim_l1 = 0 ## For block stimulus
        self.stim_l2 = None
        self.stim_ts = 10 ## For block stimulus

        # Variables for generating/loading measured data
        self.meas_data = None
        self.meas_file = None
        self.meas_data_seed = 0
        self.meas_noise = [1.0]

        # Variables for integrating model/twin data generation
        self.model = models.MWC_MM()  ## Default is MWC MM model
        self.params_set = []  ## Value of true parameter set used in the model (used for forward integration)
        self.bounds_set = 'Default'  ## Name of the bounds set used in the model inference algorithm
        self.true_states = None  ## Stores state of integrated system over the complete timetrace
        self.x0 = [2.3, 7.0]  ## TODO: why?

        # Variables for estimation and prediction windows
        self.est_beg_T = None
        self.est_end_T = None
        self.pred_end_T = None
        self.est_wind_idxs = None
        self.pred_wind_idxs = None

        # Variables for optimization of single annealing step
        self.nP = 9  ## TODO: why shouldn't this just be the model values?
        self.param_bounds = None
        self.state_bounds = None
        self.bounds = None
        self.init_seed = 0
        self.x_init = None
        self.p_init = None

        # Variables for annealing run
        self.alpha = 2.0
        self.beta_array = range(0, 61)
        self.Rf0 = 1e-6

        # Variables for linear kernel estimation
        self.kernel_length = 50
        self.regularization = 1.0
        self.kernel_tikhonov_matrix = None
        self.kernel_inverse_hessian = None
        self.kernel_Aa = None
        self.kernel_estimator_nT = self.nT - self.kernel_length

        # Overwrite variables with passed arguments
        for key in kwargs:
            assert hasattr(self, '%s' % key), "'%s' is not an attribute of " \
                                              "the single_cell_FRET class. Check or add to __init__" % key
            val = kwargs[key]

            if key in INT_PARAMS:
                exec('self.%s = int(val)' % key)
            elif key in STR_PARAMS:
                exec('self.%s = str(val)' % key)
            elif key in FLOAT_PARAMS:
                exec('self.%s = float(val)' % key)
            elif key in MODEL_PARAMS:  ## Need value to be a model from models class
                assert hasattr(models, '%s' % val), 'Model class "%s" not in ' \
                                                    'models module' % val
                exec('self.%s = models.%s()' % (key, val))
            elif key in LIST_PARAMS:
                try:
                    exec('len(%s)' % val)
                except:
                    print('The value of %s (%s) is not a list' % (key, val))
                    quit()
                exec('self.%s = %s' % (key, val))
            else:
                exec('self.%s = float(val)' % key)



    #############################################
    ##### 		Stimulus functions			#####
    #############################################

    def set_stim(self):
        """
        Set the stimulus vector, either from file or generated from a function.
        """
        print("In set_stim, stim file is", self.stim_file)
        if self.stim_file is not None:
            self.import_stim_data()
            print('Stimulus data imported from %s.stim.' % self.stim_file)
        else:
            try:
                self.generate_stim()
            except:
                print('Stimulus type stim_type=%s unknown' % self.stim_type)
                quit()

        if self.stim_smooth_dt is not None:
            window_len = int(1. * self.stim_smooth_dt / self.dt)
            self.stim = smooth_vec(self.stim, window_len, self.stim_smooth_type)

    def import_stim_data(self):
        """
        Import stimulus from file.
        """
        Tt_stim = load_stim_file(self.stim_file)
        assert Tt_stim.shape[0] == self.nT, "Loaded stimulus file has %s " \
                                            "timepoints, but nT is set to %s" % (Tt_stim.shape[0], self.nT)
        assert Tt_stim[1, 0] - Tt_stim[0, 0] == self.dt, "Loaded stimulus " \
                                                         "file has timestep %s, but self.dt is %s" % (
                                                             Tt_stim[1, 0] - Tt_stim[0, 0], self.dt)
        self.Tt = Tt_stim[:, 0]
        self.stim = Tt_stim[:, 1]

    def generate_stim(self):
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
            for i in range(1, dT):
                self.stim[i] = np.random.choice(self.stim_vals, p=transition[self.stim[i - 1]])

        elif stim_type == 'block':  ## TODO: Finish other stimulus inputs; clean this up too
            ## t_s, dl_1, dl_2 must be kwargs
            self.stim = np.zeros(self.nT)
            adapt_time = 30 ## time given for adaptation
            pre_stim = 10
            t_s, l1 = self.stim_params
            if self.stim_l2 == None:
                l2 = -self.stim_l1
            else:
                l2 = self.stim_l2
            block = np.concatenate((np.ones(int(pre_stim/self.dt))*self.stim_bg,
                                    np.ones(int(t_s/self.dt))*(self.stim_bg+l1),
                                    np.ones(int(adapt_time/self.dt))*self.stim_bg))
            repeats = self.nT//len(block)
            for i in range(repeats):
                self.stim[i*len(block):(i+1)*len(block)] = block
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
            assert self.true_states is not None, "Since no measurement file " \
                                                 "is specified, twin data will be generated. But before calling " \
                                                 "set_meas_data(), you must first generate true data with " \
                                                 "gen_true_states."

            self.meas_data = sp.zeros((self.nT, len(self.L_idxs)))
            sp.random.seed(self.meas_data_seed)

            assert len(self.meas_noise) == len(self.L_idxs), "meas_noise must " \
                                                             "be a list of length L_idxs = %s" % len(self.L_idxs)

            ## TODO: allow measurement noise to vary over time

            for iL_idx, iL in enumerate(self.L_idxs):
                self.meas_data[:, iL_idx] = self.true_states[:, iL] + \
                                            sp.random.normal(0, self.meas_noise[iL_idx], self.nT)

    def import_meas_data(self):
        """
		Import measured data from file
		"""
        Tt_meas_data = load_meas_file(self.meas_file)
        self.meas_data = Tt_meas_data[:, self.L_idxs]
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
        self.state_bounds = self.model.bounds[self.bounds_set]['states']
        self.param_bounds = self.model.bounds[self.bounds_set]['params']
        self.bounds = np.vstack((self.state_bounds, self.param_bounds))
        self.x_init = np.zeros((self.nT, self.nD))
        self.p_init = np.zeros(self.nP)

        ## Generate random states within state bounds over timetrace
        np.random.seed(self.init_seed)
        for iD in range(self.nD):
            self.x_init[:, iD] = np.random.uniform(self.state_bounds[iD][0],
                                                   self.state_bounds[iD][1], self.nT)
        ## Generate random parameters in parameter bounds
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

        assert self.est_beg_T is not None, "Before setting estimation and " \
                                           "prediction windows, set est_beg_T"
        assert self.est_end_T is not None, "Before setting estimation and " \
                                           "prediction windows, set est_end_T"
        assert self.pred_end_T is not None, "Before setting estimation and " \
                                            "prediction windows, set pred_end_T"

        est_beg_idx = int(self.est_beg_T / self.dt)
        est_end_idx = min(int(self.est_end_T / self.dt), self.nT - 1)
        pred_end_idx = min(int(self.pred_end_T / self.dt), self.nT - 1)
        self.est_wind_idxs = sp.arange(est_beg_idx, est_end_idx)
        self.pred_wind_idxs = sp.arange(est_end_idx, pred_end_idx)

        assert len(self.est_wind_idxs) > 0, \
            'Estimation window has len 0; change est_beg_T and/or est_end_T'
        assert len(self.pred_wind_idxs) > 0, \
            'Prediction window has len 0; change est_end_T and/or pred_end_T'

    #############################################
    #####	Numerical Integration of Model	#####
    #############################################

    def df_data_generation(self, x, t, p):
        """
        Function to return value of vector field in format used for sp.odeint
        """
        return self.model.df(t, x, (p, self.eval_stim(t)))

    def gen_true_states(self):
        """
        Forward integrate the model given true parameters and x0.
        """
        assert len(self.x0) == self.nD, "Initial state has dimension %s != " \
                                        "model dimension %s" % (len(self.x0), self.nD)
        self.true_states = odeint(self.df_data_generation, self.x0, self.Tt,
                                  args=(self.params_set,))

    #############################################
    #####		Kernel estimation			#####
    #############################################

    def set_kernel_estimation_Aa(self, stimulus=None):
        assert 0 < self.kernel_length < self.nT, \
            'Kernel length must be nonzero and shorter than recorded time length'

        if stimulus is None:
            assert self.stim is not None, \
                'Signal vector must first be imported or set by' \
                ' set_step_stim() or import_stim_data().' \
                ' Or pass directly to set_kernel_estimation_Aa(stimulus=)'
        else:
            assert len(stimulus) == self.nT, \
                'User-defined stimulus in set_kernel_estimation_Aa(stimulus)' \
                ' must be length %s' % self.nT
            self.stim = stimulus

        mean_subtracted_stimulus = self.stim - sp.average(self.stim)

        self.kernel_estimator_nT = self.nT - self.kernel_length
        self.kernel_Aa = sp.zeros((self.kernel_estimator_nT, self.kernel_length))
        for row in range(self.kernel_estimator_nT):
            self.kernel_Aa[row, :] = \
                mean_subtracted_stimulus[row: row + self.kernel_length]

    def set_kernel_estimation_regularization(self):
        self.kernel_tikhonov_matrix = sp.eye(self.kernel_length) * self.regularization

    def set_kernel_estimation_inverse_hessian(self):
        assert self.kernel_Aa is not None, \
            'First set kernel_Aa with set_kernel_estimation_Aa'

        self.kernel_inverse_hessian = \
            LA.inv(sp.dot(self.kernel_Aa.T, self.kernel_Aa) \
                   + self.kernel_tikhonov_matrix)

    def kernel_calculation(self, response_vector):
        assert self.kernel_inverse_hessian is not None, \
            'First set inverse hessian for the estimation with' \
            ' set_inverse_hessian()'
        assert len(sp.shape(response_vector)) == 1 and \
               len(response_vector) == self.kernel_estimator_nT, \
            'response_vector should be numpy array with length equal to' \
            ' stimulus data minus kernel length'

        mean_subtracted_response_vector = response_vector - sp.average(response_vector)

        estimated_kernel = sp.dot(sp.dot(self.kernel_inverse_hessian,
                                         self.kernel_Aa.T), mean_subtracted_response_vector)

        return estimated_kernel

    def convolve_stimulus_kernel(self, kernel, stimulus=None):
        if stimulus is None:
            assert self.stim is not None, \
                'Signal vector must first be imported or set by' \
                ' set_step_stim() or import_stim_data().' \
                ' Or pass directly to set_kernel_estimation_Aa(stimulus=)'
        else:
            assert len(stimulus) == self.nT, \
                'User-defined stimulus in set_kernel_estimation_Aa(stimulus)' \
                ' must be length %s' % self.nT
            self.stim = stimulus

        mean_subtracted_stimulus = self.stim - sp.average(self.stim)
        self.kernel_length = len(kernel)
        mean_subtracted_response_vector = sp.zeros(self.nT)

        for iT in range(self.nT):
            if iT >= self.kernel_length:
                mean_subtracted_response_vector[iT] = \
                    sp.sum(mean_subtracted_stimulus[iT - self.kernel_length: iT] * kernel)
            else:
                mean_subtracted_response_vector[iT] = \
                    sp.sum(mean_subtracted_stimulus[:iT] * kernel[self.kernel_length - iT:])

        return mean_subtracted_response_vector
