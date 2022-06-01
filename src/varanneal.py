"""
Originally written by Paul Rozdeba (UCSD)
Modified and updated by Agastya Rana (Yale)

Carry out the variational annealing algorithm (VA) for estimating unobserved
dynamical model states and parameters from time series data. VA is a form of
variational data assimilation that uses numerical continuation to regularize
the variational cost function, or "action", in a controlled way. VA was first
proposed by Jack C. Quinn in his Ph.D. thesis (2010) [1], and is described by
 J. Ye et al. (2015) in detail in [2].

This code uses automatic differentiation to evaluate derivatives of the
action for optimization as implemented in autograd.

References:
[1] J.C. Quinn, "A path integral approach to data assimilation in stochastic
    nonlinear systems."  Ph.D. thesis in physics, UC San Diego (2010).
    Available at: https://escholarship.org/uc/item/0bm253qk

[2] J. Ye et al., "Improved variational methods in statistical data assimilation."
    Nonlin. Proc. in Geophys., 22, 205-213 (2015).
"""

import autograd.numpy as np
from autograd import grad, hessian
import time
import sys
import scipy.optimize as opt


class Annealer(object):

    def __init__(self):
        self.initialized = False  ## Whether VA has been initialized with the required parameters

    def set_model(self, f, D):
        """
        Set the D-dimensional dynamical model for the estimated system.
        The model function, f, must take arguments in the following order:
            t, x, p
        or, if there is a time-dependent stimulus for f (nonautonomous term):
            t, x, (p, stim)
        where x and stim are at the "current" time t.  Thus, x should be a
        D-dimensional vector, and stim similarly a D_stim-dimensional vector.
        """
        self.f = f  ## Dynamical model f(t, x, (p, stim))
        self.D = D  ## Dimension of the dynamical model

    def set_data(self, t_data, data, t_model, stim):
        """
        Pass in data and stim arrays to VA class.

        Args:
            t_data: np.array of shape (N_data,); elements are subset of t_model
            data: np.array of shape (N_data, D)
            t_model: np.array of shape (N_model,) with constant dt; t_model[-1] == t_data[-1]
            stim: np.array of shape (N_model, D_stim)
        """
        self.t_data = t_data
        self.t_model = t_model
        self.dt_model = t_model[1] - t_model[0]
        self.N_data = len(t_data)
        self.N_model = len(t_model)
        self.Y = data
        self.stim = stim
        self.data_idxs = np.searchsorted(self.t_model, self.t_data)
        assert len(data) == self.N_data, "Incorrect number of timepoints in t_data to match measured data"
        assert len(stim) == self.N_model, "Stimulus needs to be supplied for each timepoint in Tt_model"

    ############################################################################
    # Gaussian action
    ############################################################################
    def A_gaussian(self, XP):
        """
        Calculate the value of the Gaussian action.
        """
        merr = self.me_gaussian(XP[:self.N_model * self.D])
        ferr = self.fe_gaussian(XP)
        return merr + ferr

    def me_gaussian(self, X):
        """
        Gaussian measurement error.
        """
        x = np.reshape(X, (self.N_model, self.D))
        # print(x[self.data_idxs, self.Lidxs], self.Y.shape)
        # print(self.data_idxs)
        diff = x[self.data_idxs, self.Lidxs] - self.Y
        assert self.RM.shape == (self.N_data, self.L), "ERROR: RM is in an invalid shape."
        merr = np.sum(self.RM * np.square(diff))
        return merr  / (self.L * self.N_data) ## - removing bc we want total action, not averaged

    def fe_gaussian(self, XP):
        """
        Gaussian model error.
        """
        # Extract state and parameters from XP.
        x = np.reshape(XP[:self.N_model * self.D], (self.N_model, self.D))
        p = XP[self.N_model * self.D:]

        # Start calculating the model error.
        # First compute time series of error terms.
        if self.disc.__func__.__name__ == "disc_SimpsonHermite":
            disc_vec1, disc_vec2 = self.disc(x, p)
            diff1 = x[2::2] - x[:-2:2] - disc_vec1
            diff2 = x[1::2] - disc_vec2
        elif self.disc.__func__.__name__ == 'disc_forwardmap':
            diff = x[1:] - self.disc(x, p)
        else:
            diff = x[1:] - x[:-1] - self.disc(x, p)

        # Contract errors quadratically with RF.
        if type(self.RF) == np.ndarray:
            if self.RF.shape == (self.N_model - 1, self.D):
                if self.disc.__func__.__name__ == "disc_SimpsonHermite":
                    ferr1 = np.sum(self.RF[::2] * diff1 * diff1)
                    ferr2 = np.sum(self.RF[1::2] * diff2 * diff2)
                    ferr = ferr1 + ferr2
                else:
                    ferr = np.sum(self.RF * diff * diff)
            elif self.RF.shape == (self.N_model - 1, self.D, self.D):
                if self.disc.__func__.__name__ == "disc_SimpsonHermite":
                    ferr1 = 0.0
                    ferr2 = 0.0
                    for i in range((self.N_model - 1) / 2):
                        ferr1 = ferr1 + np.dot(diff1[i], np.dot(self.RF[2 * i], diff1[i]))
                        ferr2 = ferr2 + np.dot(diff2[i], np.dot(self.RF[2 * i + 1], diff2[i]))
                    ferr = ferr1 + ferr2
                else:
                    ferr = 0.0
                    for i in range(self.N_model - 1):
                        ferr = ferr + np.dot(diff[i], np.dot(self.RF[i], diff))
            else:
                print("ERROR: RF is in an invalid shape. Exiting.")
                sys.exit(1)
        else:
            if self.disc.__func__.__name__ == "disc_SimpsonHermite":
                ferr = self.RF * np.sum(diff1 * diff1 + diff2 * diff2)
            else:
                ferr = self.RF * np.sum(diff * diff) ## USECASE

        return ferr / (self.D * (self.N_model - 1))  ## - remove to find total action

    ############################################################################
    # Discretization routines
    ############################################################################
    def disc_euler(self, x, p):
        """
        Euler's method for time discretization of f.
        """
        if self.stim is None:
            if self.P.ndim == 1:
                pn = p
            else:
                pn = p[:-1]
        else:
            if self.P.ndim == 1:
                pn = (p, self.stim[:-1])
            else:
                pn = (p[:-1], self.stim[:-1])

        return self.dt_model * self.f(self.t_model[:-1], x[:-1], pn)

    def disc_trapezoid(self, x, p):
        """
        Time discretization for the action using the trapezoid rule.
        """
        if self.stim is None:
            if self.P.ndim == 1:
                pn = p
                pnp1 = p
            else:
                pn = p[:-1]
                pnp1 = p[1:]
        else:
            if self.P.ndim == 1:
                pn = (p, self.stim[:-1])
                pnp1 = (p, self.stim[1:])
            else:
                pn = (p[:-1], self.stim[:-1])
                pnp1 = (p[1:], self.stim[1:])

        fn = self.f(self.t_model[:-1], x[:-1], pn)
        fnp1 = self.f(self.t_model[1:], x[1:], pnp1)
        return self.dt_model * (fn + fnp1) / 2.0

    # Don't use RK4 yet, still trying to decide how to implement with a stimulus.
    # def disc_rk4(self, x, p):
    #    """
    #    RK4 time discretization for the action.
    #    """
    #    if self.stim is None:
    #        pn = p
    #        pmid = p
    #        pnp1 = p
    #    else:
    #        pn = (p, self.stim[:-2:2])
    #        pmid = (p, self.stim[1:-1:2])
    #        pnp1 = (p, self.stim[2::2])
    #
    #    xn = x[:-1]
    #    tn = np.tile(self.t[:-1], (self.D, 1)).T
    #    k1 = self.f(tn, xn, pn)
    #    k2 = self.f(tn + self.dt/2.0, xn + k1*self.dt/2.0, pmid)
    #    k3 = self.f(tn + self.dt/2.0, xn + k2*self.dt/2.0, pmid)
    #    k4 = self.f(tn + self.dt, xn + k3*self.dt, pnp1)
    #    return self.dt * (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    def disc_SimpsonHermite(self, x, p):
        """
        Simpson-Hermite time discretization for the action.
        This discretization applies Simpson's rule to all the even-index time
        points, and a Hermite polynomial interpolation for the odd-index points
        in between.
        """
        if self.stim is None:
            if self.P.ndim == 1:
                pn = p
                pmid = p
                pnp1 = p
            else:
                pn = p[:-2:2]
                pmid = p[1:-1:2]
                pnp1 = p[2::2]
        else:
            if self.P.ndim == 1:
                pn = (p, self.stim[:-2:2])
                pmid = (p, self.stim[1:-1:2])
                pnp1 = (p, self.stim[2::2])
            else:
                pn = (p[:-2:2], self.stim[:-2:2])
                pmid = (p[1:-1:2], self.stim[1:-1:2])
                pnp1 = (p[2::2], self.stim[2::2])

        fn = self.f(self.t_model[:-2:2], x[:-2:2], pn)
        fmid = self.f(self.t_model[1:-1:2], x[1:-1:2], pmid)
        fnp1 = self.f(self.t_model[2::2], x[2::2], pnp1)

        disc_vec1 = (fn + 4.0 * fmid + fnp1) * (2.0 * self.dt_model) / 6.0
        disc_vec2 = (x[:-2:2] + x[2::2]) / 2.0 + (fn - fnp1) * (2.0 * self.dt_model) / 8.0

        return disc_vec1, disc_vec2

    def disc_forwardmap(self, x, p):  ## TODO: check if relevant
        """
        "Discretization" when f is a forward mapping, not an ODE.
        """
        if self.stim is None:
            if self.P.ndim == 1:
                pn = p
            else:
                pn = p[:-1]
        else:
            if self.P.ndim == 1:
                pn = (p, self.stim[:-1])
            else:
                pn = (p[:-1], self.stim[:-1])

        return self.f(self.t_model[:-1], x[:-1], pn)

    ############################################################################
    # Annealing functions
    ############################################################################
    def anneal(self, X0, P0, alpha, beta_array, RM, RF0, Lidx,
               init_to_data=True, action='A_gaussian', disc='trapezoid',
               method='L-BFGS-B', bounds=None, opt_args=None,
               track_paths=None, track_params=None, track_action_errors=None):
        """
        Convenience function to carry out a full annealing run over all values
        of beta in beta_array.
        """
        # Initialize the annealing procedure, if not already done.
        if self.initialized == False:
            self.anneal_init(X0, P0, alpha, beta_array, RM, RF0, Lidx,
                             init_to_data, action, disc, method, bounds,
                             opt_args)

        # Loop through all beta values for annealing.
        for i in beta_array:
            print('------------------------------')
            print('Step %d of %d' % (self.betaidx + 1, len(self.beta_array)))
            # Print RF
            if type(self.RF) == np.ndarray:
                if self.RF.shape == (self.N_model - 1, self.D):
                    print('beta = %d, RF[n=0, i=0] = %.8e' % (self.beta, self.RF[0, 0]))
                elif self.RF.shape == (self.N_model - 1, self.D, self.D):
                    print('beta = %d, RF[n=0, i=0, j=0] = %.8e' % (self.beta, self.RF[0, 0, 0]))
                else:
                    print("Error: RF has an invalid shape. You really shouldn't be here...")
                    sys.exit(1)
            else:
                print('beta = %d, RF = %.8e' % (self.beta, self.RF))
            print('')

            self.anneal_step()

            # Track progress by saving to file after every step
            if track_paths is not None:
                try:
                    dtype = track_paths['dtype']
                except:
                    dtype = np.float64
                try:
                    fmt = track_paths['fmt']
                except:
                    fmt = "%.8e"
                self.save_paths(track_paths['filename'], dtype, fmt)

            if track_params is not None:
                try:
                    dtype = track_params['dtype']
                except:
                    dtype = np.float64
                try:
                    fmt = track_params['fmt']
                except:
                    fmt = "%.8e"
                self.save_params(track_params['filename'], dtype, fmt)

            if track_action_errors is not None:
                try:
                    cmpt = track_action_errors['cmpt']
                except:
                    cmpt = 0
                try:
                    dtype = track_action_errors['dtype']
                except:
                    dtype = np.float64
                try:
                    fmt = track_action_errors['fmt']
                except:
                    fmt = "%.8e"
                self.save_action_errors(track_action_errors['filename'], cmpt, dtype, fmt)

    def anneal_init(self, X0, P0, alpha, beta_array, RM, RF0, Lidxs,
                    init_to_data=True, action='A_gaussian', disc='trapezoid',
                    method='L-BFGS-B', bounds=None, opt_args=None):
        """
        Initialize the annealing procedure.
        """
        if method not in ('L-BFGS-B', 'NCG', 'LM', 'TNC'):
            print("ERROR: Optimization routine not recognized. Annealing not initialized.")
            return None
        else:
            self.method = method

        # get optimization extra arguments
        self.opt_args = opt_args

        # set up parameters and determine if static or time series
        self.P = P0
        self.NP = len(P0)

        # get indices of measured components of f
        self.Lidxs = Lidxs
        self.L = len(Lidxs)

        # Store optimization bounds. Will only be used if the chosen
        # optimization routine supports it.
        if bounds is not None:
            self.bounds = []
            state_b = bounds[:self.D]
            param_b = bounds[self.D:]
            # set bounds on states for all N time points
            for n in range(self.N_model):
                for i in range(self.D):
                    self.bounds.append(state_b[i])
            # set bounds on estimated parameters
            for i in range(self.NP):
                self.bounds.append(param_b[i])
        else:
            self.bounds = None

        # Reshape RM and RF so that they span the whole time series, if they
        # are passed in as vectors or matrices.
        if type(RM) == list:
            RM = np.array(RM)
        if type(RM) == np.ndarray:
            if RM.shape == (self.L,):
                self.RM = np.resize(RM, (self.N_data, self.L))
            elif RM.shape == (self.N_data, self.L):
                self.RM = RM
            else:
                print("ERROR: RM has an invalid shape. Exiting.")
                sys.exit(1)
        else:
            print("ERROR: RM has an invalid shape. Exiting.")
            sys.exit(1)

        if type(RF0) == list:
            RF0 = np.array(RF0)
        if type(RF0) == np.ndarray:
            if RF0.shape == (self.D,):
                self.RF0 = np.resize(RF0, (self.N_model - 1, self.D))
            elif RF0.shape == (self.D, self.D):
                self.RF0 = np.resize(RF0, (self.N_model - 1, self.D, self.D))
            elif RF0.shape in [(self.N_model - 1, self.D), (self.N_model - 1, self.D, self.D)]:
                self.RF0 = RF0
            else:
                print("ERROR: RF0 has an invalid shape. Exiting.")
                sys.exit(1)
        else:
            self.RF0 = RF0

        # set up beta array in RF = RF0 * alpha**beta
        self.alpha = alpha
        self.beta_array = np.array(beta_array, dtype=np.uint16)
        self.Nbeta = len(self.beta_array)

        # set initial RF
        self.betaidx = 0
        self.beta = self.beta_array[self.betaidx]
        self.RF = self.RF0 * self.alpha ** self.beta

        # set the desired action
        if type(action) == str:
            exec('self.A = self.%s' % action)
        else:
            # Assumption: user has passed a function pointer
            self.A = action
        self.gradient = grad(self.A)
        self.hes = hessian(self.A)

        # set the discretization
        exec('self.disc = self.disc_%s' % disc)

        # array to store minimizing paths
        if P0.ndim == 1:  ## if parameters are static (not-time series)
            self.minpaths = np.zeros((self.Nbeta, self.N_model * self.D + self.NP), dtype=np.float64)
        else:
            if self.disc.__func__.__name__ in ["disc_euler", "disc_forwardmap"]:
                nmax_p = self.N_model - 1
            else:
                nmax_p = self.N_model
            self.minpaths = np.zeros((self.Nbeta, self.N_model * self.D + nmax_p * self.NP),
                                     dtype=np.float64)

        # initialize observed state components to data if desired
        if init_to_data:
            for iL_idx, iL in enumerate(self.Lidxs):
                X0[self.data_idxs, iL] = self.Y[:, iL_idx]

        # Flatten X0 and P0 into extended XP0 path vector
        XP0 = np.append(X0.flatten(), P0)
        self.minpaths[0] = XP0

        # array to store optimization results
        self.A_array = np.zeros(self.Nbeta, dtype=np.float64)
        self.me_array = np.zeros(self.Nbeta, dtype=np.float64)
        self.fe_array = np.zeros(self.Nbeta, dtype=np.float64)
        self.params_cov = np.zeros((self.Nbeta, self.NP, self.NP), dtype=np.float64)
        self.exitflags = np.empty(self.Nbeta, dtype=np.int8)

        # Initialization successful, we're at the beta = beta_0 step now.
        self.initalized = True

    def anneal_step(self):
        """
        Perform a single annealing step. The cost function is minimized starting
        from the previous minimum (or the initial guess, if this is the first
        step). Then, RF is increased to prepare for the next annealing step.
        """
        # minimize A using the chosen method
        ## Make XP0 vector only include variable parameters
        if self.method in ['L-BFGS-B', 'NCG', 'TNC', 'LM']:
            if self.betaidx == 0:
                XP0 = np.copy(self.minpaths[0])
            else:
                XP0 = np.copy(self.minpaths[self.betaidx - 1])

            print("Beginning optimization...")
            tstart = time.time()

            if self.method == 'L-BFGS-B':
                res = opt.minimize(self.A, XP0, method='L-BFGS-B', jac=self.gradient,
                                   options=self.opt_args, bounds=self.bounds)
            elif self.method == 'NCG':
                res = opt.minimize(self.A, XP0, method='CG', jac=self.gradient,
                                   options=self.opt_args, bounds=self.bounds)
            elif self.method == 'TNC':
                res = opt.minimize(self.A, XP0, method='TNC', jac=self.gradient,
                                   options=self.opt_args, bounds=self.bounds)
            else:
                print("You really shouldn't be here.  Exiting.")
                sys.exit(1)

            XPmin, exitflag, Amin = res.x, res.status, res.fun  ## Res.fun is function value at xmin
            cov = np.linalg.inv(self.hes(XPmin))
            covariance = cov[self.N_model * self.D:, self.N_model * self.D:]
            w, v = np.linalg.eig(covariance)
            err = np.array([np.sqrt(covariance[i, i]) for i in range(len(covariance))])
            print(w, err)
            flag = True
            for e in w:
                if e < 0:
                    pass
                    flag = False
            print(flag)

            print("Optimization complete!")
            print("Time = {0} s".format(time.time() - tstart))
            print("Exit flag = {0}".format(exitflag))
            print("Exit message: {0}".format(res.message))
            print("Iterations = {0}".format(res.nit))
            print("Obj. function value = {0}\n".format(Amin))
            ## BIC is nP ln(nT) + 2 A_min
        else:
            print("ERROR: Optimization routine not implemented or recognized.")
            sys.exit(1)

        # update optimal parameter values
        self.P = np.copy(XPmin[self.N_model * self.D:])

        # store A_min and the minimizing path
        self.A_array[self.betaidx] = Amin
        self.me_array[self.betaidx] = self.me_gaussian(np.array(XPmin[:self.N_model * self.D]))
        self.fe_array[self.betaidx] = self.fe_gaussian(np.array(XPmin))
        if flag:
            self.params_cov[self.betaidx] = covariance
        else:
            self.params_cov[self.betaidx, :, :] = np.nan
        self.minpaths[self.betaidx] = np.array(np.append(XPmin[:self.N_model * self.D], self.P))

        # increase RF
        if self.betaidx < len(self.beta_array) - 1:
            self.betaidx += 1
            self.beta = self.beta_array[self.betaidx]
            self.RF = self.RF0 * self.alpha ** self.beta

        # set flags indicating that we're no longer at the beginning of the annealing procedure
        if self.initialized:
            # Indicate no longer at beta_0
            self.initialized = False

    ################################################################################
    # Routines to save annealing results.
    ################################################################################
    def save_paths(self, filename, dtype=np.float64, fmt="%.8e"):
        """
        Save minimizing paths (not including parameters).
        """
        savearray = np.reshape(self.minpaths[:, :self.N_model * self.D], \
                               (self.Nbeta, self.N_model, self.D))

        # append time
        tsave = np.reshape(self.t_model, (self.N_model, 1))
        tsave = np.resize(tsave, (self.Nbeta, self.N_model, 1))
        savearray = np.dstack((tsave, savearray))

        if filename.endswith('.npy'):
            np.save(filename, savearray.astype(dtype))
        else:
            np.savetxt(filename, savearray, fmt=fmt)

    def save_params(self, filename, dtype=np.float64, fmt="%.8e"):
        """
        Save minimum action parameter values.
        """
        savearray = self.minpaths[:, self.N_model * self.D:]

        if filename.endswith('.npy'):
            np.save(filename, savearray.astype(dtype))
        else:
            np.savetxt(filename, savearray, fmt=fmt)
        return savearray[-1]

    def save_params_err(self, filename, dtype=np.float64, fmt="%.8e"):
        """
        Save minimum action parameter error values calculated from Hessian of action.
        """
        savearray = self.params_cov

        if filename.endswith('.npy'):
            np.save(filename, savearray.astype(dtype))
        else:
            np.savetxt(filename, savearray, fmt=fmt)

    def save_action_errors(self, filename, cmpt=0, dtype=np.float64, fmt="%.8e"):
        """
        Save beta values, action, and errors (with/without RM and RF) to file.
        cmpt sets which component of RF0 to normalize by.
        """
        savearray = np.zeros((self.Nbeta, 5))
        savearray[:, 0] = self.beta_array
        savearray[:, 1] = self.A_array
        savearray[:, 2] = self.me_array
        savearray[:, 3] = self.fe_array

        # Save model error / RF
        if type(self.RF) == np.ndarray:
            if self.RF0.shape == (self.N_model - 1, self.D):
                savearray[:, 4] = self.fe_array / (self.RF0[0, 0] * self.alpha ** self.beta_array)
            elif self.RF0.shape == (self.N_model - 1, self.D, self.D):
                savearray[:, 4] = self.fe_array / (self.RF0[0, 0, 0] * self.alpha ** self.beta_array)
            else:
                print("RF shape currently not supported for saving.")
                return 1
        else:
            savearray[:, 4] = self.fe_array / (self.RF0 * self.alpha ** self.beta_array)

        if filename.endswith('.npy'):
            np.save(filename, savearray.astype(dtype))
        else:
            np.savetxt(filename, savearray, fmt=fmt)
        return savearray

    def save_as_minAone(self, savedir='', savefile=None):
        """
        Save the result of this annealing in minAone data file style.
        """
        if savedir.endswith('/') == False:
            savedir += '/'
        if savefile is None:
            savefile = savedir + 'D%d_M%d_PATH%d.dat' % (self.D, self.L)
        else:
            savefile = savedir + savefile
        betaR = self.beta_array.reshape((self.Nbeta, 1))
        exitR = self.exitflags.reshape((self.Nbeta, 1))
        AR = self.A_array.reshape((self.Nbeta, 1))
        savearray = np.hstack((betaR, exitR, AR, self.minpaths))
        np.savetxt(savefile, savearray)
