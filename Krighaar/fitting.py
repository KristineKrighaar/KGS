"""
My personal funtions libary 

Kristine M. L. Krighaar
"""
import numpy as np
import scipy as sc
from iminuit import Minuit
from scipy.special import voigt_profile

"""
Fitting functions
"""

def gauss(x, A, mu, sigma: np.ndarray) -> np.ndarray:
    return  A*np.exp(-(x-mu)** 2 / (2*sigma**2))

def gaussN(x, *params):
    """
    Sum of Gaussian curves
    
    Parameters:
    - x: array, independent variable
    - params: dict, function of parameter *should have order (A1, mu1, sigma1, ..., AN, muN, sigmaN)*
    
    returns: function values, np.ndarray 
    """
    num_gaussians = len(params) // 3
    
    gaussians = [gauss(x,A = params[i], mu = params[i+1], sigma = params[i+2]) for i in range(0,len(params),3)]
    return np.sum(gaussians, axis=0)

def fit(x: np.ndarray, y: np.ndarray, y_err: np.ndarray, model_func, initial_guess, fixed_params=None, limits=None):
    """
    Chi2 fit data to a user-defined model function using iminuit with support for fixed and limited parameters.

    NOTE: If the parameter names as specified in the function you can write in any order and it will sort.
          *If the parameter names are different make sure the order is right compared to function!*

    Parameters:
    - x: array, independent variable
    - y: array, dependent variable
    - y_err: array, uncertainty in y
    - model_func: function, the model to fit
    - initial_guess: dict, initial guesses for the model parameters (e.g., {'param_name': value})
    - fixed_params: dict, parameters to fix during fitting (e.g., {'param_name': value})
    - limits: dict, parameter limits during fitting (e.g., {'param_name': (min_value, max_value)})

    Returns: MinuitResult with (.values ; .errors ; chi2 = .fval)
    """

    # Create Minuit fitting object
    
    mObject = Chi2Regression(model_func, x[y>0], y[y>0], y_err[y>0]) # Chi**2 fit to the histograms excluding the empty bins.
    
    minuit = Minuit(mObject, **initial_guess, name=initial_guess.keys())

    # In the names are the same as the functions parameters is assings the values to proper names.
    # However if names do not corresponds it will overwrite with the new names given in parameter.
    try: 
        minuit = Minuit(mObject, **initial_guess)
    except RuntimeError as e:
        print('Given names not corresponding to function, overwriting...')
        minuit = Minuit(mObject, **initial_guess, name=initial_guess.keys())

    # Set fixed parameters
    if fixed_params:
        minuit.fixed = fixed_params

    # Set parameter limits
    if limits:
        for param_name, limit_values in limits.items():
            minuit.limits[param_name] = limit_values

    # Perform the fit
    minuit.migrad()

    return minuit






def lorentz(x, A, tau, w0: np.ndarray) -> np.ndarray:
    pi = np.pi
    return (A*tau)/(pi*(x-w0)**2+tau**2)

def gauss_lorentz(x, a1, sigma1, a2, gamma2,x0,b):
    return gauss(x, a1, x0, sigma1) + lorentz(x, a2, gamma2,x0) + b

def voigt(x, A, x0, sigma, gamma, b):
    return A*voigt_profile(x-x0, sigma, gamma)+b

def lin(x, a, b):
    return a*x+b

def T_c_func(T, I, T_c, beta, C):
    """
    I(T) = I*(np.abs(T-T_c))**(2*beta)+C
    """
    return I*(np.abs(T-T_c))**(2*beta)+C




"""
Data formatting funtions?
"""

def rebin_xy_close_values(x_values, y_values, y_errors, x_threshold=0.2):
    """
    Rebin x and y arrays by averaging y-values for x-values closer than x_threshold.

    Parameters:
    - x_values: NumPy array or list of x-axis values
    - y_values: NumPy array or list of y-axis values
    - x_threshold: Threshold for x-values to trigger rebinning (default is 0.2)

    Returns:
    - A tuple containing two arrays: rebinned_x and rebinned_y
    """
    rebinned_x = []
    rebinned_y = []
    rebinned_y_errors = []

    i = 0
    while i < len(x_values):
        current_x = x_values[i]
        close_indices = np.where(np.abs(x_values - current_x) < x_threshold)[0]

        rebinned_x.append(np.mean(x_values[close_indices]))
        
        #rebinned_y.append(np.average(y_values[close_indices], weights=y_errors[close_indices]))
        
        if len(close_indices) > 1:
            rebinned_y.append(np.average(y_values[close_indices], weights=y_errors[close_indices]))
        else:
            rebinned_y.append(y_values[i])


        if len(close_indices) > 1:
            rebinned_y_errors.append(np.mean(y_errors[close_indices])/ np.sqrt(len(close_indices))) #/ np.sqrt(len(close_indices)))
        else:
            rebinned_y_errors.append(y_errors[i])

        i = close_indices[-1] + 1

    return np.array(rebinned_x), np.array(rebinned_y), np.array(rebinned_y_errors)


"""
Code from AppStas
"""

from iminuit.util import make_func_code
from iminuit import describe #, Minuit,

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])


class Chi2Regression:  # override the class with a better one
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2