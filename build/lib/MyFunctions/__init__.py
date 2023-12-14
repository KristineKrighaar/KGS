
############### Packages I regularly use ###############

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import pi
import pandas as pd
import math
from iminuit import Minuit
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl
import sys
import scipy as sc
from scipy.special import voigt_profile

############### Fitting tool I got from AppStat ###############

# sys.path.append('/Users/kristinekrighaar/Documents/External_Functions')
# from ExternalFunctions import Chi2Regression
# from ExternalFunctions import nice_string_output, add_text_to_ax  

############### Formatting of plots as I prefere them ###############

# mpl.rcParams['lines.linewidth'] = 0.3
mpl.rcParams['errorbar.capsize'] = 2
mpl.rcParams['lines.markersize'] = 11
mpl.rcParams['font.size']        = 15 # standard er 45
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Import necessary packages
mpl.rcParams['font.family'] = 'lmodern'  # Choose the font (e.g., Latin Modern)

from .fitting import *
from .neutron import *