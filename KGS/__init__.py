"""
KGS = Kristine's Golden Stasndard

My personal funtions libary 

Kristine M. L. Krighaar
"""


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
import scipy as sci
from scipy.special import voigt_profile
import matplotlib.colors as colors
#import scipp as sc

############### Fitting tool I got from AppStat ###############

# sys.path.append('/Users/kristinekrighaar/Documents/External_Functions')
# from ExternalFunctions import Chi2Regression
# from ExternalFunctions import nice_string_output, add_text_to_ax  

############### Formatting of plots as I prefere them ###############
# mpl.rcParams['lines.linewidth'] = 0.3
mpl.rcParams['errorbar.capsize'] = 2
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['font.size']        = 15 # standard er 45
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.top'] = True
mpl.rcParams['xtick.direction']='in'
mpl.rcParams['ytick.direction']='in'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Import necessary packages
mpl.rcParams['font.family'] = 'lmodern'  # Choose the font (e.g., Latin Modern)
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#1b9e77','#7570b3', '#d95f02','#e7298a','#66a61e'])#['#56c456', '#a887d4', '#fdc086', '#ffff99', '#386cb0'])


from .fitting import *
from .neutron import *
