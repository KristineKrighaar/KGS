"""
Neutron calucations functions
"""

import numpy as np
import scipy as sc


def E_to_lambda(E: np.ndarray) -> np.ndarray: 
    """
    Converts ndarray of E in [meV] to wavelength in [AA]
    """
    return 1/(0.1156*np.sqrt(E))

