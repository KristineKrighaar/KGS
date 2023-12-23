"""
KGS = Kristine's Golden Stasndard

My personal funtions libary 

Kristine M. L. Krighaar

"""

"""
Neutron calucations functions
"""

import numpy as np
import scipy as sc


def E_to_lambda(E): 
    """
    Converts ndarray of E in [meV] to wavelength in [AA]
    """
    return 1/(0.11056*np.sqrt(E))


def bragg_angle(lamb, d, n=1):
    res = np.degrees(np.sin((n*lamb)/(2 * d))**(1e-1))
    return float(res)


def d_spacing(a, b, c, alpha, beta, gamma, hkl):
    """
    Calculate reciprocal lattice constants from crystal lattice constants.

    Parameters:
    - a, b, c: Crystal lattice constants
    - alpha, beta, gamma: Angles between lattice vectors in degrees
            -> Calculates a_star, b_star, c_star: Reciprocal lattice vectors
    - h, k, l: Miller indices for the plane

    Returns:
    - d-spacing [Å]
    """
    h, k, l = np.array([hkl[0], hkl[1], hkl[2]])

    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Step 2: Calculate lattice vectors
    a_vector = np.array([a, 0, 0])
    b_vector = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
    c_vector = np.array([c * np.cos(beta_rad),
                         c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad),
                         c * np.sqrt(1 + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad) - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2) / np.sin(gamma_rad)])

    # Step 3: Calculate volume of the parallelepiped using vector cross product
    volume = np.abs(np.dot(a_vector, np.cross(b_vector, c_vector)))


    # Step 4: Calculate reciprocal lattice vectors
    a_star = 2 * np.pi * np.cross(b_vector, c_vector) / volume
    b_star = 2 * np.pi * np.cross(c_vector, a_vector) / volume
    c_star = 2 * np.pi * np.cross(a_vector, b_vector) / volume

   

    reciprocal_vector = np.array([h, k, l])
    reciprocal_lattice_vectors = np.array([a_star, b_star, c_star])
    d_spacing = np.divide(2*np.pi, (np.linalg.norm(np.dot(reciprocal_vector, reciprocal_lattice_vectors))))

    return  d_spacing
