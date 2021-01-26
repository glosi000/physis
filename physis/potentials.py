#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:56:27 2021

List of potential energies written as functions.

@author: glosi000
"""

__author__ = "Gabriele Losi"
__copyright__ = "Copyright 2021, Gabriele Losi"
__license__ = "BSD 4-Clause License"
__version__ = "0.0.0"
__maintainer__ = "Gabriele Losi"
__email__ = "gabriele.losi@outlook.com"
__status__ = "Prototype"
__date__ = 'January 25th, 2021'

import math as m


def p2(cs):
    return 1.5*cs*cs - 0.5

def lennard_jones(r, epsilon, sigma, index=(12, 6)):
    """
    General pair potential resembling a Lennard Jones model. Default indexes
    values are for a typical LJ potential, also called 12-6 potential.

    Parameters
    ----------
    r : float or np.ndarray
        Distance between interacting particles. It can be a float or a numpy
        arrays containing a set of particle-particle distances.
    epsilon : float
        Dispersion energy, i.e. depth of the potential well.
    sigma : float
        Distance at which the potential energy is zero.
    index : tuple, optional
        Power indexes for repulsive and attractive terms. The default is (12, 6).

    Returns
    -------
    float or np.ndarray
        Potential energies at the corresponding distances.

    """
    
    sig_r = sigma/r
    return 4*epsilon*(m.pow(sig_r, index[0]) - m.pow(sig_r, index[1]))    
    