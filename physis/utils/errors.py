#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:18:07 2021

Custom errors for the physis package.

@author: glosi000
"""

__author__ = "Gabriele Losi"
__copyright__ = "Copyright 2021, Gabriele Losi"
__license__ = "BSD 4-Clause License"
__version__ = "0.0.0"
__maintainer__ = "Gabriele Losi"
__email__ = "gabriele.losi@outlook.com"
__status__ = "Prototype"
__date__ = 'January 4th, 2021'


import numpy as np


class LatticeError(Exception):
    """ General class for errors mathematical lattices.
    """
    pass


class InputLatticeError(LatticeError):
    """ Handle input errors in defining a mathematical lattice.
    """

    @staticmethod
    def cubic(alat, replica):
        # Check input parameters
        if not isinstance(replica, (list, tuple, np.ndarray)):
            raise InputLatticeError('Wrong type: replica')
        if any(i <= 0 for i in replica):
            raise InputLatticeError('Invalid Argument: replica')
        if not isinstance(alat, (int, float)):
            raise InputLatticeError('Wrong type: alat')
