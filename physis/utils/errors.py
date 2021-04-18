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
    def cubic(alat, sites, replica, perturb):

        # Check alat
        if not isinstance(alat, (int, float)) or isinstance(alat, bool):
            raise InputLatticeError('Wrong type: alat. It should be int or float.')
        
        # Check sites
        if not isinstance(sites, (list, tuple, np.ndarray)):
            raise InputLatticeError('Wrong type: sites. It should be array-like.')        

        # Check replica
        if not isinstance(replica, (list, tuple, np.ndarray)):
            raise InputLatticeError('Wrong type: replica. It should be array-like.')
        if any(i <= 0 for i in replica):
            raise InputLatticeError('Invalid Argument: replica. Elements '
                                    'should be positive.')           

        # Check perturb
        if perturb is not None:
            if not isinstance(perturb, (int, float)):
                raise InputLatticeError('Wrong type: perturb. It should be '
                                        'int or float when not None.')

class CrystalError(Exception):
    """ General custom error for crystal generation
    """
    pass

class InputCrystalError(CrystalError):
    """ Handle input errors within the constructor method of Crystal class
    """

    @staticmethod
    def atomic_basis(atoms, basis):

        # Check basis argument and type
        if basis is not None:
            if not isinstance(basis, (list, np.ndarray, np.matrix)):
                raise InputCrystalError('Wrong type: basis. It should be list, '
                                        'np.ndarray, np.matrix or None')

        # Check atoms argument
        if atoms is not None:
            # Check atoms type
            if not isinstance(atoms, (list, np.ndarray)):
                raise InputCrystalError('Wrong type: atoms. It should be list, '
                                        'np.ndarray or None')

            # Check atoms shape compared to basis. If atoms is provided, the 
            # number of elements must be the same of the rows of basis
            lbasis = 1 if basis is None else np.array(basis).shape[0]
            if len(np.array(atoms)) != lbasis:
                raise InputCrystalError('Invalid input arguments for atoms or '
                                        'basis. If atoms is not None, it must be '
                                        'len(atoms) equal to basis.shape[0], '
                                        'or 1 if basis is None')
