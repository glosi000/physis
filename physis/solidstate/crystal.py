#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 19:57:35 2021

Class to create a proper crystal object, combining cell, sites, atoms together.
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
from physis.solidstate.lattice import Lattice


class Crystal(Lattice):
    """ Create a crystal cell made of an atomic basis occupying lattice sites.
    """
    
    def __init__(self, cell, sites, basis=None):
        
        super().__init__(cell, sites)
        basis = np.array(basis) if basis is not None else self.sites.copy()
        
        