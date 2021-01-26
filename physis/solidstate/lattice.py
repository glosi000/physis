#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:57:41 2021

Classes and functions to work with lattices.

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

import numpy as np


class LatticeError(Exception):
    """ General class for errors in crystal lattices
    """
    pass


class InputLatticeError(LatticeError):
    """ Manage input errors in building lattices
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


class Cell:
    """ Simple class to store the information about the unit cell
    """
    
    def __init__(self, cell):
        self.cell = cell   
    
    @classmethod
    def create_cube(cls, alat):
        return cls(cell = np.diag(np.full(3, alat)))
    

class Lattice(Cell):
    """ Generate a crystalline lattice of points.
    """

    # Conventional lattice sites
    sc_sites  = np.array([0, 0, 0])
    bcc_sites = np.array([[0, 0, 0],
                          [0.5, 0.5, 0.5]])
    fcc_sites = np.array([[0, 0, 0],
                          [0.5, 0.5, 0],
                          [0.5, 0, 0.5],
                          [0, 0.5, 0.5]])
    
    def __init__(self, cell, lattice):
        self.cell = cell
        self.lattice = lattice

    @property
    def latx(self):
        return self.lattice[:, 0]
    
    @property
    def laty(self):
        return self.lattice[:, 1]
    
    @property
    def latz(self):
        return self.lattice[:, 2]

    
    def generate_cubic_lattice(alat, sites, replica=(1,1,1), 
                               perturbation=False, delta=1e-8):

        # Check inputs and get basic information
        InputLatticeError.cubic(alat, replica)

        # Build the lattice
        lat_sites = alat * Lattice.build_sites(sites, replica)

        # Insert perturbation
        if perturbation:
            N = np.prod(replica, dtype=int) * sites.shape[0]
            lat_sites += np.random.normal(0., delta, size=(N, 3))
        
        return lat_sites
    
    def build_sites(sites, replica=(1, 1, 1)):

        # Calculate the needed number of points and prepare output array
        N = np.prod(np.array(replica), dtype=int) * sites.shape[0]
        lat_sites = np.zeros(shape=(N, 3))

        # Loop and replicate the original sites
        j = 0
        for n in range(replica[0]):
            for m in range(replica[1]):
                for l in range(replica[2]):

                    # Calculate the repetition units for the lattice sites
                    lat_sites[j:j+4,:] = sites + np.array([n, m, l])
                    j += 4
        return lat_sites

    @classmethod
    def generate_sc(cls, alat, replica=(1,1,1), perturbation=False, delta=1e-8):

        cell = alat * np.diag(np.full(3, replica))
        lat_sites = Lattice.generate_cubic_lattice(alat, Lattice.sc_sites, 
                                                   replica, perturbation, delta)
        return cls(cell, lat_sites)

    @classmethod
    def generate_bcc(cls, alat, replica=(1,1,1), perturbation=False, delta=1e-8):
        
        cell = alat * np.diag(np.full(3, replica))
        lat_sites = Lattice.generate_cubic_lattice(alat, Lattice.bcc_sites, 
                                                   replica, perturbation, delta)
        return cls(cell, lat_sites)

    @classmethod
    def generate_fcc(cls, alat, replica=(1,1,1), perturbation=False, delta=1e-8):

        cell = alat * np.diag(np.full(3, replica))
        lat_sites = Lattice.generate_cubic_lattice(alat, Lattice.fcc_sites, 
                                                   replica, perturbation, delta)
        return cls(cell, lat_sites)
