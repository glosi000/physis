#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:57:41 2021

Basic tools for solid state physics.

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


class CrystalError(Exception): 
    """ General class for errors in crystalline structures
    """
    pass

class LatticeError(CrystalError):
    """ General class for errors in crystal lattices
    """
    pass

class InputLatticeError(LatticeError):
    
    @staticmethod
    def cubic(alat, replica):
        
        # Check input parameters
        if not isinstance(replica, (list, tuple, np.ndarray)):
            raise InputLatticeError('Argument of the wrong type: replica')
        elif not isinstance(alat, (int, float)):
            raise InputLatticeError('Argument of the wrong type: alat')
        elif any (i <= 0 for i in replica):
            raise InputLatticeError('Invalid Argument: replica cannot be <= 0')            

class Lattice:
    """ Generate a crystalline lattice of points.
    TODO : Add a crystal class that is capable to insert a basis
    """
    
    sc_sites  = np.array([0, 0, 0])
    bcc_sites = np.array([[0, 0, 0],
                          [0.5, 0.5, 0.5]])
    fcc_sites = np.array([[0, 0, 0],
                          [0.5, 0.5, 0],
                          [0.5, 0, 0.5],
                          [0, 0.5, 0.5]])
    
    def __init__(self):
        pass
    
    @staticmethod
    def generate_cubic_cell(alat, sites, replica=(1,1,1), perturbation=False, 
                            delta=1e-8):
                
        # Check inputs and get basic information
        InputLatticeError.cubic(alat, replica)

        # Build the lattice
        lattice_sites = alat * Lattice.build_sites(sites, replica)
        
        if perturbation:
            N = np.prod(replica, dtype=int) * sites.shape[0]
            lattice_sites += np.random.normal(0., delta, size=(N, 3))
        
        return lattice_sites

    @staticmethod
    def generate_sc(alat, replica=(1,1,1), perturbation=False, delta=1e-8):
        
        lattice_sites = Lattice.generate_cubic_cell(alat, Lattice.sc_sites, 
                                                    replica, perturbation, delta)
        
        return lattice_sites

    @staticmethod
    def generate_bcc(alat, replica=(1,1,1), perturbation=False, delta=1e-8):
        
        lattice_sites = Lattice.generate_cubic_cell(alat, Lattice.bcc_sites, 
                                                    replica, perturbation, delta)
        
        return lattice_sites

    @staticmethod
    def generate_fcc(alat, replica=(1,1,1), perturbation=False, delta=1e-8):
        
        lattice_sites = Lattice.generate_cubic_cell(alat, Lattice.fcc_sites, 
                                                    replica, perturbation, delta)

        return lattice_sites
    
    @staticmethod
    def generate_diamond(alat, replica=(1,1,1), basis=None, perturbation=False,
                     delta=1e-8):
        
        basis = alat * np.array([0, 0 ,0],
                                [0.25, 0.25, 0.25])
        lattice_sites =  Lattice.generate_cubic_cell(alat, basis, replica, 
                                                     perturbation, delta)
        
        return lattice_sites

    @staticmethod
    def build_sites(sites, replica=(1, 1, 1)):
        
        # Calculate the needed number of points and prepare output array
        N = np.prod(replica, dtype=int) * sites.shape[0]
        lattice_sites = np.zeros(shape=(N, 3))
        
        # Loop and replicate the original sites
        j = 0
        for n in range(replica[0]):
            for m in range(replica[1]):
                for l in range(replica[2]):
                    # Calculate the repetition units for the lattice sites
                    lattice_sites[j:j+4,:] = sites + np.array([n, m, l])
                    j += 4
            
        return lattice_sites
