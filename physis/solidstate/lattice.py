#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:57:41 2021

Classes to work with the basic definition of mathematical cells and lattices.

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
from physis.utils.errors import InputLatticeError


class Cell:
    """ Simple class to store the information about the unit cell
    """
    
    def __init__(self, cell):
        self.cell = cell   
    
    @classmethod
    def create_cube(cls, alat):
        return cls(cell = np.diag(np.full(3, alat)))
    
    @property
    def alat(self):
        return np.linalg.norm(self.cell[0, :])
    
    @property
    def blat(self):
        return np.linalg.norm(self.cell[1, :])
    
    @property
    def clat(self):
        return np.linalg.norm(self.cell[2, :])


class LatticeSites:
    """ Collection of primitive Bravais Lattices.
    """

    # Cubic lattice sites
    sc  = np.array([0, 0, 0])
    bcc = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    fcc = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


class Lattice(Cell):
    """ Generate a crystalline lattice of points.
    """
    
    def __init__(self, cell, sites=[0, 0, 0]):
        super().__init__(cell)
        self.sites = np.array(sites)
    
    def build_sites(sites, replica=(1, 1, 1)):

        # Calculate the needed number of points preparing output array
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
    def generate_sc(cls, alat, replica=(1,1,1), perturb=None):

        # Generate the cell and the lattice sites
        cell, sites = Lattice._generate_cube(alat, LatticeSites.sc, replica, perturb)
        return cls(cell, sites)

    @classmethod
    def generate_bcc(cls, alat, replica=(1, 1, 1), perturb=None):

        # Generate the cell and the lattice sites
        cell, sites = Lattice._generate_cube(alat, LatticeSites.bcc, replica, perturb)
        return cls(cell, sites)

    @classmethod
    def generate_fcc(cls, alat, replica=(1, 1, 1), perturb=None):
        
        # Generate the cell and the lattice sites
        cell, lattice = Lattice._generate_cube(alat, LatticeSites.fcc, replica, perturb)
        return cls(cell, lattice)

    def _generate_cube(alat, sites, replica=(1, 1, 1), perturb=None):

        # Search for fundamental errors in input arguments
        InputLatticeError.cubic(alat, sites, replica, perturb)
        
        # Define the crystal cell and the lattice sites
        cell = alat * np.diag(np.full(3, replica))
        sites = np.array(sites)

        # Build the lattice
        lattice = alat * Lattice.build_sites(sites, replica)

        # Insert perturbation
        if perturb is not None:
            N = np.prod(replica, dtype=int) * sites.shape[0]
            lattice += np.random.normal(0., perturb, size=(N, 3))
        
        return cell, lattice
