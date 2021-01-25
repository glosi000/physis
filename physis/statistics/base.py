#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:30:22 2021

This module contains some basic classes useful to any statistical simulations.

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


class SimulationBox:
    
    def __init__(self, nx, ny, nz, density, radius = .5, epsilon = 1.):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.a =(4/density)**(1/3.)
        self.density = density
        self.Lx = self.a*self.nx
        self.Ly = self.a*self.ny
        self.Lz = self.a*self.nz
        self.V = self.Lx*self.Ly*self.Lz
        self.N = 4*self.nx*self.ny*self.nz
        self.sigma = 2.*radius
        self.epsilon = 4*epsilon
        self.x      = np.zeros(self.N)
        self.y      = np.zeros(self.N)
        self.z      = np.zeros(self.N)
        self.energy = 0.

        rmax        = min( (self.Lx, self.Ly, self.Lz) )/2.
        if rmax/2. > 2.5*self.sigma:
            self.r2max = 2.5*self.sigma
        else:
            self.r2max = rmax * rmax