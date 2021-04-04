#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:19:37 2018

Classes and functions to work with crystalline cells and atomic positions.
The Crystal class is based on a very old personal script I wrote years ago.

@author: glosi000
"""

__author__ = "Gabriele Losi"
__copyright__ = "Copyright 2021, Gabriele Losi"
__license__ = "BSD 4-Clause License"
__version__ = "0.0.0"
__maintainer__ = "Gabriele Losi"
__email__ = "gabriele.losi@outlook.com"
__status__ = "Prototype"
__date__ = 'January 26th, 2021'


#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from copy import copy 
import numpy as np
from physis.solidstate.lattice import Lattice
import matplotlib.pyplot as plt

colors={'P' : [[1,1,0]] , 'C' : [[0.7,0.7,0.7]]}


class crystal:
    """ 
        With this class you can create a generic crystal.
        Input:
        - vectors: Give the lattice cells vectors. Insert a 3x3 list.
        - basis:   Give the atomic coordinates. Insert a nx4 list.\n
        Look at children classes for specific pre-loaded crystals.
    """
        
    def __init__(self, vectors, coords, basis, xyz =''):
        
        vectors = np.array(vectors , dtype=np.float64 )
        coords  = np.array(coords  , dtype=np.float64 )
        
        if (self.__check_input(vectors, coords, basis)) == True : return
        
        basis   = np.array(basis   , dtype=np.str     )
        
        if xyz == '':
            self.cell_vecs      = vectors.copy()
            self.atoms_coords   = coords.copy()
            self.atoms_name     = np.array(basis   , dtype=np.str).copy()
            self.cell_angs      = np.zeros(3)
            self.atoms_type     = len(set(basis))
            
            self.calculate_params()
            self.__backup()
         
    def replica(self, n=1, m=1, l=1):
        """ Replicate your cell n, m, l times along x, y, z 
        """
        
        if n==0: n=1
        if m==0: m=1
        if l==0: l=1
        
        self.restore()
        
        atoms      = []
        atoms_name = []
        vecs       = self.cell_vecs[...].copy()
        
        self.cell_a *= n
        self.cell_b *= m
        self.cell_c *= l
        self.cell_vecs[0,:] *= n
        self.cell_vecs[1,:] *= m
        self.cell_vecs[2,:] *= l
        self.calculate_params()

        for nat in range(len(self.atoms_name)):
            for i in range(n):
                for j in range(m):
                    for k in range(l):
                        add=vecs[0]*i + vecs[1]*j + vecs[2]*k
                        atoms.append(list(self.atoms_coords[nat,:]+add))
                        atoms_name.append(self.atoms_name[nat])      
        self.atoms_coords = np.array(atoms.copy()     , dtype=np.float64)
        self.atoms_name   = np.array(atoms_name.copy(), dtype=np.str    )
        
        
        
    def rotate(self, axis, theta=0., type_angle='degree'):
        """ Rotate the crystal around axis of angle theta.\n
            Default: you need to give theta in radiant.\n
            You can specify theta in degree inserting type_angle='radiant'
        """
        if type_angle == 'degree':
            theta = self.__convert_degree_rad(0, theta)
        if axis == 'x':
            R = np.array([[1, 0, 0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
        elif axis == 'y':
            R = np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]])
        elif axis == 'z':
            R = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
        else:
            print("Error. Give 'x', 'y' or 'z' as axis")
            return
        self.cell_vecs[0, :] = np.dot(R, self.cell_vecs[0, :])
        self.cell_vecs[1, :] = np.dot(R, self.cell_vecs[1, :])
        self.cell_vecs[2, :] = np.dot(R, self.cell_vecs[2, :])
        for i in range(int(np.shape(self.atoms_coords.shape)[0])):
            self.atoms_coords[i, :] = np.dot(R, self.atoms_coords[i, :])
        for i in range(int(np.shape(self.atoms_coords.shape)[0])):
            for j in range(3):
                if self.atoms_coords[i, j] < 1e-15:
                    self.atoms_coords[i, j] = 0.
        #self.__calculate_angles()

    
    def calculate_params(self):
        self.cell_a = np.sqrt(sum(self.cell_vecs[0,:]**2.))
        self.cell_b = np.sqrt(sum(self.cell_vecs[1,:]**2.))
        self.cell_c = np.sqrt(sum(self.cell_vecs[2,:]**2.))
        
        self.__calculate_angles()
            
        self.cell_angs[:] = [self.cell_alpha, self.cell_beta, self.cell_gamma]
            
        self.cell_vol = self.cell_a*self.cell_b*self.cell_c
        self.cell_vol *= np.sqrt(1 + 2*np.cos(self.cell_angs[0])*np.cos(self.cell_angs[1])*np.cos(self.cell_angs[2]) -\
                               np.cos(self.cell_angs[0])**2. - np.cos(self.cell_angs[0])**2. - np.cos(self.cell_angs[0])**2.)
        
        self.__check_periodicity()
                
            
    def __check_periodicity(self):
        
        x = self.cell_vecs[:,0].copy()
        y = self.cell_vecs[:,2].copy()
        z = self.cell_vecs[:,1].copy()
        at= self.atoms_coords.copy()
        
        for i in range(len(self.atoms_name)):
            if at[i,0] < 0:         at[i,0] += x[i]
            elif at[i,0] >= x[i] :  at[i,0] -= x[i]
            if at[i,1] < 0:         at[i,1] += y[i]
            elif at[i,1] >= x[i] :  at[i,1] -= y[i]
            if at[i,2] < 0:         at[i,2] += z[i]
            elif at[i,2] >= x[i] :  at[i,2] -= z[i]

    
    def __calculate_angles(self):
        
        PI2 = 2*np.pi
        self.cell_alpha  = np.arccos(np.dot(self.cell_vecs[1,:], self.cell_vecs[2,:]) / ( self.cell_b*self.cell_c ))
        self.cell_beta  = np.arccos(np.dot(self.cell_vecs[0,:], self.cell_vecs[2,:]) / ( self.cell_a*self.cell_c ))
        self.cell_gamma = np.arccos(np.dot(self.cell_vecs[0,:], self.cell_vecs[1,:]) / ( self.cell_a*self.cell_b ))
        
        if self.cell_alpha < 0:
            self.cell_alpha += PI2 
        elif self.cell_alpha >= PI2:
            self.cell_alpha -= PI2
        if self.cell_beta < 0:
            self.cell_beta += PI2 
        elif self.cell_beta >= PI2:
            self.cell_beta -= PI2
        if self.cell_gamma < 0:
            self.cell_gamma += PI2 
        elif self.cell_gamma >= PI2:
            self.cell_gamma -= PI2
    
    
    def __convert_degree_rad(self, conv, angle):
        if conv == 0: #Convert from degree to radiant
            return angle*np.pi/180.
        else: #Convert from radiant to degree
            return angle*180/np.pi

    def __backup(self):
           
        self.cell_vecs_BACK      = self.cell_vecs.copy()
        self.atoms_name_BACK     = self.atoms_name.copy()
        self.atoms_coords_BACK   = self.atoms_coords.copy()
        self.cell_angs_BACK      = self.cell_angs.copy()
        self.cell_alpha_BACK     = self.cell_alpha.copy()
        self.cell_beta_BACK      = self.cell_beta.copy()
        self.cell_gamma_BACK     = self.cell_gamma.copy()
        self.cell_a_BACK         = self.cell_a.copy()
        self.cell_b_BACK         = self.cell_b.copy()
        self.cell_c_BACK         = self.cell_c.copy()
        self.atoms_type_BACK     = copy(self.atoms_type)
        self.atoms_type_BACK = 3
    
    def restore(self):
        from copy import copy
        self.cell_vecs      = self.cell_vecs_BACK.copy()
        self.atoms_name     = self.atoms_name_BACK.copy()
        self.atoms_coords   = self.atoms_coords_BACK.copy()
        self.cell_angs      = self.cell_angs_BACK.copy()
        self.cell_alpha     = self.cell_alpha_BACK.copy()
        self.cell_beta      = self.cell_beta_BACK.copy()
        self.cell_gamma     = self.cell_gamma_BACK.copy()
        self.cell_a         = self.cell_a_BACK.copy()
        self.cell_b         = self.cell_b_BACK.copy()
        self.cell_c         = self.cell_c_BACK.copy()
        self.atoms_type     = copy(self.atoms_type_BACK)
    
    
    def info(self, mod=0):
        """ 
        Get a bunch of information about your crystal 
        
        Set mod to 1 or 2 if you want to plot the lattice
        
        """
        print("\nInformation about your crystal:\n")
        print("Lattice vectors (Å):")
        print("1 - magnitude: %.3f - vector: %.3f %.3f %.3f"%(self.cell_a, self.cell_vecs[0][0], self.cell_vecs[0][1], self.cell_vecs[0][2]))
        print("2 - magnitude: %.3f - vector: %.3f %.3f %.3f"%(self.cell_b, self.cell_vecs[1][0], self.cell_vecs[1][1], self.cell_vecs[1][2]))
        print("3 - magnitude: %.3f - vector: %.3f %.3f %.3f"%(self.cell_c, self.cell_vecs[2][0], self.cell_vecs[2][1], self.cell_vecs[2][2]))
        print("Cell volume: %.3f" %(self.cell_vol))
        print("")
        print("Lattice angles (degree):")
        print("alpha: %.3f"%(self.__convert_degree_rad(1, self.cell_alpha)))
        print("beta : %.3f"%(self.__convert_degree_rad(1, self.cell_beta)))
        print("gamma: %.3f"%(self.__convert_degree_rad(1, self.cell_gamma)))
        print("")
        self.print_coordinates(mod)
        self.plot_crystal()
        
                    
    def print_coordinates(self, mod=0):
        if mod==0:
            print("Atoms coordinates (Å):")
            for i in range(len(self.atoms_name)):
                print("%d - %s  %.3f %.3f %.3f"%(i+1, self.atoms_name[i], self.atoms_coords[i][0], self.atoms_coords[i][1], self.atoms_coords[i][2]))
        elif mod==1:
            print("Atoms coordinates (crystal):")
            for i in range(len(self.atoms_name)):
                print("%d - %s  %.3f %.3f %.3f"%(i+1, self.atoms_name[i], self.atoms_coords[i][0]/self.cell_a, self.atoms_coords[i][1]/self.cell_b, self.atoms_coords[i][2]/self.cell_c))
        elif mod==2:
            print("Atoms coordinates (alat):")
            atoms = self.atoms_coords.copy() / self.cell_a
            for i in range(len(self.atoms_name)):
                print("%d - %s  %.3f %.3f %.3f"%(i+1, self.atoms_name[i], atoms[i][0], atoms[i][1], atoms[i][2]))
 
    
    def __check_input(self, vectors, coords, basis):
        
        if vectors.ndim != 2 or vectors.shape != (3,3) :
            self.__error_message(0)
            return True
        
        elif len(coords.shape) == 1:
                if coords.shape[0] != 3:
                    self.__error_message(1)
                    return True
                elif coords.shape[1] != 3:
                    self.__error_message(1)
                    return True
                
        
        elif len(basis) != coords.shape[0]:
            self.__error_message(2)
            return True

        for i in basis:
            if type(i) != str:
                self.__error_message(3)
                return True
        
        return False
    
    
    def __error_message(self, i):
        if i == 0:
            print("\nProblem with the input: "+
                  "vectors must be a 3x3 matrix.\n")
        elif i == 1:
            print("\nProblem with the input: "+
                  "coords must be a nx3 matrix.\n")
        elif i == 2:
            print("\nProblem with the input: "+
                  "len(basis) must be the same of len(coords[:,0]).\n")
        elif i == 3:
            print("\nProblem with the input: "+
                  "basis must be a list of characters/string.\n")
    
    
    def plane(self, h, k, l):
        
        self.restore()
        imax = max(h,k,l)*15
        
        PI2 = 2*np.pi
        a1 = self.cell_vecs[0].copy()
        a2 = self.cell_vecs[1].copy()
        a3 = self.cell_vecs[2].copy()

        if np.count_nonzero(self.cell_vecs - np.diag(np.diagonal(self.cell_vecs))) == 0:
            #if (h,k,l) == ( (1,0,0) or (0,1,0) (0,0,1) or (1,1,0) or (0,1,1)):
            if (h,k,l) == (1,1,1):
                p1 = self.cell_vecs[0]
                p2 = self.cell_vecs[1]
                p3 = self.cell_vecs[2]
        elif h != 0 and k != 0 and l != 0:
            p1 = h/a1
            p2 = k/a2
            p3 = l/a3

        xmin = min(p1[0],p2[0],p3[0])
        ymin = min(p1[1],p2[1],p3[1])
        zmin = min(p1[2],p2[2],p3[2])
        p1 -= [xmin, ymin, zmin]
        p2 -= [xmin, ymin, zmin]
        p3 -= [xmin, ymin, zmin]
        
        p1p2 = p2-p1
        p1p3 = p3-p2
        
        vec_perp=np.array([h,k,l], dtype=np.float64)
        d = - (np.inner(vec_perp,p1))
        # Plane equation: hx+ky+lz+d=0

        
        #b1 = PI2*(np.cross(a2,a3))/(np.inner(a1,np.cross(a2,a3)))
        #b2 = PI2*(np.cross(a3,a1))/(np.inner(a2,np.cross(a3,a1)))
        #b3 = PI2*(np.cross(a1,a2))/(np.inner(a3,np.cross(a1,a2)))

        self.replica(imax,imax,imax)
        
        Nat = len(self.atoms_name)
        atoms_coords= self.atoms_coords.copy()
        atoms_name= self.atoms_name.copy()
        belong_plane = [False]*Nat
        
        plane = []
        
        i=0
        vec_min = 100
        
        #Find the atoms on the plane family <hkl>
        for at in self.atoms_coords:
            vec_dist_on_plane=at-p1
            if np.inner(vec_dist_on_plane, vec_perp) < 1e-17:
                belong_plane[i] = True
            i += 1
        
        dist_from_plane = np.abs(h*self.atoms_coords[:,0]+k*self.atoms_coords[:,1]+l*self.atoms_coords[:,2]+d) / (h**2.+k**2.+l**2.)
        smaller = min(dist_from_plane)
        i=0
        
        #Select the atoms on the plane (hkl)
        for dist in dist_from_plane:
            if dist - smaller > 1e-10: 
                belong_plane[i]=False
            i += 1
        atoms_coords = atoms_coords[belong_plane]
        atoms_name   = atoms_name[belong_plane]
        
        #atoms_coords, atoms_name, belong_basis = self.__plane_find_basis(atoms_coords, atoms_name)
        
        self.plot_plane(h,k,l,[p1,p2,p3], atoms_coords, atoms_name)        
        self.restore()
    
    
    def __plane_find_basis(self, atoms_coords, atoms_name):
        at=atoms_coords
        belong_basis = [True]*len(atoms_name)
        n = len(belong_basis)
        orig = at[0,:]
        dist = np.zeros((n-1))
        
        for i in range(n-1):
            d = orig - at[i+1,:]
            dist[i] = np.inner(d,d)
        for i in range(n-1):
            for j in range(i+1, n-1):
                r = (dist[j] - orig)
                r = np.inner(r, r)/dist[j]
                if int(r) - r < 1e-15:
                    belong_basis[i] = False
        return at[belong_basis], atoms_name[belong_basis], belong_basis
        
    def plot_crystal(self):        
        #print(len(self.atoms_name))
        self.__replica_plot()
        #print(len(self.atoms_name))
        
        a=self.cell_vecs[0,:].copy()
        b=self.cell_vecs[1,:].copy()
        c=self.cell_vecs[2,:].copy()
        
        atx=self.atoms_coords[:,0]
        aty=self.atoms_coords[:,1]
        atz=self.atoms_coords[:,2]
        
        #    0    1     2     3       4          5          6             7
        x = [0, a[0], b[0], c[0], a[0]+b[0], c[0]+b[0], a[0]+c[0], a[0]+b[0]+c[0]]
        y = [0, a[1], b[1], c[1], a[1]+b[1], c[1]+b[1], a[1]+c[1], a[1]+b[1]+c[1]]
        z = [0, a[2], b[2], c[2], a[2]+b[2], c[2]+b[2], a[2]+c[2], a[2]+b[2]+c[2]]

        v=np.zeros((8,3))
        for i in range(8):
            v[i]=np.array([x[i],y[i],z[i]])
        
        #verts3D = [[v[0],v[1],v[4],v[2]] , [v[3],v[6],v[7],v[5]] , 
                   #[v[0],v[1],v[6],v[3]] , [v[0],v[2],v[5],v[3]] , 
                   #[v[1],v[6],v[7],v[4]] , [v[2],v[5],v[7],v[4]]]
        
        verts   = [[v[0],v[1],v[4],v[2],v[0]] , [v[3],v[6],v[7],v[5],v[3]] , 
                   [v[0],v[1],v[6],v[3],v[0]] , [v[0],v[2],v[5],v[3],v[0]] , 
                   [v[1],v[6],v[7],v[4],v[1]] , [v[2],v[5],v[7],v[4],v[4]]]
        
        verts=np.array(verts)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')        
        #Make the faces of the cell colored and transparent
        #poly = Poly3DCollection(verts3D, linewidths=1, facecolor='cyan', edgecolor='k', alpha=0.25)
        #alpha=0.05
        #poly.set_facecolor((1, 1, 0, alpha))
        #ax.add_collection3d(poly)
        
        #Make the lines to close the cell
        for i in range(6):
            ax.plot(verts[i][:,0],verts[i][:,1],verts[i][:,2], color='k')
        
        #Finally put the corners to the cell and place the atoms
        ax.scatter(x,y,z, c='k', marker='')
        
        groups=np.zeros(len(self.atoms_name))
        k=0      
        for i in self.atoms_name:
            where = np.where(self.atoms_name == i)
            groups[where]= k
            k += 1
        
        for g in np.unique(groups):
            ix = np.where(groups == g)
            lab = self.atoms_name[int(g)]
            ax.scatter3D(atx[ix], aty[ix], atz[ix], label=lab, s = 100)
        
#        for name in self.atoms_name:
#            ax.scatter3D(atx,aty,atz, s=100, label=name)
        ax.legend()
        plt.show()
        self.restore()
        
        
        
    def plot_plane(self, h, k, l, p, atoms_coords, atoms_name):        
    
        atx=atoms_coords[:,0].copy()
        aty=atoms_coords[:,1].copy()
        atz=atoms_coords[:,2].copy()
        print(atoms_coords, atoms_name)
        #atoms_name = self.atoms_name[belong_plane]
        
        self.restore()
        
        a=self.cell_vecs[0,:].copy()
        b=self.cell_vecs[1,:].copy()
        c=self.cell_vecs[2,:].copy()
        
        
        #    0    1     2     3       4          5          6             7
        x = [0, a[0], b[0], c[0], a[0]+b[0], c[0]+b[0], a[0]+c[0], a[0]+b[0]+c[0]]
        y = [0, a[1], b[1], c[1], a[1]+b[1], c[1]+b[1], a[1]+c[1], a[1]+b[1]+c[1]]
        z = [0, a[2], b[2], c[2], a[2]+b[2], c[2]+b[2], a[2]+c[2], a[2]+b[2]+c[2]]

        v=np.zeros((8,3))
        for i in range(8):
            v[i]=np.array([x[i],y[i],z[i]])
        
        verts   = [[v[0],v[1],v[4],v[2],v[0]] , [v[3],v[6],v[7],v[5],v[3]] , 
                   [v[0],v[1],v[6],v[3],v[0]] , [v[0],v[2],v[5],v[3],v[0]] , 
                   [v[1],v[6],v[7],v[4],v[1]] , [v[2],v[5],v[7],v[4],v[4]]]
        
        verts=np.array(verts)

        p1=list(p[0])
        p2=list(p[1])
        p3=list(p[2])
      
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(6):
            ax.plot(verts[i][:,0],verts[i][:,1],verts[i][:,2], color='k')
        
        verts = [p1,p2,p3,p1]
        verts=np.array(verts)
        ax.plot3D(verts[:,0],verts[:,1],verts[:,2], color='k')
        ax.scatter3D(atx, aty, atz, s = 100)
        
        groups=np.zeros(len(atoms_name))
        k=0      
        for i in atoms_name:
            where = np.where(atoms_name == i)
            groups[where]= k
            k += 1
        
        for g in np.unique(groups):
            ix = np.where(groups == g)
            lab = atoms_name[int(g)]
            ax.scatter3D(atx[ix], aty[ix], atz[ix], label=lab, s = 100)
        
        ax.legend()
        plt.show()
        self.restore()


    def __replica_plot(self):
        pass
#        x  = self.cell_vecs[0,:].copy()
#        y  = self.cell_vecs[1,:].copy()
#        z  = self.cell_vecs[2,:].copy()
#        at = self.atoms_coords.copy()
#        N = len(self.atoms_name)
#        
#        for i in range(N):
#            rep     = at[i,:]
#            name_at = self.atoms_name[i]
#            
#            if sum(rep) < 1e-17: #on origin
#                at=np.vstack([at, rep+x])
#                at=np.vstack([at, rep+y])
#                at=np.vstack([at, rep+z])
#                at=np.vstack([at, rep+x+y])
#                at=np.vstack([at, rep+x+z])
#                at=np.vstack([at, rep+y+z])
#                at=np.vstack([at, rep+x+y+z])
#                self.atoms_name=np.append(self.atoms_name, [name_at]*7)
#                
#            if sum(rep) < 1e-17: #on origin
#                at=np.vstack([at, rep+x])
#                at=np.vstack([at, rep+y])
#                at=np.vstack([at, rep+z])
#                at=np.vstack([at, rep+x+y])
#                at=np.vstack([at, rep+x+z])
#                at=np.vstack([at, rep+y+z])
#                at=np.vstack([at, rep+x+y+z])
#                self.atoms_name=np.append(self.atoms_name, [name_at]*7)
#                
#                
#                
#                
#            
#            elif rep[0] < 1e-17 and rep[1] < 1e-17: #on z axis
#                at=np.vstack([at, rep+x])
#                at=np.vstack([at, rep+y])
#                at=np.vstack([at, rep+x+y])
#                self.atoms_name=np.append(self.atoms_name, [name_at]*3)
#                    
#            elif rep[0] < 1e-17 and rep[2] < 1e-17: #on y axis
#                at=np.vstack([at, rep+x])
#                at=np.vstack([at, rep+z])
#                at=np.vstack([at, rep+x+z])
#                self.atoms_name=np.append(self.atoms_name, [name_at]*3)
#                
#            elif rep[1] < 1e-17 and rep[2] < 1e-17: #on x axis
#                at=np.vstack([at, rep+y])
#                at=np.vstack([at, rep+z])
#                at=np.vstack([at, rep+y+z])
#                self.atoms_name=np.append(self.atoms_name, [name_at]*3)
#            
#            elif sum(rep - x) < 1e-17:
#                at=np.vstack([at, rep-x])
#                at=np.vstack([at, rep-y])
#                at=np.vstack([at, rep-x-y])
#                self.atoms_name=np.append(self.atoms_name, [name_at]*3)
#            
#            elif sum(rep - y) < 1e-17: #on y axis
#                at=np.vstack([at, rep-x])
#                at=np.vstack([at, rep-z])
#                at=np.vstack([at, rep-x-z])
#                self.atoms_name=np.append(self.atoms_name, [name_at]*3)
#                
#            elif sum(rep - z) < 1e-17: #on y axis
#                at=np.vstack([at, rep-y])
#                at=np.vstack([at, rep-z])
#                at=np.vstack([at, rep-y-z])
#                self.atoms_name=np.append(self.atoms_name, [name_at]*3)
#            
#            #Now replicate for the upper atoms
#            elif sum(rep - x - y - z) < 1e-17: #on origin
#                at=np.vstack([at, rep-x])
#                at=np.vstack([at, rep-y])
#                at=np.vstack([at, rep-z])
#                at=np.vstack([at, rep-x-y])
#                at=np.vstack([at, rep-x+z])
#                at=np.vstack([at, rep-y-z])
#                at=np.vstack([at, rep-x-y-z])
#                self.atoms_name=np.append(self.atoms_name, [name_at]*7)
#                
#            else: #only one coordinate is zero
#                if rep[0] < 1e-17: #(0,y,z)
#                    at=np.vstack([at, rep+x])
#                    self.atoms_name=np.append(self.atoms_name, name_at)
#                elif rep[1] < 1e-17: #(x,0,z)
#                    at=np.vstack([at, rep+y])
#                    self.atoms_name=np.append(self.atoms_name, name_at)
#                elif rep[2] < 1e-17: #(x,y,0)
#                    at=np.vstack([at, rep+z])
#                    self.atoms_name=np.append(self.atoms_name, name_at)
#                elif sum(rep - x) < 1e-17: #(0,y,z)
#                    at=np.vstack([at, rep-x])
#                    self.atoms_name=np.append(self.atoms_name, name_at)
#                elif sum(rep - y) < 1e-17: #(x,0,z)
#                    at=np.vstack([at, rep-y])
#                    self.atoms_name=np.append(self.atoms_name, name_at)
#                elif sum(rep - z) < 1e-17: #(x,y,0)
#                    at=np.vstack([at, rep-z])
#                    self.atoms_name=np.append(self.atoms_name, name_at)
#        
#        self.atoms_coords = at.copy()
        
if __name__ == '__main__':

    a=crystal( [[1,0,0], [0,1,0], [0,0,1]], [[0, 0, 0], [0.5, 0.5, 0]] , ['P', 'C']  )
    a.info()
    a.plane(1,1,1)

