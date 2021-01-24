#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class LJ:
    
    def __init__(self, nx, ny, nz, density, radius = .5, epsilon = 1. ):
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
        
        
    def minimum_distance(self,x1, y1, z1, x2, y2, z2):
        deltax = x2-x1
        deltay = y2-y1
        deltaz = z2-z1
        min_deltax = deltax - self.Lx*np.rint(deltax/self.Lx)
        min_deltay = deltay - self.Ly*np.rint(deltay/self.Ly)
        min_deltaz = deltaz - self.Lz*np.rint(deltaz/self.Lz)
        sig_r = self.sigma**2./(min_deltax**2 +min_deltay**2 + min_deltaz**2)
        for i in range(len(sig_r)):
            if sig_r[i] > 6.25:
                sig_r[i] = 1
        return sig_r
    
    def LJ_energy(self,sig_r,mode):
        if mode == 1:
            return self.epsilon*(sig_r**12.-sig_r**6.)
        elif mode == 2:
            return self.epsilon*(sig_r**6.-sig_r**3.)
        
    def system_energy(self):
        energy = 0.
        for i in range(self.N-1):
            point_x = np.zeros(self.N-(i+1))
            point_y = np.zeros(self.N-(i+1))
            point_z = np.zeros(self.N-(i+1))
            point_x[...] = self.x[i]
            point_y[...] = self.y[i]
            point_z[...] = self.z[i]
            selected_x = self.x[i+1:]
            selected_y = self.y[i+1:]
            selected_z = self.z[i+1:]
            sig_r2 = self.minimum_distance(point_x,point_y,point_z,selected_x,selected_y,selected_z)
            vect_energy = self.LJ_energy(sig_r2, 2)
            energy += sum(vect_energy)
        return energy/2.
    
    def check(self,x,y,z):
        for i in range(self.N):
            deltax = (x-self.x[i]) - self.Lx*np.rint((x-self.x[i])/self.Lx)
            deltay = (y-self.y[i]) - self.Ly*np.rint((y-self.y[i])/self.Ly)
            deltaz = (z-self.z[i]) - self.Lz*np.rint((z-self.z[i])/self.Lz)
            r2 = deltax**2 + deltay**2 + deltaz**2
            if  r2 / self.sigma**2 < 1:
                return True
        return False
        
    
    def delta_energy(self,nt,choice):
        if choice == 0:
            x = self.x[nt]
            y = self.y[nt]
            z = self.z[nt]
            point_x = np.zeros(self.N-1) ; point_x[...] = x
            point_y = np.zeros(self.N-1) ; point_y[...] = y
            point_z = np.zeros(self.N-1) ; point_z[...] = z
            
            trial_x = np.delete(self.x, nt)
            trial_y = np.delete(self.y, nt)
            trial_z = np.delete(self.z, nt)

            sig_r = self.minimum_distance(point_x, point_y, point_z, trial_x, trial_y, trial_z)
            delta_E = self.LJ_energy(sig_r,2)
            return sum(delta_E), trial_x, trial_y, trial_z
        elif choice == 1:
            x = np.random.uniform(0,self.Lx)
            y = np.random.uniform(0,self.Lx)
            z = np.random.uniform(0,self.Lx)
            cut = self.check(x,y,z)
            if cut == True:
                return 0, 0, 0, 0, True
            else:
                point_x = np.zeros(self.N) ; point_x[...] = x
                point_y = np.zeros(self.N) ; point_y[...] = y
                point_z = np.zeros(self.N) ; point_z[...] = z
                sig_r = self.minimum_distance(point_x, point_y, point_z, self.x, self.y, self.z)
                delta_E = self.LJ_energy(sig_r,2)
                return x, y, z, sum(delta_E), False

                
    def metropolis(self, T, zeta, nstep):
        acc_add = 0
        acc_rej = 0
        acc = 0
        energy = [self.energy]
        density = [self.density]
        M = np.zeros(nstep)
        density_hist = np.zeros(shape=(2,int(self.V+0.2*self.V)))
        for i in range(int(self.V+0.2*self.V)):
            density_hist[0,i] = i
        random_choice = np.random.randint(2,size=nstep)
        random_acceptance = np.random.random(nstep)
        
        for j in range(nstep):
            if self.N >0:
                choice = random_choice[j]
                nt = np.random.randint(self.N)
                if choice == 0:# and (self.N >= 0): #remove a particle
                    delta_E, trial_x, trial_y, trial_z = self.delta_energy(nt,choice)
                    if random_acceptance[j] <= self.N/(self.V*zeta)*np.exp(-delta_E/T):
                        self.x = trial_x
                        self.y = trial_y
                        self.z = trial_z
                        self.N -= 1
                        energy.append(energy[-1]-delta_E)
                        #density.append(self.N/self.V)
                        #density_hist[(self.N)] += 1.
                        acc += 1
                        acc_rej += 1 
                    else:
                        energy.append(energy[-1])
                        #density.append(density[-1])
                        #density_hist[self.N] += 1.
                elif choice == 1: #add a particle
                    trial_x, trial_y, trial_z, delta_E, cut = self.delta_energy(nt,choice)
                    #♦print(cut)
                    if cut == False and random_acceptance[j] <= zeta*self.V/(self.N+1)*np.exp(delta_E/T):
                        self.x = np.append(self.x, trial_x)
                        self.y = np.append(self.y, trial_y)
                        self.z = np.append(self.z, trial_z)
                        self.N += 1
                        energy.append(energy[-1]+delta_E)
                        #density.append(self.N/self.V)
                        #density_hist[self.N] += 1.
                        acc += 1
                        acc_add += 1
                    else:
                        energy.append(energy[-1])
                        density.append(density[-1])
            else:
                print("I've 0 particles")
                    #density_hist[self.N] += 1.
            M[j] = self.N
            density_hist[1,self.N] += 1.
            #print(j,self.N,energy[-1])
        plt.plot(range(len(M)),M/self.V)
        plt.title("Points")
        plt.show()
        
        plt.plot(range(len(energy)), np.array(energy)/self.epsilon)
        plt.title("Energy")
        plt.show()
        
        #plt.plot(range(len(density)), density); title("Density"); show()
        normalize = 1./sum(density_hist)
        plt.plot(density_hist[0,:]/self.V, density_hist[1,:]*normalize)
        plt.title("Density Prob.")
        plt.show()
        #self.write_out(T, conf_out='conf_in', gdr_out='gdr.out')
        #r,gdr = loadtxt('gdr.out', unpack=True, skiprows=1 )
        #fig2 = plt.figure()
        #plt.plot(r,gdr,'r-',r,ones(size(r)),'k-')
        plt.show()
        s = sum(random_choice)
        return energy, acc*100./nstep, acc_add*100/s, acc_rej*100/(nstep-s)
    
        
    def create_fcc_lattice(self):
      delta=0.00000001
      random_displacement_x = np.random.normal(0., delta, self.N)
      random_displacement_y = np.random.normal(0., delta, self.N)
      random_displacement_z = np.random.normal(0., delta, self.N)
      if self.a < np.sqrt(2.)*self.sigma:
          self.a = np.sqrt(2.)*self.sigma #set to the max possible lattice parameter for not having compenetration
          self.density = 4./self.a**3
          random_displacement_x[...] = 0.
          random_displacement_y[...] = 0.
          random_displacement_z[...] = 0.
      j  = 0
      xi = 0.
      yi = 0.
      zi = 0.
      for n in range(self.nx) :
          for m in range(self.ny) :
             for l in range(self.nz) :
                 self.x[j] = xi + self.a*n + random_displacement_x[j]
                 self.y[j] = yi + self.a*m + random_displacement_y[j]             
                 self.z[j] = zi + self.a*l + random_displacement_z[j]
                 j +=1
                 self.x[j] = xi + self.a*n + random_displacement_x[j] + 0.5*self.a
                 self.y[j] = yi + self.a*m + random_displacement_y[j] + 0.5*self.a     
                 self.z[j] = zi + self.a*l + random_displacement_z[j]
                 j +=1
                 self.x[j] = xi + self.a*n + random_displacement_x[j] + 0.5*self.a
                 self.y[j] = yi + self.a*m + random_displacement_y[j]             
                 self.z[j] = zi + self.a*l + random_displacement_z[j] + 0.5*self.a
                 j +=1
                 self.x[j] = xi + self.a*n + random_displacement_x[j] 
                 self.y[j] = yi + self.a*m + random_displacement_y[j] + 0.5*self.a            
                 self.z[j] = zi + self.a*l + random_displacement_z[j] + 0.5*self.a
                 j +=1
      self.energy = self.system_energy()
