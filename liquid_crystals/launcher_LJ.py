from LJ_phase_back import LJ
from numpy import exp

nx = 4
ny = 4
nz = 4
density = 0.6
radius = 0.5
epsilon = 1.

T = 1.04
mu = 9.98

system =  LJ(nx, ny, nz, density, radius, epsilon)
system.create_fcc_lattice()
energy, acc, acc_add, acc_rej = system.metropolis(T, zeta=exp(mu/T), nstep=60000)
print("acceptance rate",acc)
print("adding acc",acc_add)
print("rejecting acc",acc_rej)