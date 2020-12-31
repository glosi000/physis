from GIBBS_LOSI import *

n = 5
density = 0.323

T = 1.
Tfin = T
ntemp = 1

nstep = 2000
nswap = int(1.*4*n**3)
nvol= 1

deltaV = 40. ; delta = 0.35
init(n, density, T, Tfin, ntemp, delta, deltaV, nstep, nswap)