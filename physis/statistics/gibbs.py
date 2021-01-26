#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pypplot as plt
from numba import jit
import time

#-------------------- LAUNCH

def init(n, density, T, Tfin, ntemp, delta, deltaV, nstep, nswap, nvol=1, sig=1., eps=1.):
    t1tot = time.time()
    
    a = (4/density)**(1/3.)
    L1 = a*n; L2 = a*n ; V1 = L1**3. ; V2 = L2**3.
    N1 = 4*n**3 ; N2 = 4*n**3
    sig2 = sig**2.
    r2max1 = (3.*sig)**2. ; r2max2 = (3.*sig)**2.
    eps *= 4.
    x1, y1, z1, x2, y2, z2, en1, en2, en6_1, en12_1, en6_2, en12_2 = create_fcc(a, n, L1, eps, sig2, r2max1)
    
    if T == Tfin or ntemp == 1:
        Tfin = T ; ntemp = 1
    deltaT = (Tfin-T)/ntemp
    
    if nstep % 2 != 0:
        nstep -= 1
    
    #open the file where many information will be written
    #the file will be closed at the end of the run
    result = open("RESULT_n=%d_den=%.3f_Tin=%.2f_Tfin=%.2f_ntemp=%d.txt"%(n,density,T,Tfin,ntemp),'w')
    
    for n in range(ntemp):
        t1 = time.time()
        result.write("#Doing Mc cycle with density=%.3f Tin=%.2f Tfin=%.2f Ntemp=%d\n#\n"%(density,T,Tfin,ntemp))
        result.write("##############################################\n#\n")        
        N1, L1, V1, x1, y1, z1, en1, en6_1, en12_1, N2, L2, V2, x2, y2, z2, en2, en6_2, en12_2 = metropolis(T, nstep, x1, y1, z1, x2, y2, z2, N1, N2, L1, L2, en1, en6_1, en12_1, en2, en6_2, en12_2, eps, sig2, r2max1, r2max2, result, nswap, nvol, delta, deltaV)
        t2 = time.time()
        result.write("# Time of %d ° cycle: %.2f (minutes)\n#\n"%(n+1,(t2-t1)/60.))
        result.write("##############################################\n")
        T += deltaT
        
    t2tot = time.time()
    result.write("##############################################\n#\n")
    result.write("# Total time: %.2f minutes  ,  %.2f hours\n#\n"%((t2tot-t1tot)/60.,(t2tot-t1tot)/3600.))       
    result.close()
    print("Total time: %.2f minutes"%((t2tot-t1tot)/60.))
    
#--------------------
#create an fcc lattice 
def create_fcc(a, n, L, eps, sig2, r2max):
    N = 4*n**3
    x1 = np.zeros(N); y1 = np.zeros(N); z1 = np.zeros(N)
    x2 = np.zeros(N); y2 = np.zeros(N); z2 = np.zeros(N)
    j  = 0 ; xi = 0. ; yi = 0. ; zi = 0. ; 
    for nx in range(n) :
        for ny in range(n) :
            for nz in range(n) :
                x1[j] = xi + a*nx
                y1[j] = yi + a*ny            
                z1[j] = zi + a*nz
                j +=1
                x1[j] = xi + a*nx + 0.5*a
                y1[j] = yi + a*ny + 0.5*a     
                z1[j] = zi + a*nz
                j +=1
                x1[j] = xi + a*nx + 0.5*a
                y1[j] = yi + a*ny    
                z1[j] = zi + a*nz + 0.5*a
                j +=1
                x1[j] = xi + a*nx
                y1[j] = yi + a*ny + 0.5*a          
                z1[j] = zi + a*nz + 0.5*a
                j +=1
    ll = L/2.
    x1 -= ll ; y1 -= ll ; z1 -= ll
    if sum(x1>ll)!=False:
        print('out of the box')
    elif sum(x1<-ll)!=False:
        print('out of the box')
    elif sum(y1>ll)!=False:
        print('out of the box')
    elif sum(y1<-ll)!=False:
        print('out of the box')
    elif sum(z1>ll)!=False:
        print('out of the box')
    elif sum(z1<-ll)!=False:
        print('out of the box')

    x2[...] = x1[...] ; y2[...] = y1[...] ; z2[...] = z1[...]
    en6_1, en12_1 = system_energy(N, x1, y1, z1, L, eps, sig2, r2max)
    en1 = en6_1 + en12_1
    #the energies are the same at the beginning
    return x1, y1, z1, x2, y2, z2, en1, en1, en6_1, en12_1, en6_1, en12_1
    
#--------------------
#calculate the energy of the system
def system_energy(N, x, y, z, L, eps, sig2, r2max):
    en6 = 0. ; en12 = 0.
#    pt_x = zeros(N)
#    pt_y = zeros(N)
#    pt_z = zeros(N)
    for i in range(N-1):
#        pt_x[...] = x[i]
#        pt_y[...] = y[i]
#        pt_z[...] = z[i]
        sig_r2 = minimum_distance(x[i], y[i], z[i], x[i+1:], y[i+1:], z[i+1:], L, sig2, r2max)
        energy6, energy12 = LJ_energy(sig_r2, eps)
        en6 += sum(energy6) ; en12 += sum(energy12)
    return en6, en12
    
#--------------------
@jit(nopython=True,cache=True,nogil=True)
def LJ_energy(sig_r,eps):
    return -eps*sig_r**3., eps*sig_r**6.

#-------------------- FUNCTIONS
#calculate the minimum distance between the two set of vectors x1,y1,z1 - x2,y2,z2
#@jit(nopython=True,cache=True,nogil=True)
def minimum_distance(x1, y1, z1, x2, y2, z2, L, sig2, r2max):
    deltax = x2-x1
    deltay = y2-y1
    deltaz = z2-z1
    min_deltax = deltax - L*np.rint(deltax/L)
    min_deltay = deltay - L*np.rint(deltay/L)
    min_deltaz = deltaz - L*np.rint(deltaz/L)
    one_r2 = 1./(min_deltax**2 +min_deltay**2 + min_deltaz**2)
    #I ask to have distances lower than the cut off (3*sigma)
    rr = 1./r2max
    for k in one_r2:
        if k < rr: #if the distance is greater than k
            k = 0.
#    cut = r2max*one_r2 > 1.
    #I get True (1) when r2 < r2max and False (0) when r2 >= r2max
    #Multiplying one_r2 by cut I put to zero the inverse of the distances that
    #are bigger respect to the cut-off. In this way I'll have zero contribution
    #to the energy from these distances.
    return sig2*one_r2#*cut
    
#--------------------
#calculate the energy of the particle number nt in the set x,y,z
def delta_energy(nt, x, y, z, N, L, eps, sig2, r2max):
#    xpt = zeros(N-1) ; xpt[...] = x[nt]
#    ypt = zeros(N-1) ; ypt[...] = y[nt]
#    zpt = zeros(N-1) ; zpt[...] = z[nt]
    xn = np.zeros(N-1) ; xn[:nt] = x[:nt] ; xn[nt:] = x[nt+1:]
    yn = np.zeros(N-1) ; yn[:nt] = y[:nt] ; yn[nt:] = y[nt+1:]
    zn = np.zeros(N-1) ; zn[:nt] = z[:nt] ; zn[nt:] = z[nt+1:]
    sig_r = minimum_distance(x[nt], y[nt], z[nt], xn, yn, zn, L, sig2, r2max)
    energy6, energy12 = LJ_energy(sig_r,eps)
    #In this way I found only the energy of the nt particle, looking at its
    #interaction with all the others. This is the energy contribution that 
    #changes if I move the particle nt (looking at its energy before and after)
    #or if I take it away from the box (its energy is no more present)
    return sum(energy6), sum(energy12)

#--------------------
#try to displace a particle
def displace(T, nt, x, y, z, N, L, en, en6, en12, eps, sig2, r2max, disp, ran):
    xt = x[nt] + disp[0]
    yt = y[nt] + disp[1]
    zt = z[nt] + disp[2]
    ll = L/2.
    if xt > ll:
        xt -= L
    elif xt < -ll:
        xt += L
    if yt > ll:
        yt -= L
    elif yt < -ll:
        yt += L
    if zt > ll:
        zt -= L
    elif zt < -ll:
        zt += L
#    en6old, en12old = delta_energy(nt, x, y, z, N, L, eps, sig2, r2max)

    xn = np.zeros(N-1) ; xn[:nt] = x[:nt] ; xn[nt:] = x[nt+1:]
    yn = np.zeros(N-1) ; yn[:nt] = y[:nt] ; yn[nt:] = y[nt+1:]
    zn = np.zeros(N-1) ; zn[:nt] = z[:nt] ; zn[nt:] = z[nt+1:]

    sig_r = minimum_distance(x[nt], y[nt], z[nt], xn, yn, zn, L, sig2, r2max)
    en6old, en12old = LJ_energy(sig_r,eps)
    en6old = sum(en6old) ; en12old = sum(en12old)
#    xnew = np.zeros(N) ; xnew[...] = x ; xnew[nt] = xt
#    ynew = np.zeros(N) ; ynew[...] = y ; ynew[nt] = yt
#    znew = np.zeros(N) ; znew[...] = z ; znew[nt] = zt
#    en6new, en12new = delta_energy(nt, xt, yt, zt, N, L, eps, sig2, r2max)
    sig_r = minimum_distance(xt, yt, zt, xn, yn, zn, L, sig2, r2max)
    en6new, en12new = LJ_energy(sig_r,eps)
    en6new=sum(en6new);en12new=sum(en12new)
    #displacement energy is given by the energy contribution changement of the 
    #particle nt, all the other particles remain as before
    DELTA = en12new + en6new - en6old - en12old
    if ran < np.exp(-DELTA/T): 
        #accept the move
        en6 += (en6new - en6old); en12 += (en12new - en12old)
        en += DELTA
        x[nt] = xt; y[nt] = yt; z[nt] = zt
        return 1, x, y, z, en, en6, en12
    else:
        return 0, x, y, z, en, en6, en12
      
#--------------------
#try to change the volume of the *add box and to reduce the *sub box
@jit(nopython=True,cache=True,nogil=True)               
def change_volume(T, deltaV, ran, Nadd, Ladd, Vadd, xadd, yadd, zadd, enadd, en6add, en12add,
                  Nsub, Lsub, Vsub, xsub, ysub, zsub, ensub, en6sub, en12sub, r2add, r2sub):
    Vadd_new = Vadd + deltaV ; Vsub_new = Vsub - deltaV
    Ladd_new = Vadd_new**(1/3.) ; Lsub_new = Vsub_new**(1/3.)
    enadd_new = en6add * (Ladd / Ladd_new)**6. + en12add * (Ladd / Ladd_new)**12.
    ensub_new = en6sub * (Lsub / Lsub_new)**6. + en12sub * (Lsub / Lsub_new)**12.
    DELTA_add = enadd_new - enadd
    DELTA_sub = ensub_new - ensub
    
    if ran < ((Vadd_new/Vadd)**Nadd)*((Vsub_new/Vsub)**Nsub)*np.exp(-(DELTA_add + DELTA_sub)/T) and Vsub_new>0.:
        en6add *= (Ladd / Ladd_new)**6. ; en12add *= (Ladd/ Ladd_new)**12.
        en6sub *= (Lsub / Lsub_new)**6. ; en12sub *= (Lsub/ Lsub_new)**12.        
        xadd *= (Ladd_new / Ladd) ; yadd *= (Ladd_new / Ladd) ; zadd *= (Ladd_new / Ladd)
        xsub *= (Lsub_new / Lsub) ; ysub *= (Lsub_new / Lsub) ; zsub *= (Lsub_new / Lsub)
#        r2add = r2add*(Ladd_new / Ladd)**2. ; r2sub = r2sub*(Lsub_new / Lsub)**2.
        return 1, Ladd_new, Vadd_new, xadd, yadd, zadd, enadd_new, en6add, en12add, Lsub_new, Vsub_new, xsub, ysub, zsub, ensub_new, en6sub, en12sub, r2add, r2sub
    else:
        return 0, Ladd, Vadd, xadd, yadd, zadd, enadd, en6add, en12add, Lsub, Vsub, xsub, ysub, zsub, ensub, en6sub, en12sub, r2add, r2sub

#--------------------
#try to swap a particle from *out ensemble into *in ensemble
def swap_particle(T, nt, ran, Nout, Lout, Vout, xout, yout, zout, enout, en6out, en12out,
                  Nin, Lin, Vin, xin, yin, zin, enin, en6in, en12in, eps, sig2, r2out, r2in):
#    delta6out, delta12out = delta_energy(nt, xout, yout, zout, Nout, Lout, eps, sig2, r2out)
    xn = np.zeros(Nout-1) ; xn[:nt] = xout[:nt] ; xn[nt:] = xout[nt+1:]
    yn = np.zeros(Nout-1) ; yn[:nt] = yout[:nt] ; yn[nt:] = yout[nt+1:]
    zn = np.zeros(Nout-1) ; zn[:nt] = zout[:nt] ; zn[nt:] = zout[nt+1:]
    sig_r = minimum_distance(xout[nt], yout[nt], zout[nt], xn, yn, zn, Lout, sig2, r2out)
    delta6out, delta12out = LJ_energy(sig_r,eps)
    delta6out = sum(delta6out) ; delta12out = sum(delta12out)
    #I'm taking away the particle nt. The energy box varies of -(energy of nt particle)
    DELTA_out = -(delta6out + delta12out)
    #Extract randomly the particle position in the *in box
    xt, yt, zt = np.random.uniform(-Lin/2.,Lin/2., 3)
#    xpt = zeros(Nin) ; xpt[...] = xt
#    ypt = zeros(Nin) ; ypt[...] = yt
#    zpt = zeros(Nin) ; zpt[...] = zt
    #Now look at the interaction energy of the particle I want to insert inside the box
    sig_r = minimum_distance(xt, yt, zt, xin, yin, zin, Lin, sig2, r2in)
    delta6in, delta12in = LJ_energy(sig_r, eps)
    delta6in = sum(delta6in) ; delta12in = sum(delta12in)
    #The energy of the particle nt will be added to the *in box energy if the move is accepted
    DELTA_in = delta6in + delta12in
    if ran < (Vin/(Nin+1))*(Nout/Vout)*np.exp(-(DELTA_in + DELTA_out)/T):
        xni = np.zeros(Nin+1) ; xni[:Nin] = xin[...] ; xni[Nin] = xt
        yni = np.zeros(Nin+1) ; yni[:Nin] = yin[...] ; yni[Nin] = yt
        zni = np.zeros(Nin+1) ; zni[:Nin] = zin[...] ; zni[Nin] = zt
#        xout = delete(xout, nt) ; yout = delete(yout, nt) ; zout = delete(zout, nt)
        en6out -= delta6out ; en12out -= delta12out ; enout += DELTA_out
        en6in += delta6in ; en12in += delta12in ; enin += DELTA_in
        return 1, xn, yn, zn, enout, en6out, en12out, xni, yni, zni, enin, en6in, en12in
    else:
        return 0, xout, yout, zout, enout, en6out, en12out, xin, yin, zin, enin, en6in, en12in

#-------------------- EXTRA

#calculate the chemical potential using the particle addition techique
def chem_pot(x, y, z, N, L, eps, sig2, r2max):
    ll = L/2.
    xt = np.random.uniform(-ll,ll)
    yt = np.random.uniform(-ll,ll)
    zt = np.random.uniform(-ll,ll)
#    xpt = zeros(N) ; xpt[...] = xt
#    ypt = zeros(N) ; ypt[...] = yt
#    zpt = zeros(N) ; zpt[...] = zt
    sig_r = minimum_distance(xt, yt, zt, x, y, z, L, sig2, r2max)
    energy6, energy12 = LJ_energy(sig_r, eps)
    energy6 = sum(energy6) ; energy12 = sum(energy12)
    return energy6+energy12
    
#----------------------
#used inside radialddistr
@jit(nopython=True,cache=True,nogil=True)
def countg(N, x, y, z, gcount, L, ldel, kg, sig2, r2max):
    RMAX = (L/2.)**2
    xpt = np.zeros(N)
    ypt = np.zeros(N)
    zpt = np.zeros(N)
    for k in range(N-1):
        j=k+1
        xpt[...] = x[k]
        ypt[...] = y[k]
        zpt[...] = z[k]
        deltax = x[j:]-xpt[j:]
        deltay = y[j:]-ypt[j:]
        deltaz = z[j:]-zpt[j:]
        min_deltax = deltax - L*np.rint(deltax/L)
        min_deltay = deltay - L*np.rint(deltay/L)
        min_deltaz = deltaz - L*np.rint(deltaz/L)
        r2 = (min_deltax**2 +min_deltay**2 + min_deltaz**2)
        b = r2 < RMAX
        lm  = np.sqrt(r2[b])
        for elm in lm :
            gcount[int(elm/ldel)]+=2.
    return gcount

#--------------------    
#calculate the radial distribution
def radialdistr(T, nth, x, y, z, en, en6, en12, N, L, delta, eps, sig2, r2max):    
    ldel=0.01; kg=int(L/ldel)+1
    inter = 100
    gcount = np.zeros(kg)
    ran_acc = np.random.random(nth)
    ran_disp = delta*(1.-2.*np.random.random(size=(3,nth)))
    nt = np.random.randint(0, N, size=nth)
    for i in range(nth):
        acc, x, y, z, en, en6, en12 = displace(T, nt[i], x, y, z, N, L, en, en6, en12, eps, sig2, r2max, ran_disp[:,i], ran_acc[i])
        if i%inter == 0:        
            gcount = countg(N, x, y, z, gcount, L, ldel, kg, sig2, r2max)
    V = np.zeros(kg)
    r = np.zeros(kg)
    g = np.zeros(kg)
    rho = N/(L**3)
    for lm in range(kg) :
          V[lm] = 4./3.*np.pi*(ldel**3)*(3*lm*lm +3*lm + 1); 
          g[lm] = gcount[lm]/(V[lm]*(N -1)*T*rho);
          r[lm] = (lm+0.5)*ldel
    return r[:int(kg/2)], g[:int(kg/2)]#/(int(nth/inter))

#--------------------
#save all datas and make all plots
def save_and_plot(result, T, ntot, nvar, pres, energy1, density1, mu1, pressure1, r1, g1, energy2, density2, mu2, pressure2, r2, g2, x_plot, y_plot, N1, L1, x1, y1, z1, en1, en6_1, en12_1, N2, x2, y2, z2, L2, en2, en6_2, en12_2, hist_x, hist1_y, hist2_y):
    
    nmes = int(ntot/(2*nvar))
    nhalf = int(ntot/2)
    m1 = np.mean(mu1) ; m2 = np.mean(mu2)
    errm1 = mu1.std(ddof=1) ; errm2 = mu2.std(ddof=1)
    p1 = np.mean(pressure1[nhalf:]) ; p2 = np.mean(pressure2[nhalf:])
    errp1 = pressure1[nhalf:].std(ddof=1) ; errp2 = pressure2[nhalf:].std(ddof=1)
    men1 = np.mean(energy1[nhalf:]) ; men2 = np.mean(energy2[nhalf:])
    errmen1 = energy1[nhalf:].std(ddof=1); errmen2 = energy2[nhalf:].std(ddof=1);
    mden1 = np.mean(density1[nhalf:]); mden2 = np.mean(density2[nhalf:])
    errden1 = density1[nhalf:].std(ddof=1); errden2 = density2[nhalf:].std(ddof=1);
          
    
    plt.plot(x1,y1,'ro') ; plt.title("gas xy-plane"); plt.savefig("box1_xy_T=%.2f.png"%T); plt.show()
    plt.plot(x2,y2,'bo') ; plt.title("liquid 2 xy-plane"); plt.savefig("box2_xy_T=%.2f.png"%T); plt.show()
    
    plt.plot(range(ntot), energy1, 'r', label='gas')
    plt.plot(range(ntot), energy2, 'b', label='liquid')
    plt.xlabel("n. step"); plt.ylabel("energy"); plt.title("Energies"); plt.legend(loc='upper left');
    plt.savefig("energies_T=%.2f.png"%T); plt.show()
    
    plt.plot(range(ntot), density1, 'r', label='gas')
    plt.plot(range(ntot), density2, 'b', label='liquid')
    plt.xlabel("n. step"); plt.ylabel("density"); plt.title("Densities"); plt.legend(loc='upper left');
    plt.savefig("densities_T=%.2f.png"%T); plt.show()
    
    plt.plot(range(nmes-1), mu1, 'r', label='gas')
    plt.plot(range(nmes-1), mu2, 'b', label='liquid')
    plt.xlabel("n. step"); plt.ylabel("chem. pot."); plt.title("Chemical Potentials"); plt.legend(loc='upper left');
    plt.savefig("chempot_T=%.2f.png"%T); plt.show()
    
    plt.plot(range(ntot), pressure1, 'r', label='gas')
    plt.plot(range(ntot), pressure2, 'b', label='liquid')
    plt.xlabel("n. step"); plt.ylabel("pressure"); plt.title("Pressures"); plt.legend(loc='upper left');
    plt.savefig("pressure_T=%.2f.png"%T); plt.show()
    
    plt.plot(r1, g1, 'r')
    plt.xlabel("r/sig"); plt.ylabel("g(r)"); plt.title("Radial distr. gas");
    plt.savefig("g1_T=%.2f.png"%T); plt.show()
    
    plt.plot(r2, g2, 'b')
    plt.xlabel("r/sig"); plt.ylabel("g(r)"); plt.title("Radial distr. liquid");
    plt.savefig("g2_T=%.2f.png"%T); plt.show()
    
    plt.plot(hist_x,hist1_y,'r') ; plt.plot(hist_x,hist2_y,'b')
    plt.title("density distribution"); plt.savefig("den_dist_T=%.2f.png"%T); plt.show()
    
    plt.plot(x_plot, y_plot,'oy'); plt.plot((1.-x_plot), (1.-y_plot),'og')
    plt.xlim([0.,1.]); plt.ylim([0.,1.]); plt.xlabel("x=N1/N"); plt.ylabel("y=V1/V"); 
    plt.savefig("xy_plot_T=%.2f.png"%T); plt.show()
    
    print("Densities: %.3f +- %.3f   %.3f +- %.3f"%(mden1,errden1,mden2,errden2))
    print("\nMean Energies: \n", men1, " +- ", errmen1, "\n", men2, " +- ", errmen2)
    print("\nChemical Potentials: \n", m1, " +- ", errm1, "\n", m2, " +- ", errm2, "\nMean: ", (m1+m2)/2.)
    print("\nPressures: \n", p1, " +- ", errp1, "\n", p2, " +- ", errp2, "\nMean: ",(p1+p2)/2.)
    print("\nDisplace box-1: pres = %.2f  acc = %.2f\n"%(pres[0],pres[1]))
    print("Displace box-2: pres = %.2f  acc = %.2f\n"%(pres[2],pres[3]))
    print("Vol.Change 1-grow: pres = %.2f  acc = %.2f\n"%(pres[4],pres[5]))
    print("Vol.Change 2-grow: pres = %.2f  acc = %.2f\n"%(pres[6],pres[7]))
    print("Par.Swap 2->1: pres = %.2f  acc = %.2f\n"%(pres[8],pres[9]))
    print("Par.Swap 1->2: pres = %.2f  acc = %.2f\n"%(pres[10],pres[11]))
    
    data = open("data_T=%.2f.txt"%T,'w')
    for i in range(ntot):
        data.write("%f %f %f %f\n"%(energy1[i],density1[i],energy2[i],density2[i]))
    data.close()

    xy = open("xy_plot_T=%.2f.txt"%T,'w')
    for i in range(nmes):
        xy.write("%f %f\n"%(x_plot[i], y_plot[i]))
    xy.close()
    
    potpres = open("chempot_pres_T=%.2f.txt"%T,'w')
    for i in range(nmes-1):
        potpres.write("%f %f %f %f\n"%(mu1[i], pressure1[i], mu2[i], pressure2[i]))
    potpres.close()
    
    if N1*N2>0:
        gdr = open("gdr1_T=%.2f.txt"%T, 'w')
        for i in range(len(r1)):
            gdr.write("%f %f\n"%(r1[i], g1[i]))
        gdr.close()  
        gdr2 = open("gdr2_T=%.2f.txt"%T, 'w')
        for i in range(len(r2)):
            gdr2.write("%f %f\n"%(r2[i], g2[i]))
        gdr2.close()
    
    den = open("den_dist_T=%.2f.txt"%T, 'w')
    for i in range(len(hist_x)):
        den.write("%f %f %f\n"%(hist_x[i], hist1_y[i], hist2_y[i]))
    den.close()
    
    coord1 = open("coord1_T=%.2f.txt"%T, 'w')
    coord1.write("%f %f %f %d %f\n"%(en1, en6_1, en12_1, N1, L1))
    for i in range(N1):
        coord1.write("%f %f %f\n"%(x1[i],y1[i],z1[i]))
    coord1.close()
    
    coord2 = open("coord2_T=%.2f.txt"%T, 'w')
    coord2.write("%f %f %f %d %f\n"%(en2, en6_2, en12_2, N2, L2))
    for i in range(N2):
        coord2.write("%f %f %f\n"%(x2[i],y2[i],z2[i]))
    coord2.close()    
    
    result.write("# MC cycle with T = %.2f ntot = %d\n#\n# Percentage of presence and acceptance for the steps:\n"%(T,ntot))
    result.write("# Displace box-1: pres = %.2f  acc = %.2f\n"%(pres[0],pres[1]))
    result.write("# Displace box-2: pres = %.2f  acc = %.2f\n"%(pres[2],pres[3]))
    result.write("# Vol.Change 1-grow: pres = %.2f  acc = %.2f\n"%(pres[4],pres[5]))
    result.write("# Vol.Change 2-grow: pres = %.2f  acc = %.2f\n"%(pres[6],pres[7]))
    result.write("# Par.Swap 1->2: pres = %.2f  acc = %.2f\n"%(pres[8],pres[9]))
    result.write("# Par.Swap 2->1: pres = %.2f  acc = %.2f\n#\n"%(pres[10],pres[11]))
    result.write("# Box-1: N = %d  L = %.2f  En. = %.3f  En.6 = %.3f  En.12 = %.3f\n"%(N1, L1, en1, en6_1, en12_1))
    result.write("# Box-2: N = %d  L = %.2f  En. = %.3f  En.6 = %.3f  En.12 = %.3f\n#\n"%(N2, L2, en2, en6_2, en12_2))
    result.write("# Mean energies: %.3f+-%.3f   %.3f+-%.3f\n#\n"%(men1, errmen1, men2, errmen2))
    result.write("# Densities: %.3f+-%.3f   %.3f+-%.3f\n#\n"%(mden1,errden1, mden2, errden2))
    result.write("# Chemical Potentials: %.3f+-%.3f   %.3f+-%.3f\n# Mean: %.3f\n#\n"%(m1, errm1, m2, errm2, (m1+m2)/2.))
    result.write("# Pressures: %.3f+-%.3f   %.3f+-%.3f\n# Mean: %.3f\n#\n"%(p1, errp1, p2, errp2, (p1+p2)/2.)) 
    result.write("# Energy correlation %.3f"%(np.mean((energy1[nhalf:]-men1)*(energy2[nhalf:]-men2))/errmen1/errmen2))
    result.write("\n# Density correlation %.3f"%(np.mean((density1[nhalf:]-mden1)*(density2[nhalf:]-mden2))/errden1/errden2))
    result.write("\n# Chem. pot. correlation %.3f"%(np.mean((mu1-m1)*(mu2-m2))/errm1/errm2))
    result.write("\n# Pressure correlation %.3f"%(np.mean((pressure1[nhalf:]-p1)*(pressure2[nhalf:]-p2))/errp1/errp2))
    result.write("\n#\n")
                 
    return result

#-------------------- METROPOLIS
    
def metropolis(T, nstep, x1, y1, z1, x2, y2, z2, N1, N2, L1, L2, en1, en6_1, 
               en12_1, en2, en6_2, en12_2, eps, sig2, r2max1, r2max2, result,
               nswap=128, nvol=1, delta=0.4, deltaV=30.):
#V initially is about 850 so deltaV must be "high", otherwise acceptance=100%
    ld = 300
    N = N1+N2 ; V1 = L1**3 ; V2 = L2**3 ; V = V1+V2
    nvar = int(N1 + N2 + nvol + nswap)
    ntot = int(nvar*nstep)
    nhalf = int(ntot/2)
    nmes = int(ntot/(2*nvar))
    
    pres_ratio = np.zeros(6)
    acc_ratio = np.zeros(6)
    
    energy1 = np.zeros(ntot)
    density1 = np.zeros(ntot)
    energy2 = np.zeros(ntot)
    density2 = np.zeros(ntot)
    
    pressure1 = np.zeros(ntot)
    pressure2 = np.zeros(ntot)
    
    den_mu1 = np.zeros(nvar)
    den_mu2 = np.zeros(nvar)
    delta_mu1 = np.zeros(nvar)
    delta_mu2 = np.zeros(nvar)    
    vol_plot = np.zeros(nvar)
    num_plot = np.zeros(nvar)    
    
    mu1 = np.zeros(nmes)
    mu2 = np.zeros(nmes)
    x_plot = np.zeros(nmes)
    y_plot = np.zeros(nmes)

    k = 0 ; m = 0 ; q = 0 ; acc = 0 
#    nt1 = random.randint(0,N1,size=ntot)
#    nt2 = random.randint(0,N2,size=ntot)
    ran_choice = np.random.uniform(0.,1.,ntot)
    ran_acc = np.random.uniform(0.,1.,ntot)
    ran_disp = delta * (1 - 2 * np.random.uniform(0., 1., size=(3, ntot)))
    ran_vol = deltaV * np.random.uniform(0., 1., size=ntot)
    ran_vol_ext = np.random.random(ntot)
    
    r1 = N1/nvar
    r2 = N/nvar
    r3 = (N+nvol)/nvar
    r4 = (N+nvol+ 0.5*nswap)/nvar
    
    vol1 = np.zeros(ntot)
    vol2 = np.zeros(ntot)
    num1 = np.zeros(ntot)
    num2 = np.zeros(ntot)
    
    for j in range(nstep):
        for i in range(nvar):
            if ran_choice[k] < r1 and N1>0: #displace randomly a particle in box-1
                nt = np.random.randint(0, N1)                
                pres_ratio[0] += 1
                acc, x1, y1, z1, en1, en6_1, en12_1 = displace(T, nt, x1, y1, z1, N1, L1,
                                                 en1, en6_1, en12_1, eps, sig2,
                                                 r2max1, ran_disp[:,k], ran_acc[k])
                acc_ratio[0] += acc
            elif ran_choice[k] < r2 and N2>0:#displace randomly a particle in box-2
                nt = np.random.randint(0,N2)                
                pres_ratio[1] += 1
                acc, x2, y2, z2, en2, en6_2, en12_2 = displace(T, nt, x2, y2, z2, N2, L2,
		                                                  en2, en6_2, en12_2, eps, sig2,
		                                                  r2max2, ran_disp[:,k], ran_acc[k])
                acc_ratio[1] += acc
            elif ran_choice[k] < r3: #change volume
                if ran_vol_ext[k] < 0.5: #increase box-1, decrease box-2
                    pres_ratio[2] += 1
                    acc, L1, V1, x1, y1, z1, en1, en6_1, en12_1, L2, V2, x2, y2, z2, en2, en6_2, en12_2, r2max1, r2max2 = change_volume(T, ran_vol[k], ran_acc[k], N1, L1, V1, x1, y1, z1, en1, en6_1, en12_1, N2, L2, V2, x2, y2, z2, en2, en6_2, en12_2, r2max1, r2max2)
                    acc_ratio[2] += acc
                else: #increase box-2, decrease box-1
                    pres_ratio[3] += 1 
                    acc, L2, V2, x2, y2, z2, en2, en6_2, en12_2, L1, V1, x1, y1, z1, en1, en6_1, en12_1, r2max2, r2max1 = change_volume(T, ran_vol[k], ran_acc[k], N2, L2, V2, x2, y2, z2, en2, en6_2, en12_2, N1, L1, V1, x1, y1, z1, en1, en6_1, en12_1, r2max2, r2max1)
                    acc_ratio[3] += acc


            elif ran_choice[k] < r4 and N2>0: #swap a particle from 2 to 1
                #I cannot have an empty box otherwise when recalculating nt2 I get error
                nt = np.random.randint(0, N2)                
                pres_ratio[4] += 1
                acc, x2, y2, z2, en2, en6_2, en12_2, x1, y1, z1, en1, en6_1, en12_1 = swap_particle(T, nt, ran_acc[k],
                N2, L2, V2, x2, y2, z2, en2, en6_2, en12_2, N1, L1, V1, x1, y1, z1, en1, en6_1, en12_1,eps, sig2, r2max2, r2max1)                    
                if acc == 1:
                    N1 += 1 ; N2 -= 1
                    acc_ratio[4] += acc
                        #r1 = N1/nvar
                    
                    
            elif N1>0: #swap a particle from 1 to 2
                #I cannot have an empty box otherwise when recalculating nt1 I get error
                nt = np.random.randint(0,N1)                
                pres_ratio[5] += 1
                acc, x1, y1, z1, en1, en6_1, en12_1, x2, y2, z2, en2, en6_2, en12_2 = swap_particle(T, nt, ran_acc[k],
                N1, L1, V1, x1, y1, z1, en1, en6_1, en12_1, N2, L2, V2, x2, y2, z2, en2, en6_2, en12_2,eps, sig2, r2max1, r2max2)    
                if acc == 1:
                    N1 -= 1 ; N2 += 1
                    acc_ratio[5] += acc
                        #r1 = N1/nvar
                    
            energy1[k] = en1        
            energy2[k] = en2
            density1[k] = N1/V1
            density2[k] = N2/V2           
            pressure1[k] = T*N1/V1 + 6*(2*en12_1+en6_1)/(3*V1)
            pressure2[k] = T*N2/V2 + 6*(2*en12_2+en6_2)/(3*V2)  
            num1[k] = N1 ; num2[k] = N2
            vol1[k] = V1 ; vol2[k] = V2        
            
            if k > nhalf:
                #delta /= 1.8
                delta_mu1[m]=chem_pot(x1, y1, z1, N1, L1, eps, sig2, r2max1)
                delta_mu2[m]=chem_pot(x2, y2, z2, N2, L2, eps, sig2, r2max2)
                den_mu1[m] = density1[k]
                den_mu2[m] = density2[k]
                num_plot[m] = N1
                vol_plot[m] = V1
                m += 1
            k +=1
        if k > nhalf:
            mu1[q] = T*(np.log(np.mean(den_mu1) / np.mean(np.exp(-delta_mu1/T))))
            mu2[q] = T*(np.log(np.mean(den_mu2) / np.mean(np.exp(-delta_mu2/T))))
            x_plot[q] = np.mean(num_plot)
            y_plot[q] = np.mean(vol_plot)
            q += 1
            m = 0
            
    ll = L1/2.
    if sum(x1>ll)!=False:
        print('out of the box')
    elif sum(x1<-ll)!=False:
        print('out of the box')
    elif sum(y1>ll)!=False:
        print('out of the box')
    elif sum(y1<-ll)!=False:
        print('out of the box')
    elif sum(z1>ll)!=False:
        print('out of the box')
    elif sum(z1<-ll)!=False:
        print('out of the box')
    ll = L2/2.
    if sum(x2>ll)!=False:
        print('out of the box')
    elif sum(x2<-ll)!=False:
        print('out of the box')
    elif sum(y2>ll)!=False:
        print('out of the box')
    elif sum(y2<-ll)!=False:
        print('out of the box')
    elif sum(z2>ll)!=False:
        print('out of the box')
    elif sum(z2<-ll)!=False:
        print('out of the box') 
    
    x_plot /= N ; y_plot /= V    
    hist1_y = np.zeros(ld)
    hist2_y = np.zeros(ld)
    hist_x = np.zeros(ld)

    for i in range(ld):
        hist_x[i] = i/ld
    den1 = np.sort(density1[nhalf:])
    den2 = np.sort(density2[nhalf:])
    j = 0
    for i in range(ld):
        while den1[j] < hist_x[i] and j < nhalf-1:
            hist1_y[i] += 1
            j +=1
    j = 0
    for i in range(ld):
        while den2[j] < hist_x[i] and j < nhalf-1:
            hist2_y[i] += 1
            j +=1
    hist1_y /= sum(hist1_y) ; hist2_y /= sum(hist2_y)
    
    pres=pres_ratio*100./ntot; acc=acc_ratio*100./pres_ratio
    
    if N1*N2>0: #If one of the boxes is empty I don't do the radialdistr calculation
        r1, g1 =radialdistr(T, 100, x1, y1, z1, en1, en6_1, en12_1, N1, L1, delta, eps, sig2, r2max1)
        r2, g2 =radialdistr(T, 100, x2, y2, z2, en2, en6_2, en12_2, N2, L2, delta, eps, sig2, r2max2)
    else:
        r1 = 0; r2 = 0; g1 = 0; g2 = 0
    
    #now save all the data. I save gas and liquid phases with red, blue colors
    if np.mean(density1[nhalf:]) <= np.mean(density2[nhalf:]):
        plt.plot(num1,'r')
        plt.plot(num2,'b')
        plt.xlabel("step")
        plt.ylabel("n.part.")
        plt.title("Particles")
        plt.savefig("number_T=%.2f.png"%T)
        plt.show()
        
        plt.plot(vol1,'r')
        plt.plot(vol2,'b')
        plt.xlabel("step")
        plt.ylabel("volume")
        plt.title("Volume")
        plt.savefig("volume_T=%.2f.png"%T)
        plt.show()
         
        pres = (pres[0], acc[0], pres[1], acc[1], pres[2], acc[2], 
                pres[3], acc[3], pres[4], acc[4], pres[5], acc[5])
        result=save_and_plot(result, T, ntot, nvar, pres, energy1, density1, mu1[1:], pressure1, r1, g1, energy2, density2, mu2[1:], pressure2, r2, g2, x_plot, y_plot, N1, L1, x1, y1, z1, en1, en6_1, en12_1, N2, x2, y2, z2, L2, en2, en6_2, en12_2, hist_x, hist1_y, hist2_y)    
    else:
        plt.plot(num1,'b')
        plt.plot(num2,'r')
        plt.xlabel("step")
        plt.ylabel("n-part-")
        plt.title("Particles")
        plt.savefig("number_T=%.2f.png"%T)
        plt.show()
        
        plt.plot(vol1,'b')
        plt.plot(vol2,'r')
        plt.xlabel("step")
        plt.ylabel("volume")
        plt.title("Volume")
        plt.savefig("volume_T=%.2f.png"%T)
        plt.show()

        pres = (pres[1], acc[1], pres[0], acc[0], pres[3], acc[3], 
                pres[2], acc[2], pres[5], acc[5], pres[4], acc[4])
        result=save_and_plot(result, T, ntot, nvar, pres, energy2, density2, mu2[1:], pressure2, r2, g2, energy1, density1, mu1[1:], pressure1, r1, g1, x_plot, y_plot, N2, L2, x2, y2, z2, en2, en6_2, en12_2, N1, x1, y1, z1, L1, en1, en6_1, en12_1, hist_x, hist2_y, hist1_y)     
    return N1, L1, V1, x1, y1, z1, en1, en6_1, en12_1, N2, L2, V2, x2, y2, z2, en2, en6_2, en12_2            
