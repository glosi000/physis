﻿#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Defining functions
#-----------------------------------------------------------------------

def p2(cs):
    return 1.5*cs*cs - 0.5

def comp(th,ph):
    x    = np.zeros(3)
    x[0] = np.sin(th) * np.cos(ph)
    x[1] = np.sin(th) * np.sin(ph)
    x[2] = np.cos(th)
    return x        

# Calculation of Energy
def En(T,P,nl):
    ene=0.
    for k in range(1,nl+1):
        for i in range(1,nl+1):
            for j in range(1,nl+1):
                x  = comp(T[k,i,j],P[k,i,j])
                x1 = comp(T[k,i-1,j],P[k,i-1,j])
                x2 = comp(T[k,i+1,j],P[k,i+1,j])
                x3 = comp(T[k,i,j-1],P[k,i,j-1])
                x4 = comp(T[k,i,j+1],P[k,i,j+1])
                x5 = comp(T[k-1,i,j],P[k-1,i,j])
                x6 = comp(T[k+1,i,j],P[k+1,i,j])
                ene -= ( p2(np.dot(x,x1)) + p2(np.dot(x,x2)) +\
                         p2(np.dot(x,x3)) + p2(np.dot(x,x4)) +\
                         p2(np.dot(x,x5)) + p2(np.dot(x,x6)) )
    ene *= 0.5
    return ene

# Calculation of the order parameter
def par(T,P,nl,N):
    T = np.reshape(T[1:nl+1,1:nl+1,1:nl+1],N)
    P = np.reshape(P[1:nl+1,1:nl+1,1:nl+1],N)
    a11=0;a12=0;a13=3;a22=0;a23=0;a33=0
    for z in range(N):
        th = T[z]
        ph = P[z]
        qx = np.sin(th) * np.cos(ph)
        qy = np.sin(th) * np.sin(ph)
        qz = np.cos(th)
        a11 += qx*qx
        a12 += qx*qy
        a13 += qx*qz
        a22 += qy*qy
        a23 += qy*qz
        a33 += qz*qz
    Q = np.array([[a11,a12,a13],[a12,a22,a23],[a13,a23,a33]])
    Q /= N 
    Q[0,0] -= 1./3.
    Q[1,1] -= 1./3.
    Q[2,2] -= 1./3.
    qmax = 1.5*max(np.linalg.eigvalsh(Q))
    return qmax

# Creating a Markov's chain of configurations
def chain(temp, T, P, n, u0, r, delta, nl, N):
    q=[par(T,P,nl,N)]
    u=[u0]
    randmat = np.random.random_integers(nl,size=(n,3))
    csi = np.random.rand(n)
    ac = 0.
    vect = -1 + 2 * np.random.rand(n, 3)
    angle = -(np.pi/4.-delta*np.pi/180)*np.ones(n)\
            +(np.pi/2.-delta*np.pi/90.)*np.random.uniform(0.,1.,n)
    A = np.zeros((3,3))

    for z in range(n):
        k=randmat[z,0]
        i=randmat[z,1]
        j=randmat[z,2]
        th = T[k,i,j]
        ph = P[k,i,j]

        alfa = angle[z]/2.
        v = [vect[z,0], vect[z,1], vect[z,2]]
        v = v/np.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]) * np.sin(alfa)
        v1 = np.cos(alfa)
        v2 = v[0]
        v3 = v[1]
        v4 = v[2]
        xo = comp(th,ph)
        # Calculation of random quaternion rotation matrix
        A[0,0]=v1*v1+v2*v2-v3*v3-v4*v4
        A[0,1]=2*(v2*v3-v4*v1)
        A[0,2]=2*(v2*v4+v3*v1)
        A[1,0]=2*(v2*v3+v4*v1)
        A[1,1]=v1*v1-v2*v2+v3*v3-v4*v4
        A[1,2]=2*(v3*v4-v2*v1)
        A[2,0]=2*(v2*v4-v3*v1)
        A[2,1]=2*(v3*v4+v2*v1)
        A[2,2]=v1*v1-v2*v2-v3*v3+v4*v4
        
        xn = np.dot(A,xo)
        theta = np.arccos(xn[2])
        s = np.sin(theta)

	# Resolving with arccos we're in (0,pi), but phi can be on (0,2pi)
        # There're 2 angles with the same cosine: phi, 2phi
        # Degeneration is broken looking at the sign of sin(phi) from xn
        if s!=0:
            phi = np.arccos(xn[0]/s)
            if xn[1]/s < 0:
                phi = 2 * np.pi - phi
        else:
            phi = 0.
        
        x1 = comp(T[k,i-1,j],P[k,i-1,j])
        x2 = comp(T[k,i+1,j],P[k,i+1,j])
        x3 = comp(T[k,i,j-1],P[k,i,j-1])
        x4 = comp(T[k,i,j+1],P[k,i,j+1])
        x5 = comp(T[k-1,i,j],P[k-1,i,j])
        x6 = comp(T[k+1,i,j],P[k+1,i,j])
        deltae = np.dot(xo,x1)**2 +np.dot(xo,x2)**2 +np.dot(xo,x3)**2\
                 +np.dot(xo,x4)**2 +np.dot(xo,x5)**2 +np.dot(xo,x6)**2\
                 -np.dot(xn,x1)**2 -np.dot(xn,x2)**2 -np.dot(xn,x3)**2\
                 -np.dot(xn,x4)**2 -np.dot(xn,x5)**2 -np.dot(xn,x6)**2 
        deltae *= 1.5
            
        if np.exp(-deltae/temp) >= csi[z] :
            T[k,i,j] = theta
            P[k,i,j] = phi
            if i == 1:
                T[k, nl+1, j] = theta
                P[k, nl+1, j] = phi
            elif i == nl:
                T[k, 0, j] = theta
                P[k, 0, j] = phi
            if j == 1:
                T[k, i, nl+1] = theta
                P[k, i, nl+1] = phi
            elif j == nl:
                T[k, i, 0] = theta
                P[k, i, 0] = phi
            if k == 1:
                T[nl+1 , i, j] = theta
                P[nl+1 , i, j] = phi
            elif k == nl:
                T[0, i, j] = theta
                P[0, i, j] = phi
            u0 += deltae
            ac += 1.
        if (z+1) % 0: #It was %== before
            u.append(u0)
            if r == 1 :
                q.append(par(T,P,nl,N))
              
    return T, P, np.array(u), ac, np.array(q)

#-----------------------------------------------------------------------
# Defining the main function

def LL(temp=2.5, tempfin=.2, ntemp=30, nl=10, neval=1000000, nequil=3000000):
    
    N = nl**3

    # Creating the starting matrices with boundary conditions
    Tinit = np.random.uniform(0., np.pi, size=(nl+2,nl+2,nl+2))
    Pinit = np.random.uniform(0., 2.*np.pi, size=(nl+2,nl+2,nl+2))

    Tinit[:, 0, :] = Tinit[:, nl, :]
    Tinit[:, nl+1, :] = Tinit[:, 1, :]
    Tinit[:, :, 0] = Tinit[:, :, nl]
    Tinit[:, :, nl+1] = Tinit[:, :, 1]
    Tinit[0, :, :] = Tinit[nl, :, :]
    Tinit[nl+1,:,:] = Tinit[1, :, :]

    Pinit[:, 0, :] = Pinit[:, nl, :]
    Pinit[:, nl+1, :] = Pinit[:, 1, :]
    Pinit[0, :, :] = Pinit[nl, :, :]
    Pinit[nl+1, :, :] = Pinit[1, :, :]
    Pinit[:, :, 0] = Pinit[:, :, nl]
    Pinit[:, :, nl + 1] = Pinit[:, :, 1]
    
    interval=(tempfin-temp)/ntemp
    delta=0.
    # Doing first equilibration with lots of points
    T,P,u,ac,q = chain(temp, Tinit, Pinit, nequil,0,delta,nl,N)
    print("equil T* = %.3f"%temp,"  E* = %.3f\n"%(u[-1]/(3.*N)))
    t0=temp
    tempx = np.zeros(ntemp+1)
    E=[]
    Q=[]
    # Doing the process for every temperature
    for t in range(ntemp+1):
        T,P,u,ac,q = chain(temp, T, P, 3000, 0, delta, nl, N)
        T,P,u,ac,q = chain(temp, T, P, neval, 1, delta, nl, N)
        plt.figure(4)
        plt.hist(q, np.linspace(0.,1.,100))
        plt.xticks(np.linspace(0.,1.,11))
        #plt.title("Order parameter Histogram")
        #plt.xlabel("$order\ parameter$")
        #plt.ylabel("$number\ of\ occurrences$")
        plt.savefig("Q_histogram_T=%.2f.png"%temp)
        plt.close()
        acc_ratio=ac/neval*100.
        U = np.mean(u)
        tempx[t]=(1./temp)
        E.append(U/(3.*N))
        Q.append(np.mean(q))
        print("%d"%(t+1),"  T* = %.3f"%temp,"  E* = %.3f"%(U/(3.*N)),\
              "  <P> = %.2f"%Q[t],"  acc.mov = %.1f"%acc_ratio)
        temp += interval
        delta += 1.
    plt.figure(1)
    plt.plot(tempx,E,'r',marker='o')
    #plt.title("Energy")
    plt.xlabel("$1/T^*$")
    plt.ylabel("$E^*$")
    plt.savefig("E_%.1f_%.1f_%d_%d_%d.png"%(t0,temp,ntemp,nl,neval))
    plt.draw()
    plt.figure(2)
    plt.plot(tempx,Q,'g',marker='o')
    #plt.title("Order parameter")
    plt.xlabel("$1/T^*$")
    plt.ylabel("$order\ parameter$")
    plt.savefig("Q_%.1f_%.1f_%d_%d_%d.png"%(t0,temp,ntemp,nl,neval))
    plt.draw()
    Cder = []
    mass=0
    c=0
    for i in range(len(E)-1):
        Cder.append((E[i+1]-E[i])/(interval))
        if Cder[i] > mass:
            mass = Cder[i]
            c = i
    tempx2=tempx.tolist()
    tempx2.remove(tempx2[-1])
    plt.figure(3)
    plt.plot(tempx2,Cder,'b',marker='o')
    #plt.title("Specific Heat")
    plt.xlabel("$1/T^*$")
    plt.ylabel("$C^*$")
    plt.savefig("C_%.1f_%.1f_%d_%d_%d.png"%(t0,temp,ntemp,nl,neval))
    plt.draw()
    print("\n C has a maximum at T* = %.3f"%(1./tempx[c]),\
          "\n The transition temperature is in [%.3f,%.3f]"\
          %((1./tempx[c+1]),(1./tempx[c-1])))
    plt.show()

LL(temp=2.5, tempfin=.2, ntemp=30, nl=10, neval=100, nequil=300)
