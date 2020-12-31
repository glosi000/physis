from numpy import exp, arange, random, zeros
from matplotlib.pyplot import plot, figure

N = 64
Max = 1000
InT = arange(0.1, 6.0, 0.5)
Magnet = zeros(len(InT))
Energy = zeros(len(InT))
M = zeros((N+2,N+2))

u =0

for Temp in InT:
    Mag = 0.0
    Mag1 = 0.0
    En = 0.0

    for i in range(1,N+1):
        for j in range(1,N+1):
            M[i,j] = -1+2*random.randint(2)

    M[0,:] = M[N,:]
    M[N+1,:] = M[1,:]    
    M[:,0] = M[:,N]
    M[:,N+1] = M[:,1] 
    
    for l in range(1, 3*N**2):
        i = random.randint(1,N+1, size = 1)
        j = random.randint(1,N+1, size = 1)
        a = -M[i,j]
        Delta = 2*M[i,j]*(M[i+1,j] + M[i-1,j] + M[i,j-1] + M[i,j+1])
        r = random.uniform(0,1)
        if r <= exp(-Delta/Temp):
            M[i,j] = a
        if i == 1:
            M[N+1,j] = M[1,j]
        elif i == N:
            M[0,j] = M[N,j]
        if j == 1:
            M[i,N+1] = M[i,1]
        elif j == N:
            M[i,0] = M[i,N]
        
        
    for i in range(1,N+1):
        for j in range(1,N+1):
            Mag += M[i,j]
            En -= M[i,j]*(M[i,j-1]+M[i-1,j])
            
    for l in range(1, Max):
        i = random.randint(1,N+1, size = 1)
        j = random.randint(1,N+1, size = 1)
        a = -M[i,j]
        Delta = 2*M[i,j]*(M[i+1,j] + M[i-1,j] + M[i,j-1] + M[i,j+1])
        r = random.uniform(0,1)
        if r < exp(-Delta/Temp):
            M[i,j] = a
            En += Delta
            Mag += 2*a/Max
            if i == 1:
                M[N+1,j] = M[1,j]
            elif i == N:
                M[0,j] = M[N,j]
            if j == 1:
                M[i,N+1] = M[i,1]
            elif j == N:
                M[i,0] = M[i,N]


        
    Magnet[u] = abs(Mag/Max)
    Energy[u] = En/(Max*N**2)
    u = u + 1
    
plot(InT,Magnet)
figure(2)      
plot(InT,Energy)         

  
        

