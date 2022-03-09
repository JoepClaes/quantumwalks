from scipy.integrate import odeint
import math
import numpy as np
import matplotlib.pyplot as plt

n=100   #number of steps in walk

pos_wall = -30

#initial state: (up_initial*|up> + down_initial*|down>) tensor |pos0>
pos0 = 0    #initial position
up_initial = 1/math.sqrt(2)
down_initial = 1j/math.sqrt(2)
#up_initial = 1
#down_initial = 0

positions = np.arange(-n+pos0,n+1+pos0,2)

c_up = np.zeros(2*n+1,dtype=complex)
c_down = np.zeros(2*n+1,dtype=complex)

def qrw(n):
    c_up[n] = up_initial
    c_down[n] = down_initial
    for i in range(n):
        #print(c_up,c_down)
        c_up_new = np.zeros(len(c_up),dtype=complex)
        c_down_new = np.zeros(len(c_down),dtype=complex)
        for k in range(2*n):
            c_up_new[k] = c_down[k-1] + c_up[k-1]
            c_down_new[k] = -1*c_down[k+1] + c_up[k+1]
        #print(c_up_new,c_down_new)
        for x in range(2*n):
            c_up[x] = c_up_new[x]
            c_down[x] = c_down_new[x]
    prob_up = np.zeros(len(c_up),dtype=complex)
    prob_down = np.zeros(len(c_down),dtype=complex)
    prob_tot = np.zeros(len(c_up),dtype=complex)
    for i in range(len(c_up)):
        prob_up[i] = np.abs(c_up[i])**2
        prob_down[i] = np.abs(c_down[i])**2
        prob_tot[i] = (prob_up[i] + prob_down[i])/(2**n)
    return prob_tot

def absorption(n,pos_wall):
    c_up[n] = up_initial
    c_down[n] = down_initial
    for i in range(n):
        print(c_up,c_down)
        c_up_new = np.zeros(len(c_up),dtype=complex)
        c_down_new = np.zeros(len(c_down),dtype=complex)
        for k in range(2*n):
            c_up_new[k] = c_down[k-1] + c_up[k-1]
            c_down_new[k] = -1*c_down[k+1] + c_up[k+1]
        #print(c_up_new,c_down_new)
        for x in range(2*n):
            c_up[x] = c_up_new[x]
            c_down[x] = c_down_new[x]
            c_up[n+pos_wall] = 0
            c_down[n+pos_wall] = 0
    prob_up = np.zeros(len(c_up),dtype=complex)
    prob_down = np.zeros(len(c_down),dtype=complex)
    prob_tot = np.zeros(len(c_up),dtype=complex)
    for i in range(len(c_up)):
        prob_up[i] = np.abs(c_up[i])**2
        prob_down[i] = np.abs(c_down[i])**2
        prob_tot[i] = (prob_up[i] + prob_down[i])/(2**n)
    return prob_tot

def reflection(n,pos_wall):
    c_up[n] = up_initial
    c_down[n] = down_initial
    for i in range(n):
        #print(c_up,c_down)
        c_up_new = np.zeros(len(c_up),dtype=complex)
        c_down_new = np.zeros(len(c_down),dtype=complex)
        for k in range(2*n):
            c_up_new[k] = c_down[k-1] + c_up[k-1]
            c_down_new[k] = -1*c_down[k+1] + c_up[k+1]
        #print(c_up_new,c_down_new)
        for x in range(2*n):
            if pos_wall > 0:
                if x == n+pos_wall:
                    c_up[x-2]=c_up[x-2]-c_up_new[x]
                    c_down[x-2]=c_down[x-2]-c_down_new[x]
                else:
                    c_up[x] = c_up_new[x]
                    c_down[x] = c_down_new[x]
            if pos_wall < 0:
                if x == n+pos_wall:
                    c_up[x+2]=c_up[x+2]-c_up_new[x]
                    c_down[x+2]=c_down[x+2]-c_down_new[x]
                else:
                    c_up[x] = c_up_new[x]
                    c_down[x] = c_down_new[x]
    prob_up = np.zeros(len(c_up),dtype=complex)
    prob_down = np.zeros(len(c_down),dtype=complex)
    prob_tot = np.zeros(len(c_up),dtype=complex)
    for i in range(len(c_up)):
        prob_up[i] = np.abs(c_up[i])**2
        prob_down[i] = np.abs(c_down[i])**2
        prob_tot[i] = (prob_up[i] + prob_down[i])/(2**n)
    return prob_tot

d = 2
jumprate = 1
k = 1

#make Mab matrix for formula (2) of Childs
def bintree(n,jumprate,k):
    matrix = np.zeros((2**(n+1)+2**n-2,2**(n+1)+2**n-2))
    for i in range(0,len(matrix)):
        matrix[i,i] = jumprate
        if i <= 2**n-2:
            matrix[i,2*(i+1)-1] = -k*jumprate
            matrix[i,2*(i+1)] = -k*jumprate
            matrix[2*(i+1)-1,i] = -k*jumprate
            matrix[2*(i+1),i] = -k*jumprate
            matrix[-(i+1),-2*(i+1)-1] = -k*jumprate
            matrix[-(i+1),-2*(i+1)] = -k*jumprate
            matrix[-2*(i+1)-1,-(i+1)] = -k*jumprate
            matrix[-2*(i+1),-(i+1)] = -k*jumprate   
    return matrix

print(bintree(d,jumprate,k))

d = 6
gamma = 1

def reductionmatrix(n,gamma):
    matrix = np.zeros((2*n+1,2*n+1))
    for i in range(2*n+1):
        if i == 0:
            matrix[i,i] = 2*gamma
            matrix[i,i+1] = -2*gamma
        elif i == n:
            matrix[i,i] = 2*gamma
            matrix[i,i+1] = -2*gamma
            matrix[i,i-1] = -2*gamma
        elif i == 2*n:
            matrix[i,i] = 2*gamma
            matrix[i,i-1] = -2*gamma
        else:
            matrix[i,i] = 3*gamma
            matrix[i,i+1] = -2*gamma
            matrix[i,i-1] = -2*gamma
    return matrix

print(reductionmatrix(d,gamma))

distr = qrw(n)
ptot = 0
for i in range(2*n):
    ptot = ptot+distr[i]

print(ptot)
plt.plot(positions,distr[::2])
print(len(positions),len(distr[::2]))

plt.plot(positions,absorption(n,pos_wall)[::2])
print(len(positions),len(absorption(n,pos_wall)[::2]))

plt.plot(positions,reflection(n,pos_wall)[::2])
print(len(positions),len(reflection(n,pos_wall)[::2]))
