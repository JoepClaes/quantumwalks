from scipy.integrate import solve_ivp
import math
import numpy as np
import matplotlib.pyplot as plt

n=100   #number of steps in walk

#initial state: (up_initial*|up> + down_initial*|down>) tensor |pos0>
pos0 = 0    #initial position
up_initial = 1/math.sqrt(2)
down_initial = 1j/math.sqrt(2)
#up_initial = 1
#down_initial = 0

positions = np.arange(-n+pos0,n+1+pos0,2)

c_up = np.zeros(2*n+1,dtype=complex)
c_down = np.zeros(2*n+1,dtype=complex)

#plots the probability distribution of a QRW based on initial conditions
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

distr = qrw(n)
check = 0
for i in range(2*n):
    check = check+distr[i]

print(check)
plt.xlabel("position")
plt.ylabel("probability)
plt.plot(positions,distr[::2])

pos_wall = 20

positions = np.arange(-n+pos0,n+1+pos0,2)

c_up = np.zeros(2*n+1,dtype=complex)
c_down = np.zeros(2*n+1,dtype=complex)

#plots the probability disctribution of a QRW with an absorbing wall at pos_wall
def absorption(n,pos_wall):
    c_up = np.zeros(2*n+1,dtype=complex)
    c_down = np.zeros(2*n+1,dtype=complex)
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


distr = absorption(n,pos_wall)
check = 0
for i in range(len(distr)):
    check = check+distr[i]

print(check)
plt.xlabel("position")
plt.ylabel("probability)
plt.plot(positions,distr[::2])

pos_wall = 20


positions = np.arange(-n+pos0,n+1+pos0,2)

c_up = np.zeros(2*n+1,dtype=complex)
c_down = np.zeros(2*n+1,dtype=complex)

#plots the probability disctribution of a QRW with a reflecting wall at pos_wall
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

distr = reflection(n,pos_wall)
check = 0
for i in range(len(distr)):
    check = check+distr[i]

print(check)
plt.xlabel("position")
plt.ylabel("probability)
plt.plot(positions,distr[::2])

n = 2

jumprate = 1
#make Mab matrix for formula (2) of Childs
def bintree(n,jumprate):
    matrix = np.zeros((2**(n+1)+2**n-2,2**(n+1)+2**n-2),dtype=complex)
    k = np.zeros(2**(n+1)+2**n-2,dtype=complex)
    for i in range(0,len(matrix)):
        if i == 0:
            k[i] = 2
        elif 2**n-1 <= i <= 2**(n+1)-2 :
            k[i] = 2
        elif i == 2**(n+1)+2**n-3:
            k[i] = 2
        else:
            k[i] = 3
        matrix[i,i] = k[i]*jumprate
        if i <= 2**n-2:
            matrix[i,2*(i+1)-1] = -jumprate
            matrix[i,2*(i+1)] = -jumprate
            matrix[2*(i+1)-1,i] = -jumprate
            matrix[2*(i+1),i] = -jumprate
            matrix[-(i+1),-2*(i+1)-1] = -jumprate
            matrix[-(i+1),-2*(i+1)] = -jumprate
            matrix[-2*(i+1)-1,-(i+1)] = -jumprate
            matrix[-2*(i+1),-(i+1)] = -jumprate   
    return matrix


t = np.linspace(0,20,100)

p0= np.zeros(2**(n+1)+2**n-2,dtype=complex)
p0[0] = 1/np.sqrt(2)
p0[-1] = 1/np.sqrt(2)                       #starts at left vertex
print(p0)        
M = bintree(n,jumprate)


#solve formula (2) or (3) of Childs
def partdiff(t,p):
    dpdt = np.zeros((2**(n+1)+2**n-2),dtype=complex)
    for a in range(len(dpdt)):
        for b in range(len(dpdt)):
            dpdt[a] = dpdt[a] + M[a,b]*p[b]*1j
    return dpdt

dp_dt = partdiff((t[0],t[-1]),p0)

prob = solve_ivp(partdiff,(t[0],t[-1]),p0,t_eval = t)
p = np.zeros((len(t),2**(n+1)+2**n-2),dtype=complex)
for i in range(len(t)):
    for j in range(2**(n+1)+2**n-2):
        p[i,j] = abs(prob.y[j,i])**2
print(p)
p_new = p
#which vertex are we looking at
vertex = 4
p_vertex = np.zeros(len(p),dtype=complex)

for i in range(len(p_vertex)):
    p_vertex[i] = p[i,vertex]

##plotting the probability to be in 1 vertex against time
plt.xlabel("time")
plt.ylabel("Probability particle is in the vertex " + str(vertex))
plt.plot(t,p_vertex)
plt.show()

p_t = np.zeros(2**(n+1)+2**n-2,dtype=complex)
vertices = np.arange(0,2**(n+1)+2**n-2,dtype=complex)

for i in range(len(p_t)):
    for x in range(len(p_vertex)):
        p_vertex[x] = p[x,i]
    p_t[i] = p_vertex[-3]

print(p_t)
plt.xlabel("vertex")
plt.ylabel("Probability particle is in the vertex")
plt.plot(vertices,p_t)
plt.show()

print(p_t)
check = sum(p_t)
print(check)

#adding the values for all vertices in the same columns
p_column = np.zeros(2*n+1,dtype=complex)
columns = np.arange(0,2*n+1)
for i in range(len(p_t)):
    for x in range(len(p_t)+1):
        if x <= 2**(n+1)-1:
            if 2**i <= x <= 2**(i+1)-1:
                p_column[i] = p_column[i] + p_t[x-1]
        if x >= 2**(n+1)-1:
            if (2**(n+1)+2**n-1)-(2**(i+1)-1) <= x <= (2**(n+1)+2**n-1)-(2**i):
                p_column[-i-1] = p_column[-i-1] + p_t[x-1]

print(p_column)
check = sum(p_column)
print(check)

plt.xlabel("column")
plt.ylabel("Probability particle is in the column")
plt.plot(columns,p_column)
plt.show()
           
           n = 5               #n-hypercube
vertices = 2**n   #number of vertices
jumprate = 1

def DecToBin(n):
    return bin(n).replace("0b","")

def hammingWeight(n):
      """
      :type n: int
      :rtype: int
      """
      x = DecToBin(n)
      print(x)
      one_count = 0
      for i in x:
         if i == "1":
            one_count+=1
      return one_count

num = 939
print(hammingWeight(num))

#make matrix for an n-hypercub
def hypercube(n,jumprate):
    matrix = np.zeros((vertices,vertices),dtype=complex)
    for i in range(0,len(matrix)):
        for  x in range(n):
            if i <= 2**(x)-1:
                matrix[2**(x)+i,i] = -jumprate
                matrix[i,2**(x)+i] = -jumprate
                matrix[-2**(x)-(i+1),-(i+1)] = -jumprate
                matrix[-(i+1),-2**(x)-(i+1)] = -jumprate
            if 2**(x-1) <= i <=2**x-1:
                for a in range(2**(x-1),2**x):
                    matrix[i,a] = matrix[i-2**(x-1),a-2**(x-1)]
                    matrix[-(i+1),-(a+1)] = matrix[i-2**(x-1),a-2**(x-1)]
    return matrix

M = hypercube(n,jumprate)
t = np.linspace(0,0.7,1000)

p0= np.zeros(vertices,dtype=complex)
p0[0] = 1                        #starts at left vertex
print(p0)


#solve formula (2) or (3) of Childs
def partdiff(t,p):
    dpdt = np.zeros((vertices),dtype=complex)
    for a in range(len(dpdt)):
        for b in range(len(dpdt)):
            dpdt[a] = dpdt[a] + M[a,b]*p[b]*1j
    return dpdt

dp_dt = partdiff((t[0],t[-1]),p0)

prob = solve_ivp(partdiff,(t[0],t[-1]),p0,t_eval = t)
p = np.zeros((len(t),vertices),dtype=complex)
for i in range(len(t)):
    for j in range(vertices):
        p[i,j] = abs(prob.y[j,i])**2
print(p)
p_new = p
#which vertex are we looking at
vertex = -1
p_vertex = np.zeros(len(p),dtype=complex)

for i in range(len(p_vertex)):
    p_vertex[i] = p[i,vertex]

##plotting the probability to be in 1 vertex against time
plt.xlabel("time")
plt.ylabel("Probability particle is in the vertex " + str(vertex))
plt.plot(t,p_vertex)
plt.show()

p_t = np.zeros(vertices,dtype=complex)
vertices = np.arange(0,vertices,dtype=complex)

for i in range(len(p_t)):
    for x in range(len(p_vertex)):
        p_vertex[x] = p[x,i]
    p_t[i] = p_vertex[-3]


print(p_t)
plt.xlabel("vertex")
plt.ylabel("Probability particle is in the vertex")
plt.plot(vertices,p_t)
plt.show()

check1 = sum(p_t)

#adding the values for all vertices in the same columns
p_column = np.zeros(n+1,dtype=complex)
columns = np.arange(0,n+1)
for i in range(len(p_t)):
    p_column[hammingWeight(i)] = p_column[hammingWeight(i)] + p_t[i]

print(p_column)
check2 = sum(p_column)

plt.xlabel("column")
plt.ylabel("Probability particle is in the column")
plt.plot(columns,p_column)
plt.show()
           
