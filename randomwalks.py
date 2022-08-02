from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import curve_fit
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
           
def DecToBin(n):
    return bin(n).replace("0b","")

def hammingWeight(n):
      """
      :type n: int
      :rtype: int
      """
      x = DecToBin(n)
      #print(x)
      one_count = 0
      for i in x:
         if i == "1":
            one_count+=1
      return one_count

num = 939
#print(hammingWeight(num))

#make matrix for an n-hypercub
def qhypercube(n,jumprate):
    vertices = 2**n   #number of vertices
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
        matrix[i,i] = n*jumprate
#    for i in range(0,len(matrix)):
#        matrix[i,-1] = 0
    matrix[-1,-1] =  - 1j*n
    return matrix

#make matrix for an n-hypercub
def chypercube(n,jumprate):
    vertices = 2**n   #number of vertices
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
        matrix[i,i] = n*jumprate
    for i in range(0,len(matrix)):
        matrix[i,-1] = 0
#    matrix[-1,-1] = - 1j*n*jumprate
    return matrix

steps = 60000
t_max = 3000



t = np.linspace(0,t_max,steps)

def q_hitting_time(dim,steps,t_max,jumprate):
    
    M = qhypercube(dim,jumprate)
    vertices = 2**dim   #number of vertices
    dt = t_max/steps
    p0= np.zeros(vertices,dtype=complex)
    p0[0] = 1                        #starts at left vertex
    #print(p0)

    prob = np.zeros((vertices,len(t)), dtype = complex)
    test = np.zeros((vertices,vertices))
    
    test = expm(-M * dt * 1j)
    prob[:, 0] = p0

    for j in range(1,len(t)):
        prob[:, j] = np.dot(test , prob[:,j-1])
    


    p = np.zeros((len(t),vertices),dtype=complex)
    for i in range(vertices):
        p[:,i] = np.abs(prob[i])**2

    hitting_time = 0                                
    for i in range(len(t)):
        hitting_time += sum(p[i,:-1]).real*dt  
    print("t_max: ",t_max," steps: ", steps," dt: ",dt)
    print("Hitting time: ",hitting_time)
    print(" ")
        
        
        #$print(p)
    p_new = p
    #which vertex are we looking at
    vertex = -1
    p_vertex = np.zeros(len(p),dtype=complex)
    
    
    for i in range(len(p_vertex)):
        p_vertex[i] = p[i,vertex]
        
    
    p_threshold = 0.1
    time = 0
    
    for i in p_vertex:
        if i >= p_threshold:
            time = np.where(p_vertex == i)
#            print("P reached at t =",t[time])
            break
        
    p_vertex = np.zeros(len(p),dtype=complex)
    
    p_t = np.zeros(vertices,dtype=complex)
    
    
    vertices = np.arange(0,vertices,dtype=complex)
    
    for i in range(len(p_t)):
        for x in range(len(p_vertex)):
            p_vertex[x] = p[x,i]
        p_t[i] = p_vertex[-3]
    
    
    
    #adding the values for all vertices in the same columns
    p_column = np.zeros(dim+1,dtype=complex)
    columns = np.arange(0,dim+1)
    
    for i in range(len(p_t)):
        p_column[hammingWeight(i)] = p_column[hammingWeight(i)] + p_t[i]
    
    
    
    plt.xlabel("column")
    plt.ylabel("Probability particle is in the column")
    plt.plot(columns,p_column, label = "after t = "+str(t[-1])+" with dt = "+str(dt))
    plt.legend(loc="upper left")
    plt.show()
    return hitting_time






def c_hitting_time(dim,steps,t_max,jumprate):
    M = chypercube(dim,jumprate)
    vertices = 2**dim   #number of vertices
    dt = t_max/steps
    
    p0= np.zeros(vertices,dtype=complex)
    p0[0] = 1                        #starts at left vertex
    #print(p0)
    
    
    prob = np.zeros((vertices,len(t)), dtype = complex)
    prob[:, 0] = p0
    
    test = np.zeros((vertices,vertices))
    test = expm(-M * dt)
    
    for j in range(1,len(t)):
        prob[:, j] = np.dot(test , prob[:, j-1])
        
        
    p = np.zeros((len(t),vertices),dtype=complex)
    for i in range(len(t)):
        for j in range(vertices):
            p[i,j] = prob[j,i]    
                     
    hitting_time = 0                                
    for i in range(len(t)):
        hitting_time += sum(p[i,:-1]).real*dt  
    print("t_max: ",t_max," steps: ", steps," dt: ",dt)
    print("Hitting time: ",hitting_time)
    print(" ")
    
    #$print(p)
    p_new = p
    #which vertex are we looking at
    vertex = -1
    p_vertex = np.zeros(len(p),dtype=complex)
    
    
    for i in range(len(p_vertex)):
        p_vertex[i] = p[i,vertex]
        
    
    p_threshold = 0.1
    time = 0
    
    for i in p_vertex:
        if i >= p_threshold:
            time = np.where(p_vertex == i)
#            print("P reached at t =",t[time])
            break
        
    p_vertex = np.zeros(len(p),dtype=complex)
    
    p_t = np.zeros(vertices,dtype=complex)
    
    
    vertices = np.arange(0,vertices,dtype=complex)
    
    for i in range(len(p_t)):
        for x in range(len(p_vertex)):
            p_vertex[x] = p[x,i]
        p_t[i] = p_vertex[-3]
    
    
    
    #adding the values for all vertices in the same columns
    p_column = np.zeros(dim+1,dtype=complex)
    columns = np.arange(0,dim+1)
    
    for i in range(len(p_t)):
        p_column[hammingWeight(i)] = p_column[hammingWeight(i)] + p_t[i]
    
    
    
    plt.xlabel("column")
    plt.ylabel("Probability particle is in the column")
    plt.plot(columns,p_column, label = "after t = "+str(t[-1])+" with dt = "+str(dt))
    plt.legend(loc="upper left")
    plt.show()
    return hitting_time
    

def polynom(x,p,a,b,):
    y = a*x**p+b
    return y

def expon(x,p,a,b,):
    y = a*p**x+b
    return y



q1 = q_hitting_time(1,steps,t_max,jumprate)
q2 = q_hitting_time(2,steps,t_max,jumprate)
q3 = q_hitting_time(3,steps,t_max,jumprate)
q4 = q_hitting_time(4,steps,t_max,jumprate)
q5 = q_hitting_time(5,steps,t_max,jumprate)
q6 = q_hitting_time(6,steps,t_max,jumprate)
q7 = q_hitting_time(7,steps,t_max,jumprate)
q8 = q_hitting_time(8,steps,t_max,jumprate)
q9 = q_hitting_time(9,steps,t_max,jumprate)
q10 = q_hitting_time(10,steps,t_max,jumprate)
q11 = q_hitting_time(11,steps,t_max,jumprate)
q12 = q_hitting_time(12,steps,t_max,jumprate)
#q13 = q_hitting_time(13,steps,t_max,jumprate)

c1 = c_hitting_time(1,steps,t_max,jumprate)
c2 = c_hitting_time(2,steps,t_max,jumprate)
c3 = c_hitting_time(3,steps,t_max,jumprate)
c4 = c_hitting_time(4,steps,t_max,jumprate)
c5 = c_hitting_time(5,steps,t_max,jumprate)
c6 = c_hitting_time(6,steps,t_max,jumprate)
c7 = c_hitting_time(7,steps,t_max,jumprate)
c8 = c_hitting_time(8,steps,t_max,jumprate)
c9 = c_hitting_time(9,steps,t_max,jumprate)
c10 = c_hitting_time(10,steps,t_max,jumprate)
c11 = c_hitting_time(11,steps,t_max,jumprate)
c12 = c_hitting_time(12,steps,t_max,jumprate)
#c13 = c_hitting_time(13,steps,t_max,jumprate)

ctimes = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12]
qtimes = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12]

#hitting times for n = 1,2,3,4,5,6,7,8,9
#classic = [1.025246, 2.02476225, 3.358,5.35777, 8.55751, 13.890254, 23.03225, 39.03053, 67.436853, 116.994]
#quantum = [1.52485, 2.52475, 3.358, 4.08015, 4.72453, 5.3111378, 5.852751, 6.35586, 6.81537, 7.227]

#classic = np.asarray(classic)
#quantum = np.asarray(quantum)

dimensions = [1,2,3,4,5,6,7,8,9,10,11,12]
dimensions = np.asarray(dimensions)

c_param, c_cov = curve_fit(expon, dimensions, ctimes)
c_fit_p = c_param[0]
c_fit_a = c_param[1]
c_fit_b = c_param[2]

print("Exponential function for CRW: y = ",c_fit_a," * ", c_fit_p,"**x + ",c_fit_b)

c_fit_y = expon(dimensions, c_fit_p, c_fit_a, c_fit_b)

q_param, q_cov = curve_fit(polynom, dimensions, qtimes)
q_fit_p = q_param[0]
q_fit_a = q_param[1]
q_fit_b = q_param[2]

print("Polynomial function for QRW: y = ",q_fit_a," * x**",q_fit_p," + ",q_fit_b)

q_fit_y = polynom(dimensions, q_fit_p, q_fit_a, q_fit_b)

#plt.plot(dimensions,ctimes, color = 'blue')
#plt.plot(dimensions,qtimes, color = 'red')
plt.plot(dimensions,ctimes, 'o', markersize = 4, color = 'blue', label = 'CRW')
plt.plot(dimensions,c_fit_y, '-', linewidth = 1, color = 'green', label = 'exponential fit')
plt.plot(dimensions,qtimes, 'x', markersize = 4, color = 'red', label = 'QRW')
plt.plot(dimensions,q_fit_y, '-', linewidth = 1, color = 'orange', label = 'polynomial fit')
plt.legend()
plt.title("Hitting times on n-dimensional hypercubes")
plt.xlabel("n")
plt.ylabel("Hitting Time")
plt.yscale('log')
plt.show()


#plt.legend()
#plt.title("Hitting times on n-dimensional hypercubes")
#plt.xlabel("n")
#plt.ylabel("Hitting Time")
#plt.show()

def markov(n):
    
    
    expect = np.zeros(n+1)
    expect[0] = 2**n
    expect[1] = 2**n - 1
    expect[2] = n/(n-1) * (2**n - 2)
    
    for i in range(2,len(expect)-1):
        expect[i+1] = n/(n-i) * (expect[i] - 1 - i/n * expect[i-1])
    
    return expect

def expon(x,p,a,b,):
    y = a*p**x+b
    return y

ht = [markov(5)[-1],markov(6)[-1]]

until = 8

hitting_time = np.zeros(until-2)

dimensions = np.linspace(3,until,until-2)

for i in range(3, until+1):
    hitting_time[i-3] = markov(i)[-1]

c_param, c_cov = curve_fit(expon, dimensions, hitting_time)
c_fit_p = c_param[0]
c_fit_a = c_param[1]
c_fit_b = c_param[2]

print("Exponential function for CRW: y = ",c_fit_a," * ", c_fit_p,"**x + ",c_fit_b)

c_fit_y = expon(dimensions, c_fit_p, c_fit_a, c_fit_b)

plt.plot(dimensions,hitting_time, 'o', color = 'blue', label = 'CRW')
plt.plot(dimensions,c_fit_y, '-', color = 'green', label = 'exponential fit')
plt.title("Analytical hitting times of a CRW on an n-hypercube")
plt.xlabel("n")
plt.ylabel("Hitting times")
plt.show()
