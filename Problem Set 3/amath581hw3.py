import numpy as np
from numpy import linalg
from scipy.linalg import eig
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.special import hermite

def factorial(n):
   result = 1
   for i in range(1, n + 1):
       result *= i
   return result

def hermpoly(x,n):
    return 1/(np.sqrt(2**n*factorial(n)))*((1/np.pi))**(1/4)*np.exp(-(x**2)/2)*hermite(n)(x)

def harmonic(x, y, K, beta,gamma):
    y1, y2 = y
    return [y2, (gamma*y1**2 + K*x**2-beta) * y1]


def shootmethod(L,dx,K,gamma,beta_start,tol,num,plot=False):
    N = int(2*L/dx)+1
    x = np.linspace(-L,L,N)
    Afunc = np.zeros((len(x),num))
    Avals = np.zeros(num)

    for modes in range(1, num+1): # begin mode loop
        beta = beta_start # initial value of eigenvalue beta
        dbeta = 1 # default step size in beta
        for _ in range(1000): # begin convergence loop for beta
            yinit = [0.5, np.sqrt(K*L**2-beta)/2]
            #y = odeint(harmonic, yinit, x, args=(K,beta,gamma),tfirst = True)
            sol = solve_ivp(harmonic,t_span = (-L,L),y0 = yinit ,args=(K,beta,gamma),dense_output=True)
            y = sol.sol(x).T
            if abs(np.sqrt(K*L**2-beta)*y[-1,0] + y[-1,1]) < tol: # check for convergence
                #print(beta) # write out eigenvalue
                break # get out of convergence loop
            if (-1) ** (modes + 1) * y[-1, 0] > (-1) ** (modes) * (y[-1,1]/np.sqrt(K*L**2-beta)):
                beta += dbeta
            else:
                beta -= dbeta / 2
                dbeta /= 2
        Avals[modes-1] = beta
        beta_start = beta + 1 # after finding eigenvalue, pick new start
        norm = np.trapz(y[:, 0] * y[:, 0], x)
        Afunc[:,modes-1] = abs(y[:,0]/np.sqrt(norm))
        if plot:
            plt.plot(x, Afunc[:,modes-1]) # plot modes
            plt.show()
    return Afunc,Avals

def shootmethodnonlin(L,dx,K,gamma,beta_start,tol,num,A_start,plot=False):
    N = int(2*L/dx)+1
    x = np.linspace(-L,L,N)
    Afunc = np.zeros((len(x),num))
    Avals = np.zeros(num)
    for modes in range(1, num+1): # begin mode loop
        A = A_start
        dA = 0.1
        beta = beta_start # initial value of eigenvalue beta
        dbeta = 0.5 # default step size in beta
        for _ in range(1000):
            yinit = [A,A*np.sqrt(K*L**2-beta)]
            beta = beta_start
            dbeta = 1
            for _ in range(1000): # begin convergence loop for beta
                
                #y = odeint(harmonic, yinit, x, args=(K,beta,gamma),tfirst = True)
                sol = solve_ivp(harmonic,t_span = (-L,L),y0 = yinit ,args=(K,beta,gamma),dense_output=True)
                y = sol.sol(x).T
                if np.abs(np.sqrt(K*L**2-beta)*y[-1,0] + y[-1,1]) < tol: # check for convergence
                    break # get out of convergence loop
                if (-1) ** (modes + 1) * y[-1, 0] > (-1) ** (modes) * (y[-1,1]/(np.sqrt(K*L**2-beta))):
                    beta += dbeta/2
                    dbeta /= 2
                else:
                    beta -= dbeta
            area = np.trapz(np.abs(y[:,0]),x)
            if np.abs(area-1)<=tol:
                print(area)
                print(beta)
                print(A)
                break
            if area<1:
                A += dA
                #dA = dA/2
            else:
                A -= dA/2
                dA = dA/2
        Avals[modes-1] = beta
        A_start = A
        beta_start = beta + 1 # after finding eigenvalue, pick new start
        norm = np.trapz(y[:, 0] * y[:, 0], x)
        Afunc[:,modes-1] = abs(y[:,0]/np.sqrt(norm))
        if plot:
            plt.plot(x, Afunc[:,modes-1]) # plot modes
            plt.show()
    return Afunc,Avals




## Part a

A1, A2 = shootmethod(4,0.1,1,0,1,1e-4,5)
print(A2)

## Part b

L = 4
dx = 0.1
K = 1
N = int(2*L/dx)+1
x = np.linspace(-L,L,N)

B = np.zeros((N-2,N-2))
P = np.zeros((N-2,N-2))
for j in range(N-2):
    B[j,j] = 2
    P[j,j] = -K*x[j+1]**2
for j in range(N-3):
    B[j,j+1] = -1
    B[j+1,j] = -1
B[0,0] = 2-4/3
B[0,1] = -1+1/3
B[-1,-1] = 2-4/3
B[-1,-2] = -1+1/3
B1 = B/(dx**2)
#P[0,0] = 0
#P[-1,-1] = 0
#B1[0,0] = 0
#B1[0,1] = 0
#B1[-1,-1] = 0
#B1[-1,-2] = 0
#B1[0,0] = 3
#B1[0,1] = -4
#B1[0,2] = 1
#B1[-1,-1] = -3
#B1[-1,-2] = 4
#B1[-1,-3] = -1
#N = 79
#B1 = B1[1:-1,1:-1]
#P = P[1:-1,1:-1]

#B1[0,0] += 4/3/(dx**2)
#B1[0,1] -= 1/3/(dx**2)
#B1[-1,-1] += 4/3/(dx**2)
#B1[-1,-2] -= 1/3/(dx**2)

linL = B1 - P

D,V = eig(linL)

sorted_indices = np.argsort(np.abs(D))[::-1]
Dsort = D[sorted_indices]
Vsort = V[:,sorted_indices]

N=79
D5 = Dsort[N-5:N]
V5 = Vsort[:,N-5:N]

V5 = np.flip(V5,1)

front = []
back = []
for i in range(5):
    front.append(4/3*V5[1,i]-1/3*V5[2,i])
    back.append(4/3*V5[-1,i]-1/3*V5[-2,i])
V5 = np.vstack((np.reshape(front,(1,5)),V5,np.reshape(back,(1,5))))


for i in range(5):
    norm = np.trapz(V5[:, i] * V5[:, i], x)
    V5[:,i] = abs(V5[:,i]/np.sqrt(norm))

D5 = abs(np.flip(D5))

#plt.plot(V5)
#plt.show()

A3 = V5
A4 = D5
print(A4)
#print(A3)
#print(np.linalg.norm(A1-A3))


## Part c1

A5, A6 = shootmethodnonlin(2,0.1,1,0.05,1,1e-4,2,0.15,False)
print(A6)

## Part c2

A7, A8 = shootmethodnonlin(2,0.1,1,-0.05,1,1e-4,2,0.15,False)
print(A8)

## Part d 

K = 1
beta = 1
gamma = 0
TOL = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
logTOL = [-4,-5,-6,-7,-8,-9,-10]
avestepsizer45 = []
avestepsizer23 = []
avestepsizerad = []
avestepsizebdf = []

for tol in TOL:
    options = {'rtol':tol,'atol': tol}
    sol1 = solve_ivp(harmonic,[-2,2],[1,np.sqrt(3)],method='RK45',args=(K,beta,gamma),**options)
    sol2 = solve_ivp(harmonic,[-2,2],[1,np.sqrt(3)],method='RK23',args=(K,beta,gamma),**options)
    sol3 = solve_ivp(harmonic,[-2,2],[1,np.sqrt(3)],method='Radau',args=(K,beta,gamma),**options)
    sol4 = solve_ivp(harmonic,[-2,2],[1,np.sqrt(3)],method='BDF',args=(K,beta,gamma),**options)
    avestepsizer45.append(np.mean(np.diff(sol1.t)))
    avestepsizer23.append(np.mean(np.diff(sol2.t)))
    avestepsizerad.append(np.mean(np.diff(sol3.t)))
    avestepsizebdf.append(np.mean(np.diff(sol4.t)))

plt.loglog(avestepsizer45,TOL,label = "RK45")
plt.loglog(avestepsizer23,TOL,label = "RK23")
plt.loglog(avestepsizerad,TOL,label = "Radau")
plt.loglog(avestepsizebdf,TOL,label = "RBF")
plt.legend()
#plt.show()

A9 = np.vstack((np.polyfit(np.log10(avestepsizer45),logTOL,deg=1)[0],np.polyfit(np.log10(avestepsizer23),logTOL,deg=1)[0],np.polyfit(np.log10(avestepsizerad),logTOL,deg=1)[0],np.polyfit(np.log10(avestepsizebdf),logTOL,deg=1)[0])).reshape(4,)
print(A9)


## Part e

L = 4
dx = 0.1
K = 1
N = int(2*L/dx)+1
x = np.linspace(-L,L,N)

trueeigvals = [1,3,5,7,9]
trueeigfuncs = [hermpoly(x,1),hermpoly(x,2),hermpoly(x,3),hermpoly(x,4),hermpoly(x,5)]

A10 = [np.trapz((A1[:,0]-np.abs(hermpoly(x,0)))**2),np.trapz((A1[:,1]-np.abs(hermpoly(x,1)))**2),np.trapz((A1[:,2]-np.abs(hermpoly(x,2)))**2),np.trapz((A1[:,3]-np.abs(hermpoly(x,3)))**2),np.trapz((A1[:,4]-np.abs(hermpoly(x,4)))**2)]
A11 = [100*(np.abs(A2[0]-1)/1),100*(np.abs(A2[1]-3)/3),100*(np.abs(A2[2]-5)/5),100*(np.abs(A2[3]-7)/7),100*(np.abs(A2[4]-9)/9)]
A12 = [np.trapz((A3[:,0]-np.abs(hermpoly(x,0)))**2),np.trapz((A3[:,1]-np.abs(hermpoly(x,1)))**2),np.trapz((A3[:,2]-np.abs(hermpoly(x,2)))**2),np.trapz((A3[:,3]-np.abs(hermpoly(x,3)))**2),np.trapz((A3[:,4]-np.abs(hermpoly(x,4)))**2)]
A13 = [100*(np.abs(A4[0]-1)/1),100*(np.abs(A4[1]-3)/3),100*(np.abs(A4[2]-5)/5),100*(np.abs(A4[3]-7)/7),100*(np.abs(A4[4]-9)/9)]

print(A10)
print(A11)
print(A12)
print(A13)