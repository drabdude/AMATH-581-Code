import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
tol = 1e-10 # define a tolerance level
#col = ['r', 'b', 'g', 'c', 'm', 'k'] # eigenfunc colors
#n0 = 1; A = 1; x0 = [0, 1]
#K = 1; L = 4
#num = 5
#xshoot = np.arange(-L, L, 0.1)

L = 4
dx = 0.1
N = int(2*L/dx)+1
K = 1
y0 = [0, 1]
x = np.linspace(-L,L,N)
tol = 1e-8
num = 5


def shoot2(y, x, K, beta):
    return [y[1], (K*x**2-beta) * y[0]]

A1 = np.zeros((len(x),num))
A2 = np.zeros(num)
beta_start = 0 # beginning value of beta
for modes in range(1, num+1): # begin mode loop
    beta = beta_start # initial value of eigenvalue beta
    dbeta = 0.3 # default step size in beta
    for _ in range(1000): # begin convergence loop for beta
        y = odeint(shoot2, y0, x, args=(K,beta))
        if abs(y[-1, 0] - 0) < tol: # check for convergence
            A2[modes-1] = beta
            print(beta) # write out eigenvalue
            break # get out of convergence loop
        if (-1) ** (modes + 1) * y[-1, 0] > 0:
            beta += dbeta
        else:
            beta -= dbeta / 2
            dbeta /= 2
    beta_start = beta + 1 # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], x)
    #norm = np.trapz(y[:, 0], x) # calculate the normalization
    A1[:,modes-1] = abs(y[:,0]/np.sqrt(norm))
    #plt.plot(x, A1[:,modes-1]) # plot modes
    
    #plt.show()
print(np.shape(A1))
print(A2)

solver_correct = [1.00969336, 3.02828029, 5.04546687, 7.06285421, 9.08810517]