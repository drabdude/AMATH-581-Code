import numpy as np
from scipy.linalg import eig
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
L = 4
K = 1
#dx = 0.1
#xdisc = np.arange(-L, L, dx)
#N = len(xdisc)

N = 81
xdisc = np.linspace(-L,L,N+2)
dx = xdisc[1]-xdisc[0]
print(dx)


B = np.zeros((N,N))
P = np.zeros((N,N))
for j in range(N):
    B[j,j] = -2
    P[j,j] =K*xdisc[j+1]**2
for j in range(N-1):
    B[j,j+1] = 1
    B[j+1,j] = 1
B1 = B/(dx**2)

linL = -B1 + P

D,V = eig(linL)

sorted_indices = np.argsort(np.abs(D))[::-1]
Dsort = D[sorted_indices]
Vsort = V[:,sorted_indices]

D5 = Dsort[N-5:N]
V5 = Vsort[:,N-5:N]

for i in range(5):
    curr_evec = V5[:,i]
    V5[:,i] = normalize(curr_evec.reshape(-1,1),axis=0).reshape(N,)
D5 = np.fliplr(D5)
V5 = np.fliplr(V5)
print(D5)
plt.plot(abs(V5))
plt.show()
