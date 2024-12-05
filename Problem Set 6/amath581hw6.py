import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


#cheb stuff
def sech(x):
    return 1 / np.cosh(x)

def tanh(x):
    return np.sinh(x) / np.cosh(x)

def cheb(N):
	if N==0: 
		D = 0.; x = 1.
	else:
		n = arange(0,N+1)
		x = cos(pi*n/N).reshape(N+1,1) 
		c = (hstack(( [2.], ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = tile(x,(1,N+1))
		dX = X - X.T
		D = dot(c,1./c.T)/(dX+eye(N+1))
		D -= diag(sum(D.T,axis=0))
	return D, x.reshape(N+1)

N = 30
D, x = cheb(N)

D[N,:] = 0
D[0,:] = 0
Dxx = np.dot(D,D)/100
y = x
N2 = (N+1)*(N+1)
I = np.eye(len(Dxx))
L = kron(I,Dxx) + kron(Dxx,I)
X,Y = meshgrid(10*x,10*y)
x = x*10
y = y*10

m=1
beta=1
X2 = np.power(X,2)
Y2 = np.power(Y,2)
u = tanh(sqrt(X2 + Y2))*cos(m*angle(X+1j*Y)-sqrt(X2+Y2))
v = tanh(sqrt(X2 + Y2))*sin(m*angle(X+1j*Y)-sqrt(X2+Y2))
uv0 = np.hstack([u.reshape(N2),v.reshape(N2)])

def RD_2D(t,uv,beta,N2,L):
    u = uv[0:N2]
    v = uv[N2:]
    A = u**2 + v**2
    lam = 1-A
    omega = -beta*A
    rhsu = 0.1*np.dot(L,u) + lam*u - omega*v
    rhsv = 0.1*np.dot(L,v) + omega*u + lam*v
    rhs = np.hstack([rhsu,rhsv])
    return rhs

tspan = np.linspace(0,4,9)
chebsol = solve_ivp(RD_2D,[0,4],uv0,method = 'RK45',t_eval=tspan,args=(beta,N2,L))

A2 = chebsol.y

# ufinal = chebsol.y[0:N2,-1]
# vfinal = chebsol.y[N2:,-1]

# plt.figure(figsize=(6, 6))
# plt.contourf(10*X, 10*Y, np.reshape(u,np.shape(u)))
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Initial u")
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.contourf(X, Y, np.reshape(v,np.shape(v)))
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Initial v")
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.contourf(X, Y, np.reshape(ufinal,np.shape(u)))
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Final u")
# plt.show()

# plt.figure(figsize=(6, 6))
# plt.contourf(X, Y, np.reshape(vfinal,np.shape(v)))
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Final v")
# plt.show()


# Set up the figure and axis
# fig, ax = plt.subplots(figsize=(6, 6))
# contour = ax.contourf(u)

# Update function for the animation
# def update(frame):
#     ax.clear()
#     unew = chebsol.y[0:N2, frame] 
#     contour = ax.contourf(X,Y,np.reshape(unew,np.shape(u)))
#     ax.set_title(f"Time Step: {frame}")
#     return contour

# # Create the animation
# anim = FuncAnimation(fig, update, frames=len(tspan), interval=10)


#anim.save("vorticity_animation.gif", writer="pillow", fps=10)

# Show the animation

# plt.show()



tspan = np.arange(0,4.5,0.5)
Lx, Ly = 20, 20
nx, ny = 64, 64
Nfft = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
#x2 = np.linspace(-Lx/2,Lx/2,nx)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
#y2 = np.linspace(-Ly/2,Ly/2,ny)
y = y2[:ny]
Xf, Yf = np.meshgrid(x, y)
# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
#kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
#ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

X2 = np.power(Xf,2)
Y2 = np.power(Yf,2)
uf = tanh(sqrt(X2 + Y2))*cos(m*angle(Xf+1j*Yf)-sqrt(X2+Y2))
vf = tanh(sqrt(X2 + Y2))*sin(m*angle(Xf+1j*Yf)-sqrt(X2+Y2))

ufft = fft2(uf)
vfft = fft2(vf)

# ufft = np.concatenate((ufft,ufft[0,:].reshape(1,nx-1)),0)
# ufft = np.concatenate((ufft,ufft[:,0].reshape(ny,1)),1)
# vfft = np.concatenate((vfft,vfft[0,:].reshape(1,nx-1)),0)
# vfft = np.concatenate((vfft,vfft[:,0].reshape(ny,1)),1)

uv0fft = np.hstack([ufft.reshape(nx**2),vfft.reshape(ny**2)])

def RD_2Dfft2(t, uv_fft, beta, nx, ny, K2):
    # Split u and v in Fourier domain
    u_fft = uv_fft[:Nfft].reshape((nx, ny))
    v_fft = uv_fft[Nfft:].reshape((nx, ny))
    
    # Transform back to physical space
    u = ifft2(u_fft)
    v = ifft2(v_fft)

    # Compute nonlinear terms in physical space
    A2 = u**2 + v**2
    lam = 1 - A2
    omega = -beta * A2
    
    # Compute the right-hand side in physical space
    rhsu_phys = lam * u - omega * v
    rhsv_phys = omega * u + lam * v
    
    # Transform the RHS back to Fourier space
    rhsu_fft = fft2(rhsu_phys)
    rhsv_fft = fft2(rhsv_phys)
    
    # Add diffusion terms in Fourier space
    rhsu_fft += -0.1 * K2 * u_fft
    rhsv_fft += -0.1 * K2 * v_fft
    
    # Flatten and return
    rhsu_fft_flat = rhsu_fft.flatten()
    rhsv_fft_flat = rhsv_fft.flatten()
    rhs_fft = np.hstack([rhsu_fft_flat, rhsv_fft_flat])
    return rhs_fft

#tspan = np.linspace(0,50,101)
fftsol = solve_ivp(RD_2Dfft2,[0,4],uv0fft,method = 'RK45',t_eval=tspan,args=(beta,nx,ny,K2))

A1 = fftsol.y


N = nx
X = Xf
Y = Yf
ufinal = ifft2(fftsol.y[0:N**2,-1].reshape(N,N)).real
vfinal = ifft2(fftsol.y[N**2:,-1].reshape(N,N)).real

plt.figure(figsize=(6, 6))
plt.contourf(10*X, 10*Y, np.reshape(uf,(N,N)))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Initial u")
plt.show()

plt.figure(figsize=(6, 6))
plt.contourf(X, Y, np.reshape(vf,(N,N)))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Initial v")
plt.show()

plt.figure(figsize=(6, 6))
plt.contourf(X, Y, np.reshape(ufinal,(N,N)))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Final u")
plt.show()

plt.figure(figsize=(6, 6))
plt.contourf(X, Y, np.reshape(vfinal,(N,N)))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Final v")
plt.show()


# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
contour = ax.contourf(u)

# Update function for the animation
def update(frame):
    ax.clear()
    unew = ifft2(fftsol.y[0:N**2,frame].reshape(N,N)).real
    contour = ax.contourf(X,Y,np.reshape(unew,(N,N)))
    ax.set_title(f"Time Step: {frame}")
    return contour

# Create the animation
anim = FuncAnimation(fig, update, frames=len(tspan), interval=100)


#anim.save("vorticity_animation.gif", writer="pillow", fps=10)

# Show the animation

plt.show()
