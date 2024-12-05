import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

m=1
beta=1
left = 0
right = 4
steps = 9
tspan = np.linspace(left,right,steps)

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


Nfft = 64
#fft initial
kx = (2 * np.pi / 20) * np.concatenate((np.arange(0, Nfft/2), np.arange(-Nfft/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / 20) * np.concatenate((np.arange(0, Nfft/2), np.arange(-Nfft/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2
x = np.linspace(-10,10,Nfft)
y = x
Xf,Yf = meshgrid(x,y)
X2 = np.power(Xf,2)
Y2 = np.power(Yf,2)
uf = tanh(sqrt(X2 + Y2))*cos(m*angle(Xf+1j*Yf)-sqrt(X2+Y2))
vf = tanh(sqrt(X2 + Y2))*sin(m*angle(Xf+1j*Yf)-sqrt(X2+Y2))
ufft = fft2(uf)
vfft = fft2(vf)
uv0fft = np.hstack([ufft.reshape(Nfft**2),vfft.reshape(Nfft**2)])



Ncheb = 30
#cheb initial
D, x = cheb(Ncheb)
D[Ncheb,:] = 0
D[0,:] = 0
Dxx = np.dot(D,D)/(20/2)**2
y = x
N2 = (Ncheb+1)*(Ncheb+1)
I = np.eye(len(Dxx))
L = kron(I,Dxx) + kron(Dxx,I)
X,Y = meshgrid(10*x,10*y)
x = x*10
y = y+10
X2 = np.power(X,2)
Y2 = np.power(Y,2)
u = tanh(sqrt(X2 + Y2))*cos(m*angle(X+1j*Y)-sqrt(X2+Y2))
v = tanh(sqrt(X2 + Y2))*sin(m*angle(X+1j*Y)-sqrt(X2+Y2))
uv0 = np.hstack([u.reshape(N2),v.reshape(N2)])

def RD_2D(t,uv,beta,N2,L):
    u = uv[0:N2]
    v = uv[N2:]
    A2 = u**2 + v**2
    lam = 1-A2
    omega = -beta*A2
    rhsu = 0.1*np.dot(L,u) + lam*u - omega*v
    rhsv = 0.1*np.dot(L,v) + omega*u - lam*v
    rhs = np.hstack([rhsu,rhsv])
    return rhs


def RD_2Dfft2(t, uv_fft, beta, Nfft, K2):
    # Split u and v in Fourier domain
    u_fft = uv_fft[:Nfft**2].reshape((Nfft, Nfft))
    v_fft = uv_fft[Nfft**2:].reshape((Nfft, Nfft))
    
    # Transform back to physical space
    u = ifft2(u_fft)
    v = ifft2(v_fft)
    
    # Compute nonlinear terms in physical space
    A2 = u**2 + v**2
    lam = 1 - A2
    omega = -beta * A2
    
    # Compute the right-hand side in physical space
    rhsu_phys = lam * u - omega * v
    rhsv_phys = omega * u - lam * v
    
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

def RD_2Dfft(t,uv,beta,Nfft,K2):
    u = uv[0:Nfft**2]
    v = uv[Nfft**2:]
    u = np.reshape(u,(Nfft,Nfft))
    v = np.reshape(v,(Nfft,Nfft))
    A2 = u**2 + v**2
    lam = 1-A2
    omega = -beta*A2
    #rhsu = -0.1*np.dot(K2_flat,u) + lam*u - omega*v
    #rhsv = -0.1*np.dot(K2_flat,v) + omega*u - lam*v

    rhsu = -0.1*K2*u + lam@u - omega@v
    rhsv = -0.1*K2*v + omega@u - lam@v

    rhsu = np.reshape(rhsu,(Nfft**2))
    rhsv = np.reshape(rhsv,(Nfft**2))
    rhs = np.hstack([rhsu,rhsv])
    return rhs



fftsol = solve_ivp(RD_2Dfft2,(left,right),uv0fft,method = 'RK45',t_eval=tspan,args=(beta,Nfft,K2))
chebsol = solve_ivp(RD_2D,(left,right),uv0,method = 'RK45',t_eval=tspan,args=(beta,N2,L))

A1 = fftsol.y
A2 = chebsol.y
print(A1)
print(A2)



def plotter(sol,N,X,Y,u,v,bool):
    if bool:
        ufinal = ifft2(sol.y[0:N**2,-1].reshape(N,N)).real
        vfinal = ifft2(sol.y[N**2:,-1].reshape(N,N)).real
    else:
         ufinal = sol.y[0:N**2,-1]
         vfinal = sol.y[N**2:,-1]


    plt.figure(figsize=(6, 6))
    plt.contourf(10*X, 10*Y, np.reshape(u,(N,N)))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Initial u")
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.contourf(X, Y, np.reshape(v,(N,N)))
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
    print(np.shape(u))
    contour = ax.contourf(u)

    # Update function for the animation
    def update(frame):
        ax.clear()
        if bool:
            unew = ifft2(sol.y[0:N**2,frame].reshape(N,N)).real
        else:
             unew = sol.y[0:N**2,frame]
        contour = ax.contourf(X,Y,np.reshape(unew,(N,N)))
        ax.set_title(f"Time Step: {frame}")
        return contour

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(tspan), interval=100)


    #anim.save("vorticity_animation.gif", writer="pillow", fps=10)

    # Show the animation

    plt.show()


plotter(chebsol,Ncheb+1,X,Y,u,v,0)

#plotter(fftsol,Nfft,Xf,Yf,uf,vf,1)