import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from numpy import linalg
import scipy
from scipy import sparse
from scipy.sparse import linalg
import scipy.integrate
from scipy.linalg import lu 
from scipy.linalg import solve_triangular
import scipy.sparse
import scipy.sparse.linalg
import time
import os

n = 64
L = 10
h = 2*L/n
x = np.linspace(-L,L,n+1)
x = x[:n]
y = np.linspace(-L,L,n+1)
y = y[:n]
X, Y = np.meshgrid(x, y)
A = 1               # Amplitude
sigma_x = 1         # Standard deviation in the x direction
sigma_y = 1         # Standard deviation in the y direction
tspan = np.linspace(0,4,9)

kx = (2 * np.pi / (L*2)) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / (L*2)) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2
# Initial vorticity ω(x, y, 0)
#omega_initial = A * np.exp(-((X**2)/(sigma_x**2) + (Y**2) / (20 * sigma_y**2)))
omega_initial = A * np.exp(-X**2 - Y**2 / 20)

# Plot the initial vorticity
# plt.figure(figsize=(6, 6))
# plt.contourf(X, Y, omega_initial, levels=50, cmap="viridis")
# plt.colorbar(label="Initial Vorticity ω(x, y, 0)")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Initial Vorticity: Elliptical Gaussian Mound")
# plt.show()


def discretematrices(n,h):
    main_diag = -2 * np.ones(n)
    off_diag = np.ones(n)

    # Use spdiags to create the 1D Laplacian
    D2_1d = sparse.spdiags([off_diag, main_diag, off_diag], [-1, 0, 1], n, n).tolil()
    D2_1d[0, -1] = 1  # periodic boundary condition
    D2_1d[-1, 0] = 1  # periodic boundary condition
    D2_1d /= h**2

    # Identity matrix of size n
    I = sparse.eye(n)

    # 2D Laplacian operator A by Kronecker product
    Asparse = sparse.kron(D2_1d, I) + sparse.kron(I, D2_1d)

    # To display A as a dense matrix, you can convert it with .toarray()
    A = Asparse.toarray()

    # First derivative matrix in 1D with periodic boundary conditions
    off_diag = np.ones(n)
    D1_1d = sparse.spdiags([-off_diag, np.zeros(n), off_diag], [-1, 0, 1], n, n).tolil()
    D1_1d[0, -1] = -1  # periodic boundary condition
    D1_1d[-1, 0] = 1  # periodic boundary condition
    D1_1d /= (2 * h)

    # Identity matrix of size n
    I = sparse.eye(n)

    # Construct B (∂/∂x) and C (∂/∂y) using the Kronecker product
    Bsparse = sparse.kron(D1_1d, I)  # ∂/∂x operator
    Csparse = sparse.kron(I, D1_1d)  # ∂/∂y operator
    B = Bsparse.toarray()
    C = Csparse.toarray()
    
    return A,B,C

def vort(t,omega_current,psi_current,A,B,C):
    return 0.001*A@omega_current-np.multiply(B@psi_current,C@omega_current)+np.multiply(C@psi_current,B@omega_current)

def fftstream(omega,K,tol):
    return np.fft.ifft2(np.fft.fft2(-omega)/(K))

def backslashstream(omega,A,tol):
    A[0,0] = 2
    fl_omega = omega.flatten()
    fl_psi = np.linalg.solve(A,fl_omega)
    return np.reshape(fl_psi,(np.shape(omega)))

def LUstream(omega,A,tol):
    P = A[0]
    L = A[1]
    U = A[2]
    fl_omega = omega.flatten()
    y = solve_triangular(L, P@fl_omega, lower=True)
    return solve_triangular(U, y)

def BICstream(omega,A,tol):
    A[0,0] = 2
    fl_omega = omega.flatten()
    x, exit =  scipy.sparse.linalg.bicgstab(A,fl_omega,rtol=tol)
    return x

def GMRESstream(omega,A,tol):
    A[0,0] = 2
    fl_omega = omega.flatten()
    x, exit = scipy.sparse.linalg.gmres(A,fl_omega,tol=tol)
    return x

def timestepsolving(x,y,n,L,h,X,Y,tspan,omega_initial,method,K,tol=10e-6):
    omegasteps = []
    A,B,C = discretematrices(n,h)
    if method=='fft':
        streamsolver = fftstream
        inp = K
    elif method=='backslash':
        streamsolver = backslashstream
        inp = A
    elif method=='LU':
        streamsolver = LUstream
        Acopy = A.copy()
        Acopy[0,0] = 4
        P, L, U = lu(Acopy)
        inp = [P,L,U]
    elif method=='BIC':
        streamsolver = BICstream
        inp = A
        tol = 10e-6
    elif method=='GMRES':
        streamsolver = GMRESstream
        inp = A
        tol = 10e-6
    else:
        return "Invalid method given for solving the stream function. Ensure it is one of 'fft, 'backslash', 'LU', 'BIC', or 'GMRES'"
    omegasteps = np.reshape(omega_initial,(n**2,-1))
    psi_initial = streamsolver(omega_initial,inp,tol)
    fl_omega_current = np.reshape(omega_initial,(n**2,))
    fl_psi_current = np.reshape(psi_initial,(n**2,))
    for i in range(len(tspan)-1):
        print(i)
        sol = scipy.integrate.solve_ivp(vort,(tspan[i],tspan[i+1]),fl_omega_current,method = 'RK45',args=(fl_psi_current,A,B,C))
        if len(omegasteps) == 0:
            omegasteps = np.transpose(sol.y[:,-1]).reshape((n**2,-1))
            fl_omega_new = omegasteps
        else:
            omegasteps = np.hstack((omegasteps, sol.y[:,-1].reshape((n**2,-1))))
            fl_omega_new = omegasteps[:,-1]
        omega_current = np.reshape(fl_omega_new,np.shape(omega_initial))
        psi_new = streamsolver(omega_current,inp,tol)
        fl_psi_current = np.reshape(psi_new,(n**2,))
        fl_omega_current = np.reshape(omega_current,(n**2,))
    return omegasteps

def plotter(X,Y,solutions,omega_initial,method):
    plt.figure(figsize=(6, 6))
    plt.contourf(X, Y, np.reshape(solutions[:,-1],np.shape(omega_initial)))
    #plt.colorbar(label="Final Vorticity ω(x, y, 4)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Final Vorticity using" + " " + method)
    plt.show()


# start_time = time.time() # Record the start time
# fftsol = timestepsolving(x,y,n,L,h,X,Y,tspan,omega_initial,'fft',K)
# end_time = time.time() # Record the end time
# elapsed_time = end_time - start_time
# print(f"Elapsed time using fft: {elapsed_time:.2f} seconds")
# # Plot the final vorticity
# plotter(X,Y,fftsol,omega_initial,"fft")

# A1 = fftsol

# start_time = time.time() # Record the start time
# backslashsol = timestepsolving(x,y,n,L,h,X,Y,tspan,omega_initial,'backslash',K)
# end_time = time.time() # Record the end time
# elapsed_time = end_time - start_time
# print(f"Elapsed time using backslash: {elapsed_time:.2f} seconds")
# # Plot the final vorticity
# plotter(X,Y,backslashsol,omega_initial,"backslash")

# A2 = backslashsol

# start_time = time.time() # Record the start time
# LUsol = timestepsolving(x,y,n,L,h,X,Y,tspan,omega_initial,'LU',K)
# end_time = time.time() # Record the end time
# elapsed_time = end_time - start_time
# print(f"Elapsed time using LU decomp: {elapsed_time:.2f} seconds")
# # Plot the final vorticity
# #plotter(X,Y,LUsol,omega_initial,"LU Decomp")

# A3 = LUsol

# start_time = time.time() # Record the start time
# BICsol = timestepsolving(x,y,n,L,h,X,Y,tspan,omega_initial,'BIC',K)
# end_time = time.time() # Record the end time
# elapsed_time = end_time - start_time
# print(f"Elapsed time using BICGSTAB: {elapsed_time:.2f} seconds")
# # Plot the final vorticity
# #plotter(X,Y,BICsol,omega_initial,"BICGSTAB")

# start_time = time.time() # Record the start time
# GMRESsol = timestepsolving(x,y,n,L,h,X,Y,tspan,omega_initial,'GMRES',K)
# end_time = time.time() # Record the end time
# elapsed_time = end_time - start_time
# print(f"Elapsed time using GMRES: {elapsed_time:.2f} seconds")
# # Plot the final vorticity
# #plotter(X,Y,GMRESsol,omega_initial,"GMRES")




# ANIMATION TIME!!!!
tspannew = np.linspace(0,100,201)
A = 1
#sigma_x = 0.1         # Standard deviation in the x direction
#sigma_y = 3           # Standard deviation in the y direction
# Initial vorticity ω(x, y, 0)
omega_initial = A * np.exp(-X**2 - Y**2 / 20)

animsol = timestepsolving(x,y,n,L,h,X,Y,tspannew,omega_initial,'LU',K)

time_steps = len(tspannew)
n = omega_initial.shape[0]  # Spatial grid size

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
#contour = ax.contourf(bigBICsol[:,0].reshape(n,n), levels=50, cmap="bone")
contour = ax.contourf(animsol[:,0].reshape(n,n))
#cbar = fig.colorbar(contour, ax=ax)
#cbar.set_label("Vorticity ω")

# Update function for the animation
def update(frame):
    ax.clear()  # Clear the previous frame
    omega_reshaped = animsol[:, frame].reshape(n, n)  # Reshape in this frame
    #contour = ax.contourf(omega_reshaped, levels=50, cmap="bone")
    contour = ax.contourf(omega_reshaped)
    ax.set_title(f"Time Step: {frame + 1}")
    return contour

# Create the animation
anim = FuncAnimation(fig, update, frames=time_steps, interval=10)


anim.save("vorticity_animation.gif", writer="pillow", fps=10)

# Show the animation

plt.show()