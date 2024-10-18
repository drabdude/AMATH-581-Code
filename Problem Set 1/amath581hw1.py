import numpy as np
def f(x):
    return x*np.sin(3*x)-np.exp(x)
def fder(x):
    return 3*x*np.cos(3*x)+np.sin(3*x) - np.exp(x)

xvalsnr = np.array([-1.6])
for i in range(1000):
    xvalsnr = np.append(xvalsnr,xvalsnr[i] - f(xvalsnr[i])/fder(xvalsnr[i]))
    fnew = f(xvalsnr[-1])
    if abs(fnew)<1e-10:
        break
A1 = xvalsnr
print(f(xvalsnr[-1]))
xl = -0.7
xr = -0.4
xvalsbis = np.array([])
for j in range(1000):
    xc = (xr + xl)/2
    xvalsbis = np.append(xvalsbis,xc)
    fnew = f(xc)
    if fnew > 0:
        xl = xc 
    else:
        xr = xc
    if abs(fnew) < 1e-6: 
        break

A2 = xvalsbis
A3 = np.array([len(xvalsnr)-1,len(xvalsbis)])
print(A1)
print(A3)
A = np.array([[1,2],[-1,1]])
B = np.array([[2,0],[0,2]])
C = np.array([[2,0,-3],[0,0,-1]])
D = np.array([[1,2],[2,3],[-1,0]])
x = np.array([[1],[0]])
y = np.array([[0],[1]])
z = np.array([[1],[2],[-1]])

A4 = A+B
A5 = (3*x-4*y)
A5 = A5.reshape(len(A5),)
A6 = A@x
print(A6)
A6 = A6.reshape(len(A6),)
A7 = B@(x-y)
A7 = A7.reshape(len(A7),)
A8 = D@x
A8 = A8.reshape(len(A8),)
A9 = D@y+z
A9 = A9.reshape(len(A9),)
A10 = A@B
A11 = B@C
A12 = C@D


