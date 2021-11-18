#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:34:48 2021

@author: cantin
"""

import numpy as np
import numpy.linalg as npl
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.sparse import diags
#  Discrétisation en espace

xmin = 0.0
xmax = 1
nptx = 6
nx = nptx-2
hx = (xmax-xmin)/(nptx - 1)
xx = np.linspace(xmin, xmax, nptx)
xx = xx.transpose()

xxint = xx[1:nx+1]
ymin = 0.0
ymax = 1
npty = 6
ny = npty-2
hy = (ymax-ymin)/(npty - 1)
yy = np.linspace(ymin, ymax, npty)
yy = yy.transpose()
yyint = yy[1:ny+1]


k = np.array([-1*np.ones(nx+1), 2*np.ones(nptx), -
              1*np.ones(nx+1)], dtype=object)
offset = [-1, 0, 1]
aux = (1/(hx)**2) * diags(k, offset).toarray()
aux[0, 0] = 1
aux[0, 1] = 0
aux[-1, -2] = 0
aux[-1, -1] = 1
#  Matrix system
# On Ox
Kx = np.copy(aux)  # Local matrix of size Nx+2 relative to Ox discretization
aux = np.eye(nptx*npty)
aux[nptx:- nptx, nptx:-nptx] = np.kron(np.eye(ny), Kx)


K2Dx = np.copy(aux)  # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2

k = np.array([-1*np.ones(ny+1), 2*np.ones(npty), -
              1*np.ones(ny+1)], dtype=object)

aux2 = (1/(hy)**2) * diags(k, offset).toarray()
aux2[0, 0] = 0
aux2[0, 1] = 0


aux2[-1, -2] = 0
aux2[-1, -1] = 0
# On Oy
Ky = np.copy(aux2)  # Local matrix of size Nx+2 relative to Oy discretization
aux = np.zeros((nptx*npty, nptx*npty))


# helpi = np.eye(nptx*npty)
# helpi[0:npty,0:npty]= np.zeros((npty,npty))
# hn = len(helpi)
# helpi[hn -npty:,hn -npty:]= np.zeros(npty,npty)
aux3 = np.eye(nptx)
aux3[0, 0] = 0
aux3[-1, -1] = 0

aux = np.kron(Ky, aux3)

K2Dy = np.copy(aux)  # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
#
#
K2 = np.zeros((nptx*npty, nptx*npty))
K2 = K2Dx + K2Dy  # Final matrix of Laplacien operator with Dirichlet Boundary conditions


#
# Solution and source terms
u = np.zeros((nx+2)*(ny+2))  # Numerical solution
u_ex = np.zeros((nx+2)*(ny+2))  # Exact solution
F = np.zeros((nx+2)*(ny+2))  # Source term


def Source_int(x):
    return 2*np.pi**2*(np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))


def Source_bnd(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])


def Sol_sin(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])


for i in range(nptx):
    for j in range(npty):
        coord = np.array([i*hx, j*hy])
        u_ex[j*(nx+2) + i] = Sol_sin(coord)
    if i == 0 or i == nptx-1:  # Boundary x=0 ou x=xmax
        for j in range(npty):
            coord = np.array([i*hx, j*hy])
            F[j*(nx+2) + i] = Source_bnd(coord)
    else:
        for j in range(npty):
            coord = np.array([i*hx, j*hy])
            if j == 0 or j == npty-1:  # Boundary y=0 ou y=ymax
                F[j*(nx+2) + i] = Source_bnd(coord)
            else:
                F[j*(nx+2) + i] = Source_int(coord)
#
#
#
# Post-traintement u_ex+Visualization of the exct solution
u = npl.solve(K2, F)
uu = np.reshape(u, (nx+2, ny+2), order='F')
uu_ex = np.reshape(u_ex, (nx+2, ny+2), order='F')
X, Y = np.meshgrid(xx, yy)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, uu_ex.T, rstride=1, cstride=1, label="SOl lui")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, uu.T, rstride=1, cstride=1, label="SOl nous")
# ax.plot_surface(X,Y,np.dot(K2,u),rstride = 1, cstride = 1)
plt.show()
plt.plot(u-u_ex, label="erreur")
plt.legend()
plt.show()
print("\n Erreur      :  ", npl.norm(u-u_ex))


def Source_bnd_N(x):
    return 0


def Sol_sin_N(x, a=xmax, b=ymax):
    return np.sin(np.pi*x[1]/b)*(np.cos(np.pi*x[0]/a) - 1)


def Source_int_N(x, a=xmax, b=ymax):
    return np.pi ** 2 * np.sin(np.pi*x[1]/b)*((1/(b**2)) * np.cos(np.pi*x[0]/a) + (1/(a**2)) * (np.cos(np.pi*x[0]/a) - 1))


F_N = np.zeros((nx+2)*(ny+2))
for i in range(nptx):
    for j in range(npty):
        coord = np.array([i*hx, j*hy])
        u_ex[j*(nx+2) + i] = Sol_sin_N(coord)
    if i == 0 or i == nptx-1:  # Boundary x=0 ou x=xmax
        for j in range(npty):
            coord = np.array([i*hx, j*hy])
            F_N[j*(nx+2) + i] = Source_bnd_N(coord)
    else:
        for j in range(npty):
            coord = np.array([i*hx, j*hy])
            if j == 0 or j == npty-1:  # Boundary y=0 ou y=ymax
                F_N[j*(nx+2) + i] = Source_bnd_N(coord)
            else:
                F_N[j*(nx+2) + i] = Source_int_N(coord)


k = np.array([-1*np.ones(nx+1), 2*np.ones(nptx), -
              1*np.ones(nx+1)], dtype=object)
offset = [-1, 0, 1]
aux = (1/(hx)**2) * diags(k, offset).toarray()
aux[0, 0] = 1
aux[0, 1] = 0
aux[-1, -1] = 3/2/hx
aux[-1, -3] = 0.5/hx
aux[-1, -2] = -2/(hx)

#  Matrix system
# On Ox
# Local matrix of size Nx+2 relative to Ox discretization
Kx_neum = np.copy(aux)
aux = np.eye(nptx*npty)
aux[nptx:- nptx, nptx:-nptx] = np.kron(np.eye(ny), Kx_neum)


# Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
K2Dx_neum = np.copy(aux)
A_neum = np.copy(K2Dx_neum + K2Dy)
u_N = npl.solve(A_neum, F_N)
uu_N = np.reshape(u_N, (nx+2, ny+2), order='F')
uu_ex = np.reshape(u_ex, (nx+2, ny+2), order='F')
X, Y = np.meshgrid(xx, yy)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, uu_ex.T, rstride=1, cstride=1, label="SOln lui")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, uu_N.T, rstride=1, cstride=1, label="SOln nous")
# ax.plot_surface(X,Y,np.dot(K2,u),rstride = 1, cstride = 1)
plt.show()
plt.plot(u_N-u_ex, label="erreur N")
plt.legend()
plt.show()
print("\n Erreur Neum :  ", npl.norm(u_N-u_ex))


V = 0.01 #diffusif  change pas trop advectif change beaucoup


def Source_bnd_D(x):
    return 0


def Sol_sin_D(x, a=xmax, b=ymax):
    return np.sin(np.pi*x[1])*(np.cos(np.pi*x[0]) - 1)


def Source_int_D(x,a=xmax,b=xmax):
    return (np.pi/a)**2*np.sin(np.pi*x[1]/b)*np.cos(np.pi*x[0]/a) + (np.pi/b)**2*np.sin(np.pi*x[1]/b)*(np.cos(np.pi*x[0]/a)-1) - V/2/hx* (np.pi/a)*np.sin(np.pi*x[0]/a)*np.sin(np.pi*x[1]/b)
#def Source_int_D(x, a=xmax , b=ymax):
#return np.pi**2*np.sin(np.pi*x[1])*(np.cos(np.pi*x[0])+np.cos(np.pi*x[0])-1)- \
#        V*(np.pi)*np.sin(np.pi*x[1])*np.sin(np.pi*x[0])
#def Source_int_D(x, a=xmax, b=ymax):
#    return np.pi ** 2 * np.sin(np.pi*x[1])*(np.cos(np.pi*x[0])-1) + (np.cos(np.pi*x[0]) - 1)) - V * (np.pi * np.sin(np.pi* x[1]) * np.sin(np.pi*x[0])

k=np.array([-1*np.ones(nx+1), np.zeros(nptx), 1*np.ones(nx+1)], dtype=object)
offset=[-1, 0, 1]
aux=(V/2/hx) * diags(k, offset).toarray()
aux[0,:] = 0
aux[-1,:] = 0
aux3=np.eye(npty)
aux3[0, 0]=0
aux3[-1, -1]=0
#  Matrix system
# On Ox
# Local matrix of size Nx+2 relative to Ox discretization
K2x_dif= np.kron(aux3, aux)

F_D=np.zeros((nx+2)*(ny+2))
for i in range(nptx):
    for j in range(npty):
        coord = np.array([i*hx, j*hy])
        u_ex[j*(nx+2) + i] = Sol_sin_D(coord)
    if i == 0 or i == nptx-1:  # Boundary x=0 ou x=xmax
        for j in range(npty):
            coord=np.array([i*hx, j*hy])
            F_D[j*(nx+2) + i]=Source_bnd_D(coord)
    else:
        for j in range(npty):
            coord=np.array([i*hx, j*hy])
            if j == 0 or j == npty-1:  # Boundary y=0 ou y=ymax
                F_D[j*(nx+2) + i]=Source_bnd_D(coord)
            else:
                F_D[j*(nx+2) + i]=Source_int_D(coord)

# Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
A_dif=np.copy(K2x_dif + A_neum)

u_D = npl.solve(A_dif , F_D)
uu_D = np.reshape(u_D, (nx+2, ny+2), order='F')
uu_ex = np.reshape(u_ex, (nx+2, ny+2), order='F')
X, Y = np.meshgrid(xx, yy)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, uu_ex.T, rstride=1, cstride=1, label="SOln lui")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, uu_D.T, rstride=1, cstride=1, label="SOln nous")

u_dif=npl.solve(A_dif, F_D)
plt.show()
#plt.plot(u_dif, label="dif")
#plt.plot(u_ex, label="exa")
#plt.legend()
# plt.plot(u_ex-u_dif,label="erreur_dif")
print("\n Erreur neumann dif  :  ", npl.norm(u_dif-u_ex))
#print("\n Erreur neumann dif  :  ", npl.norm(uu_ex-uu_D))

# =============================================================================
### Parameters
mu = 0.01 # Diffusion parameter
vx = 1 # Vitesse along x
# =============================================================================
# =============================================================================
# Time stepping
# =============================================================================
A2 = -(K2 + K2x_dif)
cfl = 0.5 # cfl =mu*dt/hx^2+mu*dt/hy^2 ou v*dt/h
dt = (hx**2)*(hy**2)*cfl/(mu*(hx**2 + hy**2)) # dt = pas de temps
#dt = cfl*hx/vx
Tfinal = 1.5  # Temps final souhaitÃ©
s0 = 0.1
x0 = 0.25
y0=0.5

def Sol_init(x):
    return np.exp( -((x[0]-x0)/s0)**2 -((x[1]-y0)/s0)**2   )



u_init = np.zeros((nx+2)*(ny+2))
for i in range(nptx):
     for j in range(npty):
             coord = np.array([xmin+i*hx,ymin+j*hy])
             u_init[j*(nx+2) + i] = Sol_init(coord)
             
             
uu_init = np.reshape(u_init,(nx+2 ,ny+2),order = 'F');
fig = plt.figure(figsize=(10, 7))
X,Y = np.meshgrid(xx,yy)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, uu_init.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.view_init(60, 35)
plt.pause(1.)
             
## Initialize u by the initial data u0
u = u_init.copy()

# Nombre de pas de temps effectues
nt = int(Tfinal/dt)
Tfinal = nt*dt # on corrige le temps final (si Tfinal/dt n'est pas entier)

# Time loop
for n in range(1,nt+1):

  # Schéma explicite en temps


 # Print solution
    if n%5 == 0:
      plt.figure(1)
      plt.clf()
      fig = plt.figure(figsize=(10, 7))
      ax = plt.axes(projection='3d')
      uu = np.reshape(u,(nx+2 ,ny+2),order = 'F');
      surf = ax.plot_surface(X, Y, uu.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
      ax.view_init(60, 35)
      plt.title(['Schema explicite avec CFL=%s' %(cfl), '$t=$%s' %(n*dt)])
      plt.pause(0.1)

####################################################################
# comparaison solution exacte avec solution numerique au temps final
j0 = int((npty-1)/2)


plt.figure(2)
plt.clf()
x = np.linspace(xmin,xmax,nptx)
plt.plot(x,uu_init[:,j0],x,uu[:,j0],'k') #,x,uexacte,'or')
plt.legend(['Solution initiale','Schema explicite =%s' %(cfl)]) #,'solution exacte'],loc='best')
plt.show()
