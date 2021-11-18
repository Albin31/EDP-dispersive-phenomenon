#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:34:48 2021

@author: cantin
"""

import numpy as np
import numpy.linalg as npl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.sparse import diags
#  Discrétisation en espace


xmin = 0.0; xmax = 2.0; nptx = 61; nx = nptx-2  
hx = (xmax-xmin)/(nptx -1)
xx = np.linspace(xmin,xmax,nptx) 
xx = xx.transpose()
xxint = xx[1:nx+1]
ymin = 0.0; ymax = 1.0; npty = 31; ny = npty-2 
hy = (ymax-ymin)/(npty -1)
yy = np.linspace(ymin,ymax,npty)
yy=yy.transpose() 
yyint = yy[1:ny+1]


# =============================================================================
### Parameters
mu = 0.01 # Diffusion parameter
vx = 1 # Vitesse along x
# =============================================================================

cfl = 0.5
dt = (hx**2)*(hy**2)*cfl/(mu*(hx**2 + hy**2)) # dt = pas de temps
#•dt = cfl*hx/vx
#cfl =mu*dt/hx**2+mu*dt/hy**2 #ou v*dt/h #cfl = 1
Tfinal = 0.8 # Temps final souhaitÃ©



###### Matrice de Diffusion Dir/Neumann

k = np.array([-1*np.ones(nx+1), 2*np.ones(nptx), -
              1*np.ones(nx+1)], dtype=object)
offset = [-1, 0, 1]
aux = (mu/(hx)**2) * diags(k, offset).toarray()
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

aux = np.kron(Ky*mu, aux3)

K2Dy = np.copy(aux)  # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
#
#
K2 = np.zeros((nptx*npty, nptx*npty))
K2 = K2Dx + K2Dy 
for i in range(2,npty):
    K2[i*nptx-1,i*nptx-1] = 3/2/hy
    K2[i*nptx-1,i*nptx-2] = -1/2/hy
    K2[i*nptx-1,i*nptx-3] = 2/hy
###########################

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

###########
#### Matrice de Convection  (Centré)
k=np.array([-1*np.ones(nx+1), np.zeros(nptx), 1*np.ones(nx+1)], dtype=object)
offset=[-1, 0, 1]
aux=(vx/2/hx) * diags(k, offset).toarray()
aux[0,:] = 0
aux[-1,:] = 0
aux3=np.eye(npty)
aux3[0, 0]=0
aux3[-1, -1]=0
K2x_dif= np.kron( aux3, aux)
V2Dx=np.copy(vx * K2x_dif)

#### Global matrix : diffusion + convection
A2 = -(K2 + V2Dx) #-mu*Delta + V.grad
#
#
##  Cas explicite
u = np.zeros((nx+2)*(ny+2))
u_ex = np.zeros((nx+2)*(ny+2))
err = np.zeros((nx+2)*(ny+2))
F = np.zeros((nx+2)*(ny+2))
#
#
# =============================================================================
# Time stepping
# =============================================================================
s0 = 0.1
x0 = 0.25
y0=0.5

def Sol_init(x):
    return np.exp( -((x[0]-x0)/s0)**2 -((x[1]-y0)/s0)**2   )

def Sol_init2(x):
    to_return = 0
    term = s0 ** 2 - (x[0] - x0) ** 2 - (x[1] -y0 ) ** 2
    if  term >0:
     to_return = term
    return to_return


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

concentration_ini = max(u)
concentration = [concentration_ini /concentration_ini ]
# Nombre de pas de temps effectues
nt = int(Tfinal/dt)
Tfinal = nt*dt # on corrige le temps final (si Tfinal/dt n'est pas entier)

# Time loop
for n in range(1,nt+1):

  # Schéma explicite en temps
    u = u + dt*A2.dot(u)
    
     # Schéma implicite de Crank-Nicholson
    #u = np.linalg.solve((np.eye(A2.shape[0])-dt/2*A2),(np.eye(A2.shape[0])+dt/2*A2).dot(u)) 
    concentration.append(max(u)/concentration_ini)
 # Print solution
    if n%5 == 0:
      plt.figure(1)
      plt.clf()
      fig = plt.figure(figsize=(10, 7))
      ax = plt.axes(projection='3d')
      uu = np.reshape(u,(nx+2 ,ny+2),order = 'F');
      surf = ax.plot_surface(X, Y, uu.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
      ax.view_init(60, 35)
      plt.title(['Schema crank sol 1 avec CFL=%s' %(cfl), '$t=$%s' %(n*dt)])
      plt.pause(0.1)


####################################################################
# comparaison solution exacte avec solution numerique au temps final
j0 = int((npty-1)/2)

it = None
plt.figure(2)
plt.clf()
x = np.linspace(xmin,xmax,nptx)
plt.plot(x,uu_init[:,j0],x,uu[:,j0],'k') #,x,uexacte,'or')
plt.title("schéma crank  sol 2  T = "+str(Tfinal))
plt.legend(['Solution initiale','Schema implicite =%s' %(cfl)]) #,'solution exacte'],loc='best')
plt.show()
# temps = np.arange(0,Tfinal + dt, dt)
# for i in range(len(concentration)):
#     if concentration[i] <= 0.2 * concentration[0]:
#         print("temps  ",Tfinal/ i+1,"\ni  ",i)
#         if it == None:
#             it = i-1
        
# plt.plot(temps,concentration,label = "concentration")
# plt.axhline(y = 0.2,color="black",ls="dotted",label=" y = 0.2")
# plt.axvline(x = temps[it],color="black",ls="dotted",label=str(temps[it]))
# plt.xlabel("T")
# plt.ylabel("Concentration max")
# plt.title("Evolution de la concentration maximale pour la seconde solution")
# plt.legend()
# plt.grid()
# plt.show()
# DT = (np.arange(0.01,Tfinal,0.01))
# CFL =mu*DT/hx**2+mu*DT/hy**2 
# plt.plot(DT,CFL,label="CFL en foction de dt")
# plt.xlabel("dt")
# plt.ylabel("cfl")
# plt.grid()
# plt.axhline(0.5,color ="red",ls="--",label="y = 0.5")
# plt.legend()
# plt.show()