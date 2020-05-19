# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:27:59 2020

@author: trubin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Quadratic_Poly_Functions as QBF
import polynomial2d as p2d
import PreProc



def evalErrorFE(f,Phi,NodeList,GlobalElementMatrix,order):
    x_i=np.array([-0.932469514203152,-0.661209386466265,-0.238619186083197,0.238619186083197,0.661209386466265,0.932469514203152])
    w_i=np.array([0.171324492379170,0.360761573048139,0.467913934572691,0.467913934572691,0.360761573048139,0.171324492379170])
    if order==5:
        x_i=np.array([-np.sqrt(5+2*np.sqrt(10/7))/3,-np.sqrt(5-2*np.sqrt(10/7))/3,0,np.sqrt(5-2*np.sqrt(10/7))/3,np.sqrt(5+2*np.sqrt(10/7))/3])
        w_i=np.array([(322-13*np.sqrt(70))/900,(322+13*np.sqrt(70))/900,128/225,(322+13*np.sqrt(70))/900,(322-13*np.sqrt(70))/900])        
    L2Err=0
    for k in range(GlobalElementMatrix.shape[0]):
        XYele = NodeList[1:3,GlobalElementMatrix[k,1:10]]      
        x=NodeList[1,GlobalElementMatrix[k,1:10]]@QBF.Na
        y=NodeList[2,GlobalElementMatrix[k,1:10]]@QBF.Na
        
        for n in range(len(x_i)):
            for m in range(len(x_i)):
                x1=x(x_i[n],x_i[m])
                y1=y(x_i[n],x_i[m])

                L2Err+=w_i[n]*w_i[m]*(f(x1,y1)-(Phi[GlobalElementMatrix[k,1:10]]@QBF.Na)(x_i[n],x_i[m]))**2*PreProc.detJ(x_i[n],x_i[m], XYele)

    return np.sqrt(L2Err)


def PoissonSol(x,y):
    res=(1-x**2)/2
    for k in range(1,100,2):
        res-=16*np.pi**(-3)*(np.sin(k*np.pi*(1+x)/2)*k**(-3)/np.sinh(k*np.pi))*(np.sinh(k*np.pi*(1+y)/2)+np.sinh(k*np.pi*(1-y)/2))
    return res

def plotFEelem(Phi,GlobalElementMatrix,NodeList,fignum,res=5):   
    xp=np.linspace(-1,1,res)
    yp=np.linspace(-1,1,res)
    
    Xnode=np.array([])
    Ynode=np.array([])
    phinode=np.array([])
    for k in range(GlobalElementMatrix.shape[0]):
        for i in range(len(xp)):
            for j in range(len(yp)):
                phinode=np.append(phinode,(Phi[GlobalElementMatrix[k,1:10]]@QBF.Na)(xp[i],yp[j]))
                Xnode=np.append(Xnode,(NodeList[1,GlobalElementMatrix[k,1:10]]@QBF.Na)(xp[i],yp[j]))
                Ynode=np.append(Ynode,(NodeList[2,GlobalElementMatrix[k,1:10]]@QBF.Na)(xp[i],yp[j]))

   
    plt.figure(fignum)
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(Xnode,Ynode,phinode,
                cmap='viridis')

XDom=2
YDom=2

X=XDom*(np.array([0, 1, 1, 0, 0.5 , 1   ,0.5 ,0   ,0.5],dtype=np.float64)-0.5)
Y=YDom*(np.array([0, 0, 1, 1, 0   , 0.5 ,1   ,0.5 ,0.5],dtype=np.float64)-0.5)

XY=np.array([X,Y])

N_row=100
N_col=100

L2norm=np.zeros((20,20))

for N_row in range(1,21):
    for N_col in range(N_row,21):
 
        GlobalElementMatrix,NodeList,Connectivity=PreProc.NodeArrange(N_row,N_col,XY,0,0)
        
        GlobalStiffMat,GlobalForceMat=PreProc.assembleGlobalMats(GlobalElementMatrix,NodeList,5)
        
        rhoFE=np.ones(NodeList.shape[1])
        BCnodes=np.array([*np.arange(0,2*N_row+1), 
                          *np.arange(1+2*N_row,1+2*N_row+(N_col)*(2+4*N_row),(2+4*N_row)),
                          *np.arange(2+4*N_row,2+4*N_row+(N_col)*(2+4*N_row),(2+4*N_row)),
                          *np.arange(2*(N_row-1)+3+2*N_row,2*(N_row-1)+(N_col)*(2+4*N_row)+3+2*N_row,(2+4*N_row)),
                          *np.arange(2*(N_row-1)+4+4*N_row,2*(N_row-1)+(N_col)*(2+4*N_row)+4+4*N_row,(2+4*N_row)),
                          *np.arange((N_col)*(2+4*N_row)-1,2*N_row+(N_col)*(2+4*N_row))
                          ])
        F=GlobalForceMat@rhoFE
        F[BCnodes]=0 #BC
        
        
        for i in BCnodes:
            GlobalStiffMat[i,:]=np.zeros((1,GlobalStiffMat.shape[1]))
            GlobalStiffMat[i,i]=1
        
        Phi=np.linalg.solve(GlobalStiffMat,F)
        
        
        
        
# =============================================================================
#         plt.figure(1)
#         ax = plt.axes(projection='3d')
#         ax.plot_trisurf(NodeList[1,:],NodeList[2,:],Phi,
#                         cmap='viridis')
#         
#         
#         xp=np.linspace(-1,1,50)
#         yp=np.linspace(-1,1,50)
#         
#         Xp, Yp = np.meshgrid(xp, yp)
#         
#         fgrid=np.zeros((len(xp),len(xp)))
#         for i in range(len(Xp)):
#             for j in range(len(Yp)):
#                 fgrid[i,j]=PoissonSol(Xp[i,j], Yp[i,j])
#         
#         
#         plt.figure(2)
#         ax = plt.axes(projection='3d')
#         ax.plot_surface(Xp,Yp,fgrid,
#                         cmap='viridis')
#         
# =============================================================================
        
        L2norm[N_row-1,N_col-1]=(evalErrorFE(PoissonSol,Phi,NodeList,GlobalElementMatrix,6))
        L2norm[N_col-1,N_row-1]=L2norm[N_row-1,N_col-1]
        print(str(N_row) + ' ' +str(N_col))

xp=np.arange(1,21)
yp=np.arange(1,21)
         
Xp, Yp = np.meshgrid(xp, yp)
plt.figure(3)
ax = plt.axes(projection='3d')
ax.plot_surface(Xp,Yp,L2norm,
         cmap='viridis')


plt.figure(4)
for i in range(7):
    plt.semilogy(np.arange(1,21),L2norm[i,:],'.-',label=str(i+1)+' column elements')
plt.title('L2 norm of error')
plt.xlabel('number of row elements')
plt.ylabel('error')

plt.legend()
plt.xlim(1,20)
plt.xticks(np.arange(1,21))
