# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:14:24 2020

@author: trubin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import PreProc

def pickle(file,data):
    import pickle
    with open(file, 'wb') as fo:
        pickle.dump(data, fo)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res        
        
def projectDGintoFE(rho,NodeList,GlobalElementMatrix):
    rhoFE=np.zeros(NodeList.shape[1])
    for elemSol in rho:
        for i in range(9):
            rhoFE[GlobalElementMatrix[int(elemSol[0]),1+i]]+=elemSol[1+i]
    return rhoFE



def setInitDenDist(x0,y0,r0,denval,rho,NodeList,GlobalElementMatrix):
    for elemSol in rho:
        for i in range(9):
            dist=(NodeList[1,GlobalElementMatrix[int(elemSol[0]),i+1]]-x0)**2+(NodeList[2,GlobalElementMatrix[int(elemSol[0]),i+1]]-y0)**2
            if dist<r0**2: 
                elemSol[1+i]+=denval
    return rho,plt.Circle((x0,y0), r0, color='r')


new_mesh=True
save_mesh=True
load_mesh=False


XDom=4
YDom=3

X=XDom*(np.array([0, 1, 1, 0, 0.5 , 1   ,0.5 ,0   ,0.5],dtype=np.float64)-0.5)
Y=YDom*np.array([0, 0, 1, 1, 0   , 0.5 ,1   ,0.5 ,0.5],dtype=np.float64)

# =============================================================================
# 
# r1=1
# r2=2
# r3=3
# X=np.array([r1*np.cos(0.75*np.pi), r1*np.cos(0.25*np.pi), r3*np.cos(0.25*np.pi), r3*np.cos(0.75*np.pi), r1*np.cos(0.5*np.pi) , r2*np.cos(0.25*np.pi)   ,r3*np.cos(0.5*np.pi) ,r2*np.cos(0.75*np.pi)   ,r2*np.cos(0.5*np.pi)],dtype=np.float64)
# Y=np.array([r1*np.sin(0.75*np.pi), r1*np.sin(0.25*np.pi), r3*np.sin(0.25*np.pi), r3*np.sin(0.75*np.pi), r1*np.sin(0.5*np.pi) , r2*np.sin(0.25*np.pi)   ,r3*np.sin(0.5*np.pi) ,r2*np.sin(0.75*np.pi)   ,r2*np.sin(0.5*np.pi)],dtype=np.float64)
# 
# =============================================================================

# =============================================================================
# r1=1
# r2=2
# r3=3
# X=np.array([r1*np.cos(0.75*np.pi), -r3*np.cos(0.75*np.pi), -r1*np.cos(0.75*np.pi), r3*np.cos(0.75*np.pi), r1*np.cos(0.5*np.pi) , r2*np.cos(0.25*np.pi)   ,r3*np.cos(0.5*np.pi) ,r2*np.cos(0.75*np.pi)   ,r2*np.cos(0.5*np.pi)],dtype=np.float64)
# Y=np.array([r1*np.sin(0.75*np.pi), r3+r1-r3*np.sin(0.25*np.pi), r3+r1-r1*np.sin(0.25*np.pi), r3*np.sin(0.75*np.pi), r1*np.sin(0.5*np.pi) , r3+r1-r2*np.sin(0.25*np.pi)   ,r3*np.sin(0.5*np.pi) ,r2*np.sin(0.75*np.pi)   ,r2*np.sin(0.5*np.pi)],dtype=np.float64)
# 
# =============================================================================
XY=np.array([X,Y])





if new_mesh:
    GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,EleFluxMatx,EleFluxMaty,EleMassMat=PreProc.PreProc(XY,20,20,5)    
if save_mesh:
    Mesh=GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,EleFluxMatx,EleFluxMaty,EleMassMat
    pickle("C:\\py\\Numerical Algorithms for Scientific Computing\\Meshs\\M_2020SquareReducedQuad",Mesh)
if load_mesh:
    import os
    files=os.listdir("C:\\py\\Numerical Algorithms for Scientific Computing\\Meshs")
    print(files)
    print('Pick Mesh')
    filenum=int(input())
    LoadedMesh=unpickle("C:\\py\\Numerical Algorithms for Scientific Computing\\Meshs\\"+files[filenum-1])
    GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,EleFluxMatx,EleFluxMaty,EleMassMat=LoadedMesh
    
    
    
chargeOverPermitt=1.8095128169876496e-08


rho=np.zeros((GlobalElementMatrix.shape[0],10)) #element #, n node 1, n node 2,...
vx=np.ones((GlobalElementMatrix.shape[0],10)) #element #, n node 1, n node 2,...
vy=np.zeros((GlobalElementMatrix.shape[0],10)) #element #, n node 1, n node 2,...

rho[:,0]=GlobalElementMatrix[:,0]
vx[:,0]=GlobalElementMatrix[:,0]
vy[:,0]=GlobalElementMatrix[:,0]




F=np.zeros(GlobalForceMat.shape[0])
# =============================================================================
# for i in range(rho.shape[0]):
#     for j in range(rho.shape[1]-1):
#         rho[i,j+1]=1
# =============================================================================

#rho,circ1=setInitDenDist(0.625,0.5,0.2,1,rho,NodeList,GlobalElementMatrix)
#rho,circ2=setInitDenDist(1.325,0.5,0.2,1,rho,NodeList,GlobalElementMatrix)

rho,circ1=setInitDenDist(-0.8,2,0.3,-1,rho,NodeList,GlobalElementMatrix)
rho,circ2=setInitDenDist(0.8,2,0.3,1,rho,NodeList,GlobalElementMatrix)
rho,circ3=setInitDenDist(-0.8,0.6,0.3,2,rho,NodeList,GlobalElementMatrix)
rho,circ4=setInitDenDist(0.8,0.6,0.3,-2,rho,NodeList,GlobalElementMatrix)


plt.figure(1)
plt.plot(NodeList[1,:],NodeList[2,:],'.')
ax = plt.gca()
ax.add_artist(circ1)
ax.add_artist(circ2)
ax.add_artist(circ3)
ax.add_artist(circ4)








#BC
plt.figure(3)
for i in BCnodes:
    GlobalStiffMat[i,:]=np.zeros((1,GlobalStiffMat.shape[1]))
    GlobalStiffMat[i,i]=1
#   F[i]=0
    plt.plot(NodeList[1,i],NodeList[2,i],'*')

DT=0.01
#time loop
rhoEF=projectDGintoFE(rho,NodeList,GlobalElementMatrix)    
F=chargeOverPermitt*GlobalForceMat@rhoEF
F[BCnodes]=0 #BC

Phi=np.linalg.solve(GlobalStiffMat,F)

for k,elemSol in enumerate(rho):
    EleFluxMat=np.zeros((9,9))
    for i in range(9):
        for j in range(9):    
            for m in range(9):
                EleFluxMat[i,j]+=EleFluxMatx[k,i,j,m]*vx[k,m+1]+EleFluxMaty[k,i,j,m]*vy[k,m+1]
    
    b=(EleMassMat[k,:,:]+DT*EleFluxMat)@elemSol[1:10]


plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_trisurf(NodeList[1,:],NodeList[2,:],Phi,
                cmap='viridis')

plt.figure(4)
plt.plot(NodeList[1,:],Phi)

plt.figure(5)
plt.plot(NodeList[2,:],Phi)
# =============================================================================
# Xnode=np.array([])
# Ynode=np.array([])
# rhonode=np.array([])
# for elemSol in rho:
#     for i in range(9):
#         Xnode=np.append(Xnode,NodeList[1,GlobalElementMatrix[int(elemSol[0]),i+1]])
#         Ynode=np.append(Ynode,NodeList[2,GlobalElementMatrix[int(elemSol[0]),i+1]])
#         rhonode=np.append(rhonode,elemSol[i+1])
# 
# plt.figure(5)
# ax = plt.axes(projection='3d')
# ax.scatter(Xnode,Ynode,rhonode)
# =============================================================================

#plt.imshow(GlobalStiffMat)