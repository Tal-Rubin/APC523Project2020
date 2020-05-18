# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:14:24 2020

@author: trubin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Quadratic_Poly_Functions as QBF


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
        
def projectDGintoFE(rho,NodeList,GlobalElementMatrix,Proj,DGorder):
    rhoFE=np.zeros(NodeList.shape[1])
    for k in range(GlobalElementMatrix.shape[0]):
        rhoFE[GlobalElementMatrix[k,1:10]]+=Proj@rho[DGorder*k:DGorder*(k+1)]
    return rhoFE

def plotDG(rho,fignum,DGorder):
    Xnode=np.array([])
    Ynode=np.array([])
    rhonode=np.array([])
    for k in range(GlobalElementMatrix.shape[0]):
#        for i in range(9):
#        if rho[9*k]>0:
#            continue
        rhonode=np.append(rhonode,rho[DGorder*k])
        Xnode=np.append(Xnode,NodeList[1,GlobalElementMatrix[k,9]])
        Ynode=np.append(Ynode,NodeList[2,GlobalElementMatrix[k,9]])
            
    plt.figure(fignum)
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(Xnode,Ynode,rhonode,
                cmap='viridis')
    
    
def plotDGelem(rho,GlobalElementMatrix,NodeList,fignum,DGorder,res=5):   #Write
    xp=np.linspace(-.95,.95,res)
    yp=np.linspace(-.95,.95,res)
    
    Xnode=np.array([])
    Ynode=np.array([])
    rhonode=np.array([])
    for k in range(GlobalElementMatrix.shape[0]):
        for i in range(len(xp)):
            for j in range(len(yp)):
                
                rhonode=np.append(rhonode,(rho[DGorder*k:DGorder*(k+1)]@QBF.Ma[0:DGorder])(xp[i],yp[j]))
                Xnode=np.append(Xnode,(NodeList[1,GlobalElementMatrix[k,1:10]]@QBF.Na)(xp[i],yp[j]))
                Ynode=np.append(Ynode,(NodeList[2,GlobalElementMatrix[k,1:10]]@QBF.Na)(xp[i],yp[j]))

# =============================================================================
#     for i in range(len(Xp)):
#         for j in range(len(Yp)):
#             Jgrid[i,j]=detJ(Xp[i,j], Yp[i,j],XY)
#     
# =============================================================================
    
    plt.figure(fignum)
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(Xnode,Ynode,rhonode,
                cmap='viridis')


def setInitDenDist(x0,y0,r0,denval,rho,NodeList,GlobalElementMatrix,DGorder):
    for k in range(GlobalElementMatrix.shape[0]):
      #  for i in range(9):
        dist=(NodeList[1,GlobalElementMatrix[k,9]]-x0)**2+(NodeList[2,GlobalElementMatrix[k,9]]-y0)**2
        #if dist<r0**2: 
        rho[DGorder*k]+=denval*np.exp(-dist*r0)
        #rho[9*k]+=1
    return rho,plt.Circle((x0,y0), r0, color='r')

def setInitDenDistFE(x0,y0,r0,denval,rho,NodeList,GlobalElementMatrix):
    for k in range(NodeList.shape[1]):
        dist=(NodeList[1,k]-x0)**2+(NodeList[2,k]-y0)**2
        rho[k]+=denval*np.exp(-dist*r0)
    return rho


def sumcellav(rho,GlobalElementMatrix,DGorder):
    s=0
    for k in range(GlobalElementMatrix.shape[0]):
        s+=rho[DGorder*k]
    return s


def minmod(a,b,c):
    if a>0 and b>0 and c>0:
        return min(a,b,c)
    if a<0 and b<0 and c<0:
        return max(a,b,c)
    return 0


def CenterDist(k1,k2,GlobalElementMatrix,NodeList):
    return np.sqrt((NodeList[1,GlobalElementMatrix[k1,9]]-NodeList[1,GlobalElementMatrix[k2,9]])**2+(NodeList[2,GlobalElementMatrix[k1,9]]-NodeList[2,GlobalElementMatrix[k2,9]])**2)

new_mesh=True
save_mesh=False
load_mesh=False


XDom=10
YDom=10

X=XDom*(np.array([0, 1, 1, 0, 0.5 , 1   ,0.5 ,0   ,0.5],dtype=np.float64)-0.5)
Y=YDom*np.array([0, 0, 1, 1, 0   , 0.5 ,1   ,0.5 ,0.5],dtype=np.float64)


N_row=15
N_col=15

DGorder=3


# =============================================================================
# r1=5
# r2=7.5
# r3=10
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
    GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,EleFluxMatx,EleFluxMaty,InvMassMat,b_mast_1x,b_mast_1y,b_mast_2x,b_mast_2y,b_mast_3x,b_mast_3y,b_mast_4x,b_mast_4y,Connectivity=PreProc.PreProc(XY,N_row,N_col,DGorder,5)    
if save_mesh:
    Mesh=GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,EleFluxMatx,EleFluxMaty,InvMassMat,b_mast_1x,b_mast_1y,b_mast_2x,b_mast_2y,b_mast_3x,b_mast_3y,b_mast_4x,b_mast_4y,Connectivity
    pickle("C:\\py\\Numerical Algorithms for Scientific Computing\\Meshs\\M_5050QuartCircQuadNEW",Mesh)
if load_mesh:
    import os
    files=os.listdir("C:\\py\\Numerical Algorithms for Scientific Computing\\Meshs")
    print(files)
    print('Pick Mesh')
    filenum=int(input())
    LoadedMesh=unpickle("C:\\py\\Numerical Algorithms for Scientific Computing\\Meshs\\"+files[filenum-1])
    GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,EleFluxMatx,EleFluxMaty,InvMassMat,b_mast_1x,b_mast_1y,b_mast_2x,b_mast_2y,b_mast_3x,b_mast_3y,b_mast_4x,b_mast_4y,Connectivity=LoadedMesh
    

Proj=np.zeros((9,DGorder))
for i in range(9):
    for j in range(DGorder):
            Proj[i,j]=(QBF.Na[i]*QBF.Ma[j]).intx(-1,1).inty(-1,1).scalar()    




    
chargeOverPermitt=1.8095128169876496e-08


rho=np.zeros((GlobalElementMatrix.shape[0]*DGorder)) #element #, n node 1, n node 2,...
vx=np.zeros((GlobalElementMatrix.shape[0]*DGorder)) #element #, n node 1, n node 2,...
vy=np.zeros((GlobalElementMatrix.shape[0]*DGorder)) #element #, n node 1, n node 2,...

for i in range(GlobalElementMatrix.shape[0]):
    vx[DGorder*i]=1




# =============================================================================
# rho[:,0]=GlobalElementMatrix[:,0]
# vx[:,0]=GlobalElementMatrix[:,0]
# vy[:,0]=GlobalElementMatrix[:,0]
# =============================================================================




# =============================================================================
# for i in range(rho.shape[0]):
#     for j in range(rho.shape[1]-1):
#         rho[i,j+1]=1
# =============================================================================

#rho,circ1=setInitDenDist(0.625,0.5,0.2,1,rho,NodeList,GlobalElementMatrix)
#rho,circ2=setInitDenDist(1.325,0.5,0.2,1,rho,NodeList,GlobalElementMatrix)


rho,circ1=setInitDenDist(0,5,1,1,rho,NodeList,GlobalElementMatrix,DGorder)
#rho,circ2=setInitDenDist(0.8,2,0.3,1,rho,NodeList,GlobalElementMatrix)
#rho,circ3=setInitDenDist(-0.8,0.6,0.3,2,rho,NodeList,GlobalElementMatrix)
#rho,circ4=setInitDenDist(0.8,0.6,0.3,-2,rho,NodeList,GlobalElementMatrix)


plt.figure(1)
plt.plot(NodeList[1,:],NodeList[2,:],'.')
ax = plt.gca()
ax.add_artist(circ1)
#ax.add_artist(circ2)
#ax.add_artist(circ3)
#ax.add_artist(circ4)








#BC
plt.figure(3)
for i in BCnodes:
    GlobalStiffMat[i,:]=np.zeros((1,GlobalStiffMat.shape[1]))
    GlobalStiffMat[i,i]=1
#   F[i]=0
    plt.plot(NodeList[1,i],NodeList[2,i],'*')
rho0=rho

DT=0.001
#time loop
rhoFE=setInitDenDistFE(0,5,0.5,1,np.zeros(NodeList.shape[1])

#rhoFE=setInitDenDistFE(2,6.5,0.8,1,np.zeros(NodeList.shape[1]),NodeList,GlobalElementMatrix)
#rhoFE=setInitDenDistFE(-2,6.5,0.8,-1,rhoFE,NodeList,GlobalElementMatrix)

F=chargeOverPermitt*GlobalForceMat@rhoFE
F[BCnodes]=0 #BC

Phi=np.linalg.solve(GlobalStiffMat,F)

plt.figure(4)
ax = plt.axes(projection='3d')
ax.plot_trisurf(NodeList[1,:],NodeList[2,:],rhoFE,
                cmap='viridis')




plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_trisurf(NodeList[1,:],NodeList[2,:],Phi,
                cmap='viridis')



FluxMat=np.zeros((GlobalElementMatrix.shape[0]*DGorder,GlobalElementMatrix.shape[0]*DGorder))
for k in range(GlobalElementMatrix.shape[0]):
    XYele = NodeList[1:3,GlobalElementMatrix[k,1:10]]
    for i in range(DGorder):
        for j in range(DGorder):    
            for m in range(DGorder):
                FluxMat[DGorder*k+i,DGorder*k+j]+=EleFluxMatx[k,i,j,m]*vx[DGorder*k+m]+EleFluxMaty[k,i,j,m]*vy[DGorder*k+m]
                
                FluxMat[DGorder*k+i,DGorder*k+j]+=0.5*(b_mast_1x[k,i,j,m]*vx[DGorder*k+m]-b_mast_1y[k,i,j,m]*vy[DGorder*k+m])
                
                FluxMat[DGorder*k+i,DGorder*k+j]+=0.5*(b_mast_2x[k,i,j,m]*vx[DGorder*k+m]-b_mast_2y[k,i,j,m]*vy[DGorder*k+m])
                FluxMat[DGorder*k+i,DGorder*k+j]+=0.5*(b_mast_3x[k,i,j,m]*vx[DGorder*k+m]-b_mast_3y[k,i,j,m]*vy[DGorder*k+m])
                FluxMat[DGorder*k+i,DGorder*k+j]+=0.5*(b_mast_4x[k,i,j,m]*vx[DGorder*k+m]-b_mast_4y[k,i,j,m]*vy[DGorder*k+m])
                
                if Connectivity[k,0]!=-1:
                    FluxMat[DGorder*Connectivity[k,0]+i,DGorder*k+j]-=0.5*(b_mast_3x[k,i,j,m]*vx[DGorder*k+m]-b_mast_3y[k,i,j,m]*vy[DGorder*k+m])
                        
                if Connectivity[k,1]!=-1:
                    FluxMat[DGorder*Connectivity[k,1]+i,DGorder*k+j]-=0.5*(b_mast_4x[k,i,j,m]*vx[DGorder*k+m]-b_mast_4y[k,i,j,m]*vy[DGorder*k+m])
                        
                if Connectivity[k,2]!=-1:
                    FluxMat[DGorder*Connectivity[k,2]+i,DGorder*k+j]-=0.5*(b_mast_1x[k,i,j,m]*vx[DGorder*k+m]-b_mast_1y[k,i,j,m]*vy[DGorder*k+m])
                
                if Connectivity[k,3]!=-1: 
                    FluxMat[DGorder*Connectivity[k,3]+i,DGorder*k+j]-=0.5*(b_mast_2x[k,i,j,m]*vx[DGorder*k+m]-b_mast_2y[k,i,j,m]*vy[DGorder*k+m])

                                           

for i in range(50):

    rho1=rho+DT*InvMassMat@FluxMat@rho
# =============================================================================
#     
#     for k in range(GlobalElementMatrix.shape[0]):
#         if Connectivity[k,1]!=-1 and Connectivity[k,3]!=-1:
#             rho1[9*k+1]=minmod(rho1[9*k+1],(rho1[9*Connectivity[k,1]]-rho1[9*k])/CenterDist(k,Connectivity[k,1],GlobalElementMatrix,NodeList),(rho1[9*k]-rho1[9*Connectivity[k,3]])/CenterDist(k,Connectivity[k,3],GlobalElementMatrix,NodeList))
#         if Connectivity[k,0]!=-1 and Connectivity[k,2]!=-1:
#             rho1[9*k+2]=minmod(rho1[9*k+2],(rho1[9*Connectivity[k,2]]-rho1[9*k])/CenterDist(k,Connectivity[k,2],GlobalElementMatrix,NodeList),(rho1[9*k]-rho1[9*Connectivity[k,0]])/CenterDist(k,Connectivity[k,0],GlobalElementMatrix,NodeList))
#     
# =============================================================================
    
    
    rho2=0.75*rho+0.25*rho1+0.25*DT*InvMassMat@FluxMat@rho1

# =============================================================================
#     for k in range(GlobalElementMatrix.shape[0]):
#         if Connectivity[k,1]!=-1 and Connectivity[k,3]!=-1:
#             rho2[9*k+1]=minmod(rho2[9*k+1],(rho2[9*Connectivity[k,1]]-rho2[9*k])/CenterDist(k,Connectivity[k,1],GlobalElementMatrix,NodeList),(rho2[9*k]-rho2[9*Connectivity[k,3]])/CenterDist(k,Connectivity[k,3],GlobalElementMatrix,NodeList))
#         if Connectivity[k,0]!=-1 and Connectivity[k,2]!=-1:
#             rho2[9*k+2]=minmod(rho2[9*k+2],(rho2[9*Connectivity[k,2]]-rho2[9*k])/CenterDist(k,Connectivity[k,2],GlobalElementMatrix,NodeList),(rho2[9*k]-rho2[9*Connectivity[k,0]])/CenterDist(k,Connectivity[k,0],GlobalElementMatrix,NodeList))
# 
# 
# =============================================================================

    rho3=1/3*rho+2/3*rho2+2/3*DT*InvMassMat@FluxMat@rho2

    
# =============================================================================
#     for k in range(GlobalElementMatrix.shape[0]):
#         if Connectivity[k,1]!=-1 and Connectivity[k,3]!=-1:
#             rho3[9*k+1]=minmod(rho3[9*k+1],(rho3[9*Connectivity[k,1]]-rho3[9*k])/CenterDist(k,Connectivity[k,1],GlobalElementMatrix,NodeList),(rho2[9*k]-rho3[9*Connectivity[k,3]])/CenterDist(k,Connectivity[k,3],GlobalElementMatrix,NodeList))
#         if Connectivity[k,0]!=-1 and Connectivity[k,2]!=-1:
#             rho3[9*k+2]=minmod(rho3[9*k+2],(rho3[9*Connectivity[k,2]]-rho3[9*k])/CenterDist(k,Connectivity[k,2],GlobalElementMatrix,NodeList),(rho2[9*k]-rho3[9*Connectivity[k,0]])/CenterDist(k,Connectivity[k,0],GlobalElementMatrix,NodeList))
# 
# =============================================================================
    for k in range(GlobalElementMatrix.shape[0]):
        if Connectivity[k,1]==-1 or Connectivity[k,3]==-1 or Connectivity[k,0]==-1 or Connectivity[k,2]==-1:
            rho3[DGorder*k:DGorder*(k+1)]=rho0[DGorder*k:DGorder*(k+1)]
            
    rho=rho3

#plt.imshow(FluxMat)



########   How to deal with corners? in regular grid with v=(1,1) element 1,1 wouldnt get the right signal

#plt.imshow(GlobalStiffMat)

plotDGelem(rho, GlobalElementMatrix, NodeList, 5, DGorder)
plotDGelem(rho0, GlobalElementMatrix, NodeList, 6, DGorder)

#200 timesteps


