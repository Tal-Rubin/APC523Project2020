# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:42:00 2020

@author: trubin
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Quadratic_Poly_Functions as QBF
import numpy as np
import PreProc
import Solver
import PostProc
import pickle
import os


def pickle(file,data):
    import pickle

    with open(file, 'wb') as fo:
        pickle.dump(data, fo)
def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res        



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
   



    
    
def TorX(N_row,N_col,Connectivity):
    for k in range(N_row):
        Connectivity[k,3]=N_row*(N_col-1)+k
        Connectivity[N_row*(N_col-1)+k,1]=k
    return Connectivity

def TorY(N_row,N_col,Connectivity):
    for k in range(N_col):
        Connectivity[k*N_row,2]=k*N_row+N_row-1
        Connectivity[k*N_row+N_row-1,0]=k*N_row
    return Connectivity



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

def projectDGintoFE(rho,NodeList,GlobalElementMatrix,DGorder):
    rhoFE=np.zeros(NodeList.shape[1])
    Proj=np.zeros((9,DGorder))
    for i in range(9):
        for j in range(DGorder):
            Proj[i,j]=QBF.Mano[j]*(QBF.Na[i]*QBF.Ma[j]).intx(-1,1).inty(-1,1).scalar()    
    for k in range(GlobalElementMatrix.shape[0]):
        rhoFE[GlobalElementMatrix[k,1:10]]+=Proj@rho[DGorder*k:DGorder*(k+1)]
    return rhoFE

def projectFEintoDG(rhoFE,NodeList,GlobalElementMatrix,DGorder):
    rho=np.zeros(GlobalElementMatrix.shape[0]*DGorder)
    Proj=np.zeros((DGorder,9))
    for i in range(DGorder):
        for j in range(9):
            Proj[i,j]=QBF.Mano[i]*(QBF.Ma[i]*QBF.Na[j]).intx(-1,1).inty(-1,1).scalar()    
    for k in range(GlobalElementMatrix.shape[0]):
        rho[DGorder*k:DGorder*(k+1)]+=Proj@rhoFE[GlobalElementMatrix[k,1:10]]
    return rho


new_mesh=True
save_mesh=False
load_mesh=False


XDom=10
YDom=10

X=XDom*(np.array([0, 1, 1, 0, 0.5 , 1   ,0.5 ,0   ,0.5],dtype=np.float64)-0.5)
Y=YDom*np.array([0, 0, 1, 1, 0   , 0.5 ,1   ,0.5 ,0.5],dtype=np.float64)


N_row=50
N_col=50

DGorder=1

# =============================================================================
# 
# r1=2
# r2=6
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
    GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,InvMassMat,EleFluxMat,Bound_mat,LF_mat,Connectivity,EleVol=PreProc.PreProc(XY,N_row,N_col,DGorder,5,CalcFE=False)    
    Mesh=GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,InvMassMat,EleFluxMat,Bound_mat,LF_mat,Connectivity,EleVol
if save_mesh:
    if  os.path.isdir('./Meshs')==False:
        os.mkdir('./Meshs')
    pickle("./Meshs/M_5050QuartCircQuadNEW",Mesh)
if load_mesh and os.path.isdir('./Meshs'):
    files=[f for f in os.listdir('./Meshs') if os.path.isfile(f)] 
    print(files)
    print('Pick Mesh')
    filenum=int(input())
    LoadedMesh=unpickle("./Meshs/"+files[filenum-1])
    GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,InvMassMat,EleFluxMat,Bound_mat,LF_mat,Connectivity,EleVol=LoadedMesh
    


#pretzel x
Connectivity=TorX(N_row,N_col,Connectivity)

Connectivity=TorY(N_row,N_col,Connectivity)




rho=np.zeros((GlobalElementMatrix.shape[0]*DGorder)) #element #, n node 1, n node 2,...
vx=np.zeros((GlobalElementMatrix.shape[0]*DGorder)) #element #, n node 1, n node 2,...
vy=np.zeros((GlobalElementMatrix.shape[0]*DGorder)) #element #, n node 1, n node 2,...

for i in range(GlobalElementMatrix.shape[0]):
    vx[DGorder*i]=1
    #vy[DGorder*i]=0.1



# =============================================================================
# rho[:,0]=GlobalElementMatrix[:,0]
# vx[:,0]=GlobalElementMatrix[:,0]
# vy[:,0]=GlobalElementMatrix[:,0]
# =============================================================================

rhoFE=setInitDenDistFE(0,6,0.5,1,np.zeros(NodeList.shape[1]),NodeList,GlobalElementMatrix)


# =============================================================================
# plt.figure(3)
# plt.title('FE boundary condition nodes')
# plt.plot(NodeList[1,BCnodes],NodeList[2,BCnodes],'*')
# =============================================================================

DT=0.01
Tend=50
Sample=50

import shutil
shutil.rmtree('./Data')
if  os.path.isdir('./Data')==False:
        os.mkdir('./Data')
pickle('./Data/Grid',(GlobalElementMatrix,NodeList))


Phi= Solver.Solver(DT,Tend,Sample,rhoFE,vx,vy,Mesh,DGorder,CalcFE=False)
PostProc.PostProc()



#plotDGelem(rho0, GlobalElementMatrix, NodeList, 0, DGorder,2)



# =============================================================================
# 
# plt.figure(4)
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(NodeList[1,:],NodeList[2,:],rhoFE,
#                 cmap='viridis')
# 
# 
# 
# 
# plt.figure(2)
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(NodeList[1,:],NodeList[2,:],Phi,
#                 cmap='viridis')
# 
# =============================================================================
