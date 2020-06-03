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

def FluxLimit(rho,DGorder):
    if DGorder>3:
        for k in range(int(len(rho)/DGorder)):
            rho[DGorder*k+1]=max(-3*rho[DGorder*k],min(3*rho[DGorder*k],rho[DGorder*k+1]))
            rho[DGorder*k+2]=max(-3*rho[DGorder*k],min(3*rho[DGorder*k],rho[DGorder*k+2]))
    return rho


def CenterDist(k1,k2,GlobalElementMatrix,NodeList):
    return np.sqrt((NodeList[1,GlobalElementMatrix[k1,9]]-NodeList[1,GlobalElementMatrix[k2,9]])**2+(NodeList[2,GlobalElementMatrix[k1,9]]-NodeList[2,GlobalElementMatrix[k2,9]])**2)

def TorX(N_row,Connectivity):
    for k in range(N_row):
        Connectivity[k,3]=N_row*(N_col-1)+k
        Connectivity[N_row*(N_col-1)+k,1]=k
    return Connectivity

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




new_mesh=True
save_mesh=False
load_mesh=False


XDom=10
YDom=10

X=XDom*(np.array([0, 1, 1, 0, 0.5 , 1   ,0.5 ,0   ,0.5],dtype=np.float64)-0.5)
Y=YDom*np.array([0, 0, 1, 1, 0   , 0.5 ,1   ,0.5 ,0.5],dtype=np.float64)


N_row=5
N_col=15

DGorder=1


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
    GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,EleFluxMatx,EleFluxMaty,InvMassMat,b_mast_1x,b_mast_1y,b_mast_2x,b_mast_2y,b_mast_3x,b_mast_3y,b_mast_4x,b_mast_4y,LF_mast_1x,LF_mast_1y,LF_mast_2x,LF_mast_2y,LF_mast_3x,LF_mast_3y,LF_mast_4x,LF_mast_4y,Connectivity,EleVol=PreProc.PreProc(XY,N_row,N_col,DGorder,5)    
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
    


#pretzel x
Connectivity=TorX(N_row,Connectivity)




    
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


rho=projectFEintoDG(setInitDenDistFE(0,5,0.5,1,np.zeros(NodeList.shape[1]),NodeList,GlobalElementMatrix),NodeList,GlobalElementMatrix,DGorder)



# =============================================================================
# for i in range(rho.shape[0]):
#     for j in range(rho.shape[1]-1):
#         rho[i,j+1]=1
# =============================================================================

#rho,circ1=setInitDenDist(0.625,0.5,0.2,1,rho,NodeList,GlobalElementMatrix)
#rho,circ2=setInitDenDist(1.325,0.5,0.2,1,rho,NodeList,GlobalElementMatrix)

# =============================================================================
# 
# rho,circ1=setInitDenDist(0,5,1,1,rho,NodeList,GlobalElementMatrix,DGorder)
# #rho,circ2=setInitDenDist(0.8,2,0.3,1,rho,NodeList,GlobalElementMatrix)
# #rho,circ3=setInitDenDist(-0.8,0.6,0.3,2,rho,NodeList,GlobalElementMatrix)
# #rho,circ4=setInitDenDist(0.8,0.6,0.3,-2,rho,NodeList,GlobalElementMatrix)
# 
# 
# plt.figure(1)
# plt.plot(NodeList[1,:],NodeList[2,:],'.')
# ax = plt.gca()
# ax.add_artist(circ1)
# #ax.add_artist(circ2)
# =============================================================================
#ax.add_artist(circ3)
#ax.add_artist(circ4)








# =============================================================================
# plt.figure(3)
# plt.title('FE boundary condition nodes')
# plt.plot(NodeList[1,BCnodes],NodeList[2,BCnodes],'*')
# =============================================================================
rho0=rho

#time loop
rhoFE=setInitDenDistFE(0,5,0.5,1,np.zeros(NodeList.shape[1]),NodeList,GlobalElementMatrix)



#rhoFE=setInitDenDistFE(2,6.5,0.8,1,np.zeros(NodeList.shape[1]),NodeList,GlobalElementMatrix)
#rhoFE=setInitDenDistFE(-2,6.5,0.8,-1,rhoFE,NodeList,GlobalElementMatrix)

F=chargeOverPermitt*GlobalForceMat@rhoFE

#BC
F[BCnodes]=0 
GlobalStiffMat[BCnodes,:]=np.zeros((1,GlobalStiffMat.shape[1]))
GlobalStiffMat[BCnodes,BCnodes]=1


Phi=np.linalg.solve(GlobalStiffMat,F)
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



FluxMat=np.zeros((GlobalElementMatrix.shape[0]*DGorder,GlobalElementMatrix.shape[0]*DGorder))
FluxMat1=np.zeros((GlobalElementMatrix.shape[0]*DGorder,GlobalElementMatrix.shape[0]*DGorder))

FluxMat2=np.zeros((GlobalElementMatrix.shape[0]*DGorder,GlobalElementMatrix.shape[0]*DGorder))

FluxMat3=np.zeros((GlobalElementMatrix.shape[0]*DGorder,GlobalElementMatrix.shape[0]*DGorder))

for k in range(GlobalElementMatrix.shape[0]):
    XYele = NodeList[1:3,GlobalElementMatrix[k,1:10]]
    alpha1=0
    alpha2=0
    alpha3=0
    alpha4=0
    for i in range(DGorder):
        alpha1+=LF_mast_1x[k,i]*vx[DGorder*k+i]-LF_mast_1y[k,i]*vy[DGorder*k+i]
        alpha2+=LF_mast_2x[k,i]*vx[DGorder*k+i]-LF_mast_2y[k,i]*vy[DGorder*k+i]
        alpha3+=LF_mast_3x[k,i]*vx[DGorder*k+i]-LF_mast_3y[k,i]*vy[DGorder*k+i]
        alpha4+=LF_mast_4x[k,i]*vx[DGorder*k+i]-LF_mast_4y[k,i]*vy[DGorder*k+i]
        
        for j in range(DGorder):    
            for m in range(DGorder):
                FluxMat2[DGorder*k+i,DGorder*k+j]+=EleFluxMatx[k,i,j,m]*vx[DGorder*k+m]+EleFluxMaty[k,i,j,m]*vy[DGorder*k+m]
                
                FluxMat1[DGorder*k+i,DGorder*k+j]+=0.5*(b_mast_1x[k,i,j,m]*vx[DGorder*k+m]-b_mast_1y[k,i,j,m]*vy[DGorder*k+m])
                FluxMat1[DGorder*k+i,DGorder*k+j]+=0.5*(b_mast_2x[k,i,j,m]*vx[DGorder*k+m]-b_mast_2y[k,i,j,m]*vy[DGorder*k+m])
                FluxMat1[DGorder*k+i,DGorder*k+j]+=0.5*(b_mast_3x[k,i,j,m]*vx[DGorder*k+m]-b_mast_3y[k,i,j,m]*vy[DGorder*k+m])
                FluxMat1[DGorder*k+i,DGorder*k+j]+=0.5*(b_mast_4x[k,i,j,m]*vx[DGorder*k+m]-b_mast_4y[k,i,j,m]*vy[DGorder*k+m])
                


                
                if Connectivity[k,0]!=-1:
                    FluxMat1[DGorder*Connectivity[k,0]+i,DGorder*k+j]-=0.5*(b_mast_3x[k,i,j,m]*vx[DGorder*k+m]-b_mast_3y[k,i,j,m]*vy[DGorder*k+m])
                        
                if Connectivity[k,1]!=-1:
                    FluxMat1[DGorder*Connectivity[k,1]+i,DGorder*k+j]-=0.5*(b_mast_4x[k,i,j,m]*vx[DGorder*k+m]-b_mast_4y[k,i,j,m]*vy[DGorder*k+m])
                        
                if Connectivity[k,2]!=-1:
                    FluxMat1[DGorder*Connectivity[k,2]+i,DGorder*k+j]-=0.5*(b_mast_1x[k,i,j,m]*vx[DGorder*k+m]-b_mast_1y[k,i,j,m]*vy[DGorder*k+m])
                
                if Connectivity[k,3]!=-1: 
                    FluxMat1[DGorder*Connectivity[k,3]+i,DGorder*k+j]-=0.5*(b_mast_2x[k,i,j,m]*vx[DGorder*k+m]-b_mast_2y[k,i,j,m]*vy[DGorder*k+m])
        
    alpha1=abs(alpha1)
    alpha2=abs(alpha2)
    alpha3=abs(alpha3)        
    alpha4=abs(alpha4)      
    
    FluxMat3[DGorder*k,DGorder*k]-=0.5*(alpha1)
    FluxMat3[DGorder*k,DGorder*k]-=0.5*(alpha2)
    FluxMat3[DGorder*k,DGorder*k]-=0.5*(alpha3)
    FluxMat3[DGorder*k,DGorder*k]-=0.5*(alpha4)
            


            
    if Connectivity[k,0]!=-1:
        FluxMat3[DGorder*Connectivity[k,0],DGorder*k]+=0.5*(alpha3)
        
    if Connectivity[k,1]!=-1:
        FluxMat3[DGorder*Connectivity[k,1],DGorder*k]+=0.5*(alpha4)
        
    if Connectivity[k,2]!=-1:
        FluxMat3[DGorder*Connectivity[k,2],DGorder*k]+=0.5*(alpha1)
            
    if Connectivity[k,3]!=-1: 
        FluxMat3[DGorder*Connectivity[k,3],DGorder*k]+=0.5*(alpha2)
                                            
# =============================================================================
# plt.figure(15)
# plt.imshow(FluxMat1)
# 
# plt.figure(16)
# plt.imshow(FluxMat2)
# 
# plt.figure(17)
# plt.imshow(FluxMat3)
# =============================================================================

FluxMat=FluxMat1+FluxMat2+FluxMat3

plt.figure(18)
plt.imshow(FluxMat)

rho=rho0
plotDGelem(rho0, GlobalElementMatrix, NodeList, 0, DGorder,3)

DT=0.1
time=0
while time<XDom:
    time+=DT
    CFL=DT*np.maximum(FluxMat,0)@np.ones(rho.shape)/EleVol
    rho_f1=np.maximum(np.minimum(rho,rho/(2*CFL)),0)
    
    rho1=rho+DT*InvMassMat@(FluxMat@rho)
    rho1=FluxLimit(rho1,DGorder) 
    
    rho_f2=np.maximum(np.minimum(rho1,rho1/(2*CFL)),0)
   
    rho2=0.75*rho+0.25*rho1+0.25*DT*InvMassMat@(FluxMat@rho1)
    rho2=FluxLimit(rho2,DGorder)

    rho_f3=np.maximum(np.minimum(rho2,rho2/(2*CFL)),0)

    rho3=1/3*rho+2/3*rho2+2/3*DT*InvMassMat@(FluxMat@rho2)
    rho3=FluxLimit(rho3,DGorder)

# =============================================================================
#     for k in range(GlobalElementMatrix.shape[0]):
#         if Connectivity[k,1]==-1 or Connectivity[k,3]==-1 or Connectivity[k,0]==-1 or Connectivity[k,2]==-1:
#             rho3[DGorder*k:DGorder*(k+1)]=rho0[DGorder*k:DGorder*(k+1)]
# =============================================================================        
    rho=rho3
    
rhoRD1=rho   
plotDGelem(rhoRD1, GlobalElementMatrix, NodeList, 1, DGorder,3)    
    
    
    
while time<2*XDom:
    time+=DT
    CFL=DT*np.maximum(FluxMat,0)@np.ones(rho.shape)/EleVol
    rho_f1=np.maximum(np.minimum(rho,rho/(4*CFL)),0)
    
    rho1=rho+DT*InvMassMat@(FluxMat@rho_f1)
    rho1=FluxLimit(rho1,DGorder) 
    
    rho_f2=np.maximum(np.minimum(rho1,rho1/(4*CFL)),0)
   
    rho2=0.75*rho+0.25*rho1+0.25*DT*InvMassMat@(FluxMat@rho_f2)
    rho2=FluxLimit(rho2,DGorder)

    rho_f3=np.maximum(np.minimum(rho2,rho2/(4*CFL)),0)

    rho3=1/3*rho+2/3*rho2+2/3*DT*InvMassMat@(FluxMat@rho_f3)
    rho3=FluxLimit(rho3,DGorder)      
    rho=rho3

rhoRD2=rho
plotDGelem(rhoRD2, GlobalElementMatrix, NodeList, 2, DGorder,3)

while time<3*XDom:
    time+=DT
    CFL=DT*np.maximum(FluxMat,0)@np.ones(rho.shape)/EleVol
    rho_f1=np.maximum(np.minimum(rho,rho/(4*CFL)),0)
    
    rho1=rho+DT*InvMassMat@(FluxMat@rho_f1)
    rho1=FluxLimit(rho1,DGorder) 
    
    rho_f2=np.maximum(np.minimum(rho1,rho1/(4*CFL)),0)
   
    rho2=0.75*rho+0.25*rho1+0.25*DT*InvMassMat@(FluxMat@rho_f2)
    rho2=FluxLimit(rho2,DGorder)

    rho_f3=np.maximum(np.minimum(rho2,rho2/(4*CFL)),0)

    rho3=1/3*rho+2/3*rho2+2/3*DT*InvMassMat@(FluxMat@rho_f3)
    rho3=FluxLimit(rho3,DGorder)
  
    rho=rho3
    
rhoRD3=rho
plotDGelem(rhoRD3, GlobalElementMatrix, NodeList, 3, DGorder,3)


########   How to deal with corners? in regular grid with v=(1,1) element 1,1 wouldnt get the right signal
#plt.imshow(GlobalStiffMat)
# =============================================================================
# 
# plotDGelem(rho, GlobalElementMatrix, NodeList, 5, DGorder,3)
# plotDGelem(rho0, GlobalElementMatrix, NodeList, 6, DGorder,3)
# 
# =============================================================================
#200 timesteps



print('rho0 = ' +str(sumcellav(rho0,GlobalElementMatrix,DGorder))+'\nrho = ' +str(sumcellav(rho,GlobalElementMatrix,DGorder))+' \nDelta = '+str(sumcellav(rho0,GlobalElementMatrix,DGorder)-sumcellav(rho,GlobalElementMatrix,DGorder)))
