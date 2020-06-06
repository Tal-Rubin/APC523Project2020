# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:14:24 2020

@author: trubin
"""

import numpy as np
import Quadratic_Poly_Functions as QBF


def pickle(file,data):
    import pickle
    with open(file, 'wb') as fo:
        pickle.dump(data, fo)
def unpickle(file):
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




    


def sumcellav(rho,GlobalElementMatrix,EleVol,DGorder): #add element volume 
    s=0
    for k in range(GlobalElementMatrix.shape[0]):
        s+=rho[DGorder*k]*EleVol[k]
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


def assembleFluxMat(vx,vy,NumEle,DGorder,Connectivity,EleFluxMat,Bound_mat,LF_mat):
    FluxMat1=np.zeros((NumEle*DGorder,NumEle*DGorder))
    
    FluxMat2=np.zeros((NumEle*DGorder,NumEle*DGorder))
    
    FluxMat3=np.zeros((NumEle*DGorder,NumEle*DGorder))
    
    for k in range(NumEle):
        alpha=np.zeros(4)

        for i in range(DGorder):
            alpha[:]+=LF_mat[:,0,k,i]*vx[DGorder*k+i]-LF_mat[:,1,k,i]*vy[DGorder*k+i]
                   
            for j in range(DGorder):    
                for m in range(DGorder):
                    FluxMat2[DGorder*k+i,DGorder*k+j]+=EleFluxMat[0,k,i,j,m]*vx[DGorder*k+m]-EleFluxMat[1,k,i,j,m]*vy[DGorder*k+m]
                    
                    FluxMat1[DGorder*k+i,DGorder*k+j]+=0.5*(Bound_mat[0,0,k,i,j,m]*vx[DGorder*k+m]-Bound_mat[0,1,k,i,j,m]*vy[DGorder*k+m])
                    FluxMat1[DGorder*k+i,DGorder*k+j]+=0.5*(Bound_mat[1,0,k,i,j,m]*vx[DGorder*k+m]-Bound_mat[1,1,k,i,j,m]*vy[DGorder*k+m])
                    FluxMat1[DGorder*k+i,DGorder*k+j]+=0.5*(Bound_mat[2,0,k,i,j,m]*vx[DGorder*k+m]-Bound_mat[2,1,k,i,j,m]*vy[DGorder*k+m])
                    FluxMat1[DGorder*k+i,DGorder*k+j]+=0.5*(Bound_mat[3,0,k,i,j,m]*vx[DGorder*k+m]-Bound_mat[3,1,k,i,j,m]*vy[DGorder*k+m])
                    
                  
                    if Connectivity[k,0]!=-1:
                        FluxMat1[DGorder*Connectivity[k,0]+i,DGorder*k+j]-=0.5*(Bound_mat[2,0,k,i,j,m]*vx[DGorder*k+m]-Bound_mat[2,1,k,i,j,m]*vy[DGorder*k+m])
                            
                    if Connectivity[k,1]!=-1:
                        FluxMat1[DGorder*Connectivity[k,1]+i,DGorder*k+j]-=0.5*(Bound_mat[3,0,k,i,j,m]*vx[DGorder*k+m]-Bound_mat[3,1,k,i,j,m]*vy[DGorder*k+m])
                            
                    if Connectivity[k,2]!=-1:
                        FluxMat1[DGorder*Connectivity[k,2]+i,DGorder*k+j]-=0.5*(Bound_mat[0,0,k,i,j,m]*vx[DGorder*k+m]-Bound_mat[0,1,k,i,j,m]*vy[DGorder*k+m])
                    
                    if Connectivity[k,3]!=-1: 
                        FluxMat1[DGorder*Connectivity[k,3]+i,DGorder*k+j]-=0.5*(Bound_mat[1,0,k,i,j,m]*vx[DGorder*k+m]-Bound_mat[1,1,k,i,j,m]*vy[DGorder*k+m])
            
        alpha=abs(alpha)
              
        
        FluxMat3[DGorder*k,DGorder*k]-=0.5*sum(alpha)
                
        if Connectivity[k,0]!=-1:
            FluxMat3[DGorder*Connectivity[k,0],DGorder*k]+=0.5*(alpha[2])
            
        if Connectivity[k,1]!=-1:
            FluxMat3[DGorder*Connectivity[k,1],DGorder*k]+=0.5*(alpha[3])
            
        if Connectivity[k,2]!=-1:
            FluxMat3[DGorder*Connectivity[k,2],DGorder*k]+=0.5*(alpha[0])
                
        if Connectivity[k,3]!=-1: 
            FluxMat3[DGorder*Connectivity[k,3],DGorder*k]+=0.5*(alpha[1])
                                                
    
    return FluxMat1+FluxMat2+FluxMat3




def Solver(DT,Tend,Sample,rhoFE,vx,vy,Mesh,DGorder,CalcFE=True):
    
    
    
    GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,InvMassMat,EleFluxMat,Bound_mat,LF_mat,Connectivity,EleVol=Mesh

    chargeOverPermitt=1.8095128169876496e-08
    rho=projectFEintoDG(rhoFE,NodeList,GlobalElementMatrix,DGorder)
    rho0=rho

    #BC
    if CalcFE:
        GlobalStiffMat[BCnodes,:]=np.zeros((1,GlobalStiffMat.shape[1]))
        GlobalStiffMat[BCnodes,BCnodes]=1
    
    
    
    
    #time loop

        F=chargeOverPermitt*GlobalForceMat@rhoFE
        F[BCnodes]=0 

        Phi=np.linalg.solve(GlobalStiffMat,F)
    else: 
        Phi=None


    FluxMat=assembleFluxMat(vx,vy,GlobalElementMatrix.shape[0],DGorder,Connectivity,EleFluxMat,Bound_mat,LF_mat)
    
    
    


    pickle('./Data/Den0',(rho0,0))
    time=0
    i=0
    j=0
    
    
    while time<Tend:
        time+=DT
        i+=1
        #CFL=DT*np.maximum(FluxMat,0)@np.ones(rho.shape)/EleVol
        #rho_f1=np.maximum(np.minimum(rho,rho/(2*CFL)),0)
        
        rho1=rho+DT*InvMassMat@(FluxMat@rho)
        rho1=FluxLimit(rho1,DGorder) 
        
        #rho_f2=np.maximum(np.minimum(rho1,rho1/(2*CFL)),0)
       
        rho2=0.75*rho+0.25*rho1+0.25*DT*InvMassMat@(FluxMat@rho1)
        rho2=FluxLimit(rho2,DGorder)
    
        #rho_f3=np.maximum(np.minimum(rho2,rho2/(2*CFL)),0)
    
        rho3=1/3*rho+2/3*rho2+2/3*DT*InvMassMat@(FluxMat@rho2)
        rho3=FluxLimit(rho3,DGorder)
    
        rho=rho3
        if i==Sample:
            i=0
            j+=1
            pickle('./Data/Den'+str("%05d" % j),(rho,time))
    
    
    print('rho0 = ' +str(sumcellav(rho0,GlobalElementMatrix,EleVol,DGorder))+'\nrho = ' +str(sumcellav(rho,GlobalElementMatrix,EleVol,DGorder))+' \nDelta = '+str(sumcellav(rho0,GlobalElementMatrix,EleVol,DGorder)-sumcellav(rho,GlobalElementMatrix,EleVol,DGorder)))
    return Phi