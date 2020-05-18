# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:41:28 2020

@author: trubin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Quadratic_Poly_Functions as QBF
import polynomial2d as p2d

#use np.where for element fixing 
#and where element # is not the same as its index in the element matrix
#np.where(GlobalElementMatrix[:,0]==884)[0][0] 





def NodeArrange(N_row,N_col,XY,NodeBeginNum,ElemBeginNum):
    #splitting macro-element into elements
    LM=np.zeros((N_row*N_col,10),dtype='int')
    Connectivity=-np.ones((N_row*N_col,4),dtype='int')
    NodesList=np.zeros((3,(1+2*N_row)*(1+2*N_col)))    #node #, node xi, node eta
    for i in range((1+2*N_col)):
        NodesList[0,i*(1+2*N_row):(i+1)*(1+2*N_row)] = np.arange(i*(1+2*N_row),(i+1)*(1+2*N_row))+NodeBeginNum
        NodesList[1,i*(1+2*N_row):(i+1)*(1+2*N_row)] = -1  +  i/N_col
        NodesList[2,i*(1+2*N_row):(i+1)*(1+2*N_row)] = np.linspace(1, -1, (1+2*N_row))        
    #Moving the node to the global coordinate system
    NodesList[1:3,:] = XY@QBF.N(NodesList[1,:],NodesList[2,:])

    
    for i in range(N_col):
        for j in range(N_row):
            LM[j+i*N_row,0]  =   j+i*N_row+ElemBeginNum                     #element #
            LM[j+i*N_row,1]  =   2*j+i*(2+4*N_row)+2+NodeBeginNum           #Node 1, #    j+i*N_row+i+1
            LM[j+i*N_row,2]  =   2*j+(i+1)*(2+4*N_row)+2   +NodeBeginNum    #Node 2, #
            LM[j+i*N_row,3]  =   2*j+(i+1)*(2+4*N_row)  +NodeBeginNum       #Node 3, #
            LM[j+i*N_row,4]  =   2*j+i*(2+4*N_row)      +NodeBeginNum       #Node 4, #  j+i*N_row+i
            LM[j+i*N_row,5]  =   LM[j+i*N_row,4]+3+2*N_row  +NodeBeginNum   #Node 5, #   
            LM[j+i*N_row,6]  =   LM[j+i*N_row,3]+1 +NodeBeginNum            #Node 6, #
            LM[j+i*N_row,7]  =   LM[j+i*N_row,4]+1+2*N_row+NodeBeginNum     #Node 7, #
            LM[j+i*N_row,8]  =   LM[j+i*N_row,4]+1 +NodeBeginNum            #Node 8, #
            LM[j+i*N_row,9]  =   LM[j+i*N_row,4]+2+2*N_row  +NodeBeginNum   #Node 9, #
            if 1+j+i*N_row<N_row*N_col and j!=N_row-1:
                Connectivity[j+i*N_row,0]=1+j+i*N_row
            if i!=N_col-1:
                Connectivity[j+i*N_row,1]=j+(i+1)*N_row
            if j!=0:
                Connectivity[j+i*N_row,2]=j+i*N_row-1
            if i!=0:
                Connectivity[j+i*N_row,3]=j+(i-1)*N_row
# =============================================================================
#             #fluxes
#             LM[j+i*N_row,10]  =0
#             LM[j+i*N_row,11]  =0  
#             LM[j+i*N_row,12]  =0     
#             LM[j+i*N_row,13]  =0 
#             LM[j+i*N_row,14]  =0 
#             LM[j+i*N_row,15]  =0     
#             LM[j+i*N_row,16]  =0       
#             LM[j+i*N_row,17]  =0  
# =============================================================================
            
    return LM,NodesList,Connectivity

def detJ(xi,eta,XY):
    return np.linalg.det(XY@QBF.B(xi, eta))

def invJ(XY):
    return np.array([[(XY@QBF.Ba)[1,1],-(XY@QBF.Ba)[0,1]],[-(XY@QBF.Ba)[1,0],(XY@QBF.Ba)[0,0]]])

def CheckDetJ(XY,fineness=10):
    xp=np.linspace(-1,1,num=fineness)
    yp=np.linspace(-1,1,num=fineness)

    Xp, Yp = np.meshgrid(xp, yp)
    for i in range(len(Xp)):
        for j in range(len(Yp)):
            if detJ(Xp[i,j], Yp[i,j],XY)<0:
                print("Bad Macro Element Coordinates")
                return 0
    print("Good Macro Element Coordinates")
    return 1


def initMasterFluxMat(DGorder):   
    Gamma_mast=np.zeros((DGorder,DGorder,9,DGorder))
    for i in range(DGorder):
        for j in range(DGorder):
            for n in range(9):
                for m in range(DGorder):
                    temp=QBF.Ma[j]*QBF.Ma[m]*(QBF.Ma[i].dx()*QBF.Na[n].dy()-QBF.Ma[i].dy()*QBF.Na[n].dx())
                    temp=temp.intx(-1,1)
                    temp=temp.inty(-1,1)
                    Gamma_mast[i,j,n,m]=temp.scalar()
    return Gamma_mast

def initMasterForcingMatrix():   
    f_mast=np.zeros((9,9,9,9))
    for i in range(9):
        for j in range(9):
            for n in range(9):
                for m in range(9):
                    temp=(QBF.Na[i]*QBF.Na[j])*(QBF.Na[n].dx()*QBF.Na[m].dy())
                    temp=temp.intx(-1,1)
                    temp=temp.inty(-1,1)
                    f_mast[i,j,n,m]=temp.scalar()
    return f_mast

def initMasterMassMatrix(DGorder):   
    f_mast=np.zeros((DGorder,DGorder,9,9))
    for i in range(DGorder):
        for j in range(DGorder):
            for n in range(9):
                for m in range(9):
                    temp=(QBF.Ma[i]*QBF.Ma[j])*(QBF.Na[n].dx()*QBF.Na[m].dy())
                    temp=temp.intx(-1,1)
                    temp=temp.inty(-1,1)
                    f_mast[i,j,n,m]=temp.scalar()
    return f_mast

def initMasterBoundMatrix(DGorder):   
    b_mast_1=np.zeros((DGorder,DGorder,9,DGorder))
    b_mast_2=np.zeros((DGorder,DGorder,9,DGorder))
    b_mast_3=np.zeros((DGorder,DGorder,9,DGorder))
    b_mast_4=np.zeros((DGorder,DGorder,9,DGorder))
    for i in range(DGorder):
        for j in range(DGorder):
            for n in range(9):
                for m in range(DGorder):
                    temp13=(QBF.Ma[i]*QBF.Ma[j])*QBF.Na[n].dx()*QBF.Ma[m]
                    temp24=(QBF.Ma[i]*QBF.Ma[j])*QBF.Na[n].dy()*QBF.Ma[m]

                    temp1=temp13(y=-1)
                    temp2=temp24(x=1)
                    temp3=temp13(y=1)
                    temp4=temp24(x=-1)

                    temp1=temp1.intx(-1,1)
                    temp2=temp2.inty(-1,1)
                    temp3=temp3.intx(-1,1)
                    temp4=temp4.inty(-1,1)
                    
                    b_mast_1[i,j,n,m]=temp1.scalar()
                    b_mast_2[i,j,n,m]=temp2.scalar()
                    b_mast_3[i,j,n,m]=-temp3.scalar()
                    b_mast_4[i,j,n,m]=-temp4.scalar()
    return b_mast_1,b_mast_2,b_mast_3,b_mast_4


def initMasterLaxFriedrichs(DGorder):   
    LF_mast_1=np.zeros((9,DGorder))
    LF_mast_2=np.zeros((9,DGorder))
    LF_mast_3=np.zeros((9,DGorder))
    LF_mast_4=np.zeros((9,DGorder))

    for n in range(9):
        for m in range(DGorder):
            temp13=QBF.Na[n].dx()*QBF.Ma[m]
            temp24=QBF.Na[n].dy()*QBF.Ma[m]

            temp1=temp13(y=-1)
            temp2=temp24(x=1)
            temp3=temp13(y=1)
            temp4=temp24(x=-1)

            temp1=temp1.intx(-1,1)
            temp2=temp2.inty(-1,1)
            temp3=temp3.intx(-1,1)
            temp4=temp4.inty(-1,1)
            
            LF_mast_1[n,m]=temp1.scalar()
            LF_mast_2[n,m]=temp2.scalar()
            LF_mast_3[n,m]=-temp3.scalar()
            LF_mast_4[n,m]=-temp4.scalar()
    return LF_mast_1,LF_mast_2,LF_mast_3,LF_mast_4



def assembleGlobalMats(GlobalElementMatrix,NodeList,order=6):
   # k_mast=initMasterStiffnessMat()
    f_mast=initMasterForcingMatrix()
    GlobalStiffMat=np.zeros((NodeList.shape[1],NodeList.shape[1]))
    GlobalForceMat=np.zeros((NodeList.shape[1],NodeList.shape[1]))
    XYele=np.zeros((2,9))
    x_i=np.array([-0.932469514203152,-0.661209386466265,-0.238619186083197,0.238619186083197,0.661209386466265,0.932469514203152])
    w_i=np.array([0.171324492379170,0.360761573048139,0.467913934572691,0.467913934572691,0.360761573048139,0.171324492379170])
    if order==5:
        x_i=np.array([-np.sqrt(5+2*np.sqrt(10/7))/3,-np.sqrt(5-2*np.sqrt(10/7))/3,0,np.sqrt(5-2*np.sqrt(10/7))/3,np.sqrt(5+2*np.sqrt(10/7))/3])
        w_i=np.array([(322-13*np.sqrt(70))/900,(322+13*np.sqrt(70))/900,128/225,(322+13*np.sqrt(70))/900,(322-13*np.sqrt(70))/900])        
  
    for k in range(GlobalElementMatrix.shape[0]):
        XYele = NodeList[1:3,GlobalElementMatrix[k,1:10]]      
        NodeXYantySymOuter=np.outer(XYele[0,:],XYele[1,:])-np.outer(XYele[1,:],XYele[0,:])
        EleStiffMat=np.zeros((9,9))
        EleForceMat=np.zeros((9,9))      
        xi_x=(XYele[1,:])@(QBF.Ba[:,1])
        xi_y=-(XYele[0,:])@(QBF.Ba[:,1])
        eta_x=-(XYele[1,:])@(QBF.Ba[:,0])
        eta_y=(XYele[0,:])@(QBF.Ba[:,0])
        for i in range(9):
            for j in range(9):
                kij=QBF.Na[i].dx()*QBF.Na[j].dx()*(xi_x*xi_x+xi_y*xi_y)+QBF.Na[i].dy()*QBF.Na[j].dy()*(eta_x*eta_x+eta_y*eta_y)+(QBF.Na[i].dx()*QBF.Na[j].dy()+QBF.Na[i].dy()*QBF.Na[j].dx())*(xi_x*eta_x+xi_y*eta_y)
                for n in range(len(x_i)):
                    for m in range(len(x_i)):
                        EleStiffMat[i,j]+=w_i[n]*w_i[m]*kij(x_i[n],x_i[m])/detJ(x_i[n],x_i[m],XYele)
        for n in range(9):
            for m in range(9):
                EleForceMat+=f_mast[:,:,n,m]*NodeXYantySymOuter[n,m]
        for n in range(9):
            for m in range(9):
                GlobalStiffMat[GlobalElementMatrix[k,1+n],GlobalElementMatrix[k,1+m]]+=EleStiffMat[n,m]
                GlobalForceMat[GlobalElementMatrix[k,1+n],GlobalElementMatrix[k,1+m]]+=EleForceMat[n,m]
    return GlobalStiffMat,GlobalForceMat

def assembleElementMats(GlobalElementMatrix,NodeList,DGorder):
    Gamma_mast=initMasterFluxMat(DGorder)
    Mass_mast=initMasterMassMatrix(DGorder)
    b_mast_1,b_mast_2,b_mast_3,b_mast_4=initMasterBoundMatrix(DGorder)
    LF_mast_1,LF_mast_2,LF_mast_3,LF_mast_4=initMasterLaxFriedrichs(DGorder)
    
    EleFluxMatx=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))
    EleFluxMaty=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))
    InvMassMat=np.zeros((GlobalElementMatrix.shape[0]*DGorder,GlobalElementMatrix.shape[0]*DGorder))
    
    b_mast_1x=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))
    b_mast_1y=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))

    b_mast_2x=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))
    b_mast_2y=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))
    
    b_mast_3x=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))
    b_mast_3y=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))
    
    b_mast_4x=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))
    b_mast_4y=np.zeros((GlobalElementMatrix.shape[0],DGorder,DGorder,DGorder))
    
    
    LF_mast_1x=np.zeros((GlobalElementMatrix.shape[0],DGorder))
    LF_mast_1y=np.zeros((GlobalElementMatrix.shape[0],DGorder))

    LF_mast_2x=np.zeros((GlobalElementMatrix.shape[0],DGorder))
    LF_mast_2y=np.zeros((GlobalElementMatrix.shape[0],DGorder))
    
    LF_mast_3x=np.zeros((GlobalElementMatrix.shape[0],DGorder))
    LF_mast_3y=np.zeros((GlobalElementMatrix.shape[0],DGorder))
    
    LF_mast_4x=np.zeros((GlobalElementMatrix.shape[0],DGorder))
    LF_mast_4y=np.zeros((GlobalElementMatrix.shape[0],DGorder))

    for k in range(GlobalElementMatrix.shape[0]):
        XYele = NodeList[1:3,GlobalElementMatrix[k,1:10]]      
        NodeXYantySymOuter=np.outer(XYele[0,:],XYele[1,:])-np.outer(XYele[1,:],XYele[0,:])
   
        for n in range(9):
            for m in range(9):
                    InvMassMat[DGorder*k:DGorder*(k+1),DGorder*k:DGorder*(k+1)]+=Mass_mast[:,:,n,m]*NodeXYantySymOuter[n,m]
            
            EleFluxMatx[k,:,:,:]+=Gamma_mast[:,:,n,:]*NodeList[2,GlobalElementMatrix[k,n+1]]
            EleFluxMaty[k,:,:,:]-=Gamma_mast[:,:,n,:]*NodeList[1,GlobalElementMatrix[k,n+1]]
            
            b_mast_1x[k,:,:,:]+=b_mast_1[:,:,n,:]*NodeList[2,GlobalElementMatrix[k,1+n]]
            b_mast_1y[k,:,:,:]+=b_mast_1[:,:,n,:]*NodeList[1,GlobalElementMatrix[k,1+n]]

            b_mast_2x[k,:,:,:]+=b_mast_2[:,:,n,:]*NodeList[2,GlobalElementMatrix[k,1+n]]
            b_mast_2y[k,:,:,:]+=b_mast_2[:,:,n,:]*NodeList[1,GlobalElementMatrix[k,1+n]]
    
            b_mast_3x[k,:,:,:]+=b_mast_3[:,:,n,:]*NodeList[2,GlobalElementMatrix[k,1+n]]
            b_mast_3y[k,:,:,:]+=b_mast_3[:,:,n,:]*NodeList[1,GlobalElementMatrix[k,1+n]]
    
            b_mast_4x[k,:,:,:]+=b_mast_4[:,:,n,:]*NodeList[2,GlobalElementMatrix[k,1+n]]
            b_mast_4y[k,:,:,:]+=b_mast_4[:,:,n,:]*NodeList[1,GlobalElementMatrix[k,1+n]]
            
            LF_mast_1x[k,:]+=LF_mast_1[n,:]*NodeList[2,GlobalElementMatrix[k,1+n]]
            LF_mast_1y[k,:]+=LF_mast_1[n,:]*NodeList[1,GlobalElementMatrix[k,1+n]]

            LF_mast_2x[k,:]+=LF_mast_2[n,:]*NodeList[2,GlobalElementMatrix[k,1+n]]
            LF_mast_2y[k,:]+=LF_mast_2[n,:]*NodeList[1,GlobalElementMatrix[k,1+n]]
    
            LF_mast_3x[k,:]+=LF_mast_3[n,:]*NodeList[2,GlobalElementMatrix[k,1+n]]
            LF_mast_3y[k,:]+=LF_mast_3[n,:]*NodeList[1,GlobalElementMatrix[k,1+n]]
        
            LF_mast_4x[k,:]+=LF_mast_4[n,:]*NodeList[2,GlobalElementMatrix[k,1+n]]
            LF_mast_4y[k,:]+=LF_mast_4[n,:]*NodeList[1,GlobalElementMatrix[k,1+n]]
            
            
        InvMassMat[DGorder*k:DGorder*(k+1),DGorder*k:DGorder*(k+1)]=np.linalg.inv(InvMassMat[DGorder*k:DGorder*(k+1),DGorder*k:DGorder*(k+1)])
    return EleFluxMatx,EleFluxMaty,InvMassMat,b_mast_1x,b_mast_1y,b_mast_2x,b_mast_2y,b_mast_3x,b_mast_3y,b_mast_4x,b_mast_4y,LF_mast_1x,LF_mast_1y,LF_mast_2x,LF_mast_2y,LF_mast_3x,LF_mast_3y,LF_mast_4x,LF_mast_4y


def calcEleDims(GlobalElementMatrix,NodeList):
    EleDims=np.zeros(GlobalElementMatrix.shape[0]) #area, arc lengths
    for k in range(GlobalElementMatrix.shape[0]):
        XYele = NodeList[1:3,GlobalElementMatrix[k,1:10]]      
        NodeXYantySymOuter=np.outer(XYele[0,:],XYele[1,:])-np.outer(XYele[1,:],XYele[0,:])
        temp0=0
        for n in range(9):
            for m in range(9):
                temp0+=(QBF.Na[n].dx()*QBF.Na[m].dy())*NodeXYantySymOuter[n,m]
        temp0=temp0.intx(-1,1)
        temp0=temp0.inty(-1,1)
        EleDims[k]=temp0.scalar()
    return EleDims

    #PreProcessor
def PreProc(XY,N_row=10,N_col=10,DGorder=3,Qorder=6,ElemBeginNum=0,NodeBeginNum=0):

    



    if CheckDetJ(XY):
        LocalMatrix1,NodeList,Connectivity=NodeArrange(N_row,N_col,XY,NodeBeginNum,ElemBeginNum)
        GlobalElementMatrix=np.array(LocalMatrix1)
        print('Global Element Matrix generated')
        #plot Node Locations
       
        GlobalStiffMat,GlobalForceMat=assembleGlobalMats(GlobalElementMatrix,NodeList,Qorder)
        print('Global Stiffness and Force Matrix generated')

        EleFluxMatx,EleFluxMaty,InvMassMat,b_mast_1x,b_mast_1y,b_mast_2x,b_mast_2y,b_mast_3x,b_mast_3y,b_mast_4x,b_mast_4y,LF_mast_1x,LF_mast_1y,LF_mast_2x,LF_mast_2y,LF_mast_3x,LF_mast_3y,LF_mast_4x,LF_mast_4y=assembleElementMats(GlobalElementMatrix,NodeList,DGorder)
        
        EleVol=calcEleDims(GlobalElementMatrix,NodeList)

        BCnodes=np.array([*np.arange(0,2*N_row+1), 
                  *np.arange(1+2*N_row,1+2*N_row+(N_col)*(2+4*N_row),(2+4*N_row)),
                  *np.arange(2+4*N_row,2+4*N_row+(N_col)*(2+4*N_row),(2+4*N_row)),
                  *np.arange(2*(N_row-1)+3+2*N_row,2*(N_row-1)+(N_col)*(2+4*N_row)+3+2*N_row,(2+4*N_row)),
                  *np.arange(2*(N_row-1)+4+4*N_row,2*(N_row-1)+(N_col)*(2+4*N_row)+4+4*N_row,(2+4*N_row)),
                  *np.arange((N_col)*(2+4*N_row)-1,2*N_row+(N_col)*(2+4*N_row))
                  ])
        return GlobalStiffMat,GlobalForceMat,GlobalElementMatrix,NodeList,BCnodes,EleFluxMatx,EleFluxMaty,InvMassMat,b_mast_1x,b_mast_1y,b_mast_2x,b_mast_2y,b_mast_3x,b_mast_3y,b_mast_4x,b_mast_4y,LF_mast_1x,LF_mast_1y,LF_mast_2x,LF_mast_2y,LF_mast_3x,LF_mast_3y,LF_mast_4x,LF_mast_4y,Connectivity,EleVol
    else: 
        return 0
    



# =============================================================================
# GlobalStiffMatInv=np.linalg.inv(GlobalStiffMat)
# plt.figure(124)
# plt.imshow(GlobalStiffMatInv)
# 
# =============================================================================

#electric field 
#Ex= QBF.N1.dx()/((QBF.Ba@XY)[0,0])+QBF.N1.dy()/((QBF.Ba@XY)[0,1])
#Ey= QBF.N1.dx()/((QBF.Ba@XY)[1,0])+QBF.N1.dy()/((QBF.Ba@XY)[1,1])




# =============================================================================
# plt.figure(7)
# plt.plot(detelem00,color='r')
# plt.plot(detelempp,color='g')
# plt.plot(detelemmm,color='b')
# plt.plot(detelempm,color='y')
# plt.plot(detelemmp,color='m')
# =============================================================================
def plotting(XY):
    xp=np.linspace(-1,1)
    yp=np.linspace(-1,1)
    
    Xp, Yp = np.meshgrid(xp, yp)

    xp=np.linspace(-1,1)
    yp=np.linspace(-1,1)
    
    Xp, Yp = np.meshgrid(xp, yp)

    Jgrid=np.zeros((len(xp),len(xp)))
    for i in range(len(Xp)):
        for j in range(len(Yp)):
            Jgrid[i,j]=detJ(Xp[i,j], Yp[i,j],XY)
    
    
    plt.figure(4)
    ax = plt.axes(projection='3d')
    ax.plot_surface(Xp,Yp,Jgrid)
    ax.plot_surface(Xp,Yp,np.zeros((len(xp),len(xp))))
    
    
# =============================================================================
#     ax = plt.axes(projection='3d')
#     
#     ax.plot_surface(Xp,Yp,QBF.N9(Xp, Yp))
#     ax.plot_surface(Xp,Yp,QBF.N8(Xp, Yp))
#     ax.plot_surface(Xp,Yp,QBF.N7(Xp, Yp))
#     ax.plot_surface(Xp,Yp,QBF.N6(Xp, Yp))
#     ax.plot_surface(Xp,Yp,QBF.N5(Xp, Yp))
#     ax.plot_surface(Xp,Yp,QBF.N4(Xp, Yp))
#     ax.plot_surface(Xp,Yp,QBF.N3(Xp, Yp))
#     ax.plot_surface(Xp,Yp,QBF.N2(Xp, Yp))
#     ax.plot_surface(Xp,Yp,QBF.N1(Xp, Yp))
# =============================================================================
    
    
# =============================================================================
#     fig = plt.figure(2)
#     ax = plt.axes(projection='3d')
#     
#     ax.plot_surface(Xp,Yp,N9_1(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N8_1(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N7_1(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N6_1(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N5_1(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N4_1(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N3_1(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N2_1(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N1_1(Xp, Yp))
#     
#     
#     fig = plt.figure(3)
#     ax = plt.axes(projection='3d')
#     
#     ax.plot_surface(Xp,Yp,N9_2(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N8_2(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N7_2(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N6_2(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N5_2(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N4_2(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N3_2(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N2_2(Xp, Yp))
#     ax.plot_surface(Xp,Yp,N1_2(Xp, Yp))
#     
# =============================================================================