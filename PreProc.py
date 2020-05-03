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



def NodeArrange(N_row,N_col,XY,NodeBeginNum,ElemBeginNum):
    #splitting macro-element into elements
    LM=np.zeros((N_row*N_col,10),dtype='int')
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
            LM[j+i*N_row,5]  =   LM[j+i*N_row,4]+3+2*N_row  +NodeBeginNum  #Node 5, #   
            LM[j+i*N_row,6]  =   LM[j+i*N_row,3]+1 +NodeBeginNum            #Node 6, #
            LM[j+i*N_row,7]  =   LM[j+i*N_row,4]+1+2*N_row+NodeBeginNum    #Node 7, #
            LM[j+i*N_row,8]  =   LM[j+i*N_row,4]+1 +NodeBeginNum           #Node 8, #
            LM[j+i*N_row,9]  =   LM[j+i*N_row,4]+2+2*N_row  +NodeBeginNum  #Node 9, #
            
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
            
    return LM,NodesList

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
    return 1

def initMasterStiffnessMat():   
    k_mast=np.zeros((9,9,9,9))
    for i in range(9):
        for j in range(9):
            for n in range(9):
                for m in range(9):
                    temp=(QBF.Na[i].dx()*QBF.Na[j].dx()+QBF.Na[i].dy()*QBF.Na[j].dy())*(QBF.Na[n].dx()*QBF.Na[m].dy())
                    temp=temp.intx(-1,1)
                    temp=temp.inty(-1,1)
                    k_mast[i,j,n,m]=temp.scalar()
    return k_mast

def assembleGlobalStiffMat(GlobalElementMatrix,NodeList):
    k_mast=initMasterStiffnessMat()
    GlobalStiffMat=np.zeros((NodeList.shape[1],NodeList.shape[1]))
    XYele=np.zeros((2,9))
    for k in range(GlobalElementMatrix.shape[0]):
        XYele = NodeList[1:3,GlobalElementMatrix[k,1:10]]      
        NodeXYantySymOuter=np.outer(XYele[0,:],XYele[1,:])-np.outer(XYele[1,:],XYele[0,:])
        EleStiffMat=np.zeros((9,9))
        for i in range(9):
            for j in range(9):
                for n in range(9):
                    for m in range(9):
                        EleStiffMat[i,j]+=k_mast[i,j,n,m]*NodeXYantySymOuter[n,m]
        for n in range(9):
            for m in range(9):
                GlobalStiffMat[GlobalElementMatrix[k,1+n],GlobalElementMatrix[k,1+m]]+=EleStiffMat[n,m]
    return GlobalStiffMat

    #PreProcessor
def PreProc():
    XDom=2
    YDom=1
    
    X=XDom*np.array([0, 1, 1, 0, 0.5 , 1   ,0.5 ,0   ,0.5],dtype=np.float64)
    Y=YDom*np.array([0, 0, 1, 1, 0   , 0.5 ,1   ,0.5 ,0.5],dtype=np.float64)
    
    
    r1=1
    r2=1.6
    r3=3
    X=np.array([r1*np.cos(0.75*np.pi), r1*np.cos(0.25*np.pi), r3*np.cos(0.25*np.pi), r3*np.cos(0.75*np.pi), r1*np.cos(0.5*np.pi) , r2*np.cos(0.25*np.pi)   ,r3*np.cos(0.5*np.pi) ,r2*np.cos(0.75*np.pi)   ,r2*np.cos(0.5*np.pi)],dtype=np.float64)
    Y=np.array([r1*np.sin(0.75*np.pi), r1*np.sin(0.25*np.pi), r3*np.sin(0.25*np.pi), r3*np.sin(0.75*np.pi), r1*np.sin(0.5*np.pi) , r2*np.sin(0.25*np.pi)   ,r3*np.sin(0.5*np.pi) ,r2*np.sin(0.75*np.pi)   ,r2*np.sin(0.5*np.pi)],dtype=np.float64)


    # =============================================================================
    # r1=1
    # r2=2
    # r3=3
    # X=np.array([r1*np.cos(0.75*np.pi), -r3*np.cos(0.75*np.pi), -r1*np.cos(0.75*np.pi), r3*np.cos(0.75*np.pi), r1*np.cos(0.5*np.pi) , r2*np.cos(0.25*np.pi)   ,r3*np.cos(0.5*np.pi) ,r2*np.cos(0.75*np.pi)   ,r2*np.cos(0.5*np.pi)],dtype=np.float64)
    # Y=np.array([r1*np.sin(0.75*np.pi), r3+r1-r3*np.sin(0.25*np.pi), r3+r1-r1*np.sin(0.25*np.pi), r3*np.sin(0.75*np.pi), r1*np.sin(0.5*np.pi) , r3+r1-r2*np.sin(0.25*np.pi)   ,r3*np.sin(0.5*np.pi) ,r2*np.sin(0.75*np.pi)   ,r2*np.sin(0.5*np.pi)],dtype=np.float64)
    # =============================================================================


    XY=np.array([X,Y])
    
    
    ElemBeginNum=0
    NodeBeginNum=0


    N_row=50
    N_col=50
    
    f=0
    phi_0=0
    


    if CheckDetJ(XY):
        LocalMatrix1,NodeList=NodeArrange(N_row,N_col,XY,NodeBeginNum,ElemBeginNum)
        GlobalElementMatrix=np.array(LocalMatrix1)
    
        #plot Node Locations
        plt.figure(1)
        plt.plot(NodeList[1,:],NodeList[2,:],'*')
        
        GlobalStiffMat=assembleGlobalStiffMat(GlobalElementMatrix,NodeList)
        BCnodes=np.array([*np.arange(0,2*N_row+1), 
                  *np.arange(1+2*N_row,1+2*N_row+(N_col)*(2+4*N_row),(2+4*N_row)),
                  *np.arange(2+4*N_row,2+4*N_row+(N_col)*(2+4*N_row),(2+4*N_row)),
                  *np.arange(2*(N_row-1)+3+2*N_row,2*(N_row-1)+(N_col)*(2+4*N_row)+3+2*N_row,(2+4*N_row)),
                  *np.arange(2*(N_row-1)+4+4*N_row,2*(N_row-1)+(N_col)*(2+4*N_row)+4+4*N_row,(2+4*N_row)),
                  *np.arange((N_col)*(2+4*N_row)-1,2*N_row+(N_col)*(2+4*N_row))
                  ])
        return GlobalStiffMat,GlobalElementMatrix,NodeList,BCnodes
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
