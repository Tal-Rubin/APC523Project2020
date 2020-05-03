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
    
    Jgrid=np.zeros((len(xp),len(xp)))
    for i in range(len(Xp)):
        for j in range(len(Yp)):
            Jgrid[i,j]=detJ(Xp[i,j], Yp[i,j],XY)
    
    
    
    
    
    plt.figure(4)
    ax = plt.axes(projection='3d')
    ax.plot_surface(Xp,Yp,Jgrid)
    ax.plot_surface(Xp,Yp,np.zeros((len(xp),len(xp))))

def NodeArrange(N_row,N_col,XY,NodeBeginNum,ElemBeginNum):
    #splitting macro-element into elements
    LM=np.zeros((N_row*N_col,37))
    for i in range(N_col):
        for j in range(N_row):
            LM[j+i*N_row,0]   =   -1  +  i*2/N_col              #Node 1, xi
            LM[j+i*N_row,1]   =   1  -  (j+1)*2/N_row           #Node 1, eta
            LM[j+i*N_row,2]   =   2*j+i*(2+4*N_row)+2+NodeBeginNum           #Node 1, #    j+i*N_row+i+1
            LM[j+i*N_row,3]   =   -1  +  (i+1)*2/N_col          #Node 2, xi
            LM[j+i*N_row,4]   =   LM[j+i*N_row,1]               #Node 2, eta  
            LM[j+i*N_row,5]   =   2*j+(i+1)*(2+4*N_row)+2   +NodeBeginNum         #Node 2, #
            LM[j+i*N_row,6]   =   LM[j+i*N_row,3]               #Node 3, xi
            LM[j+i*N_row,7]   =   1  -  j*2/N_row               #Node 3, eta
            LM[j+i*N_row,8]   =   2*j+(i+1)*(2+4*N_row)  +NodeBeginNum       #Node 3, #
            LM[j+i*N_row,9]   =   LM[j+i*N_row,0]               #Node 4, xi
            LM[j+i*N_row,10]  =   LM[j+i*N_row,7]               #Node 4, eta
            LM[j+i*N_row,11]  =   2*j+i*(2+4*N_row)      +NodeBeginNum       #Node 4, #  j+i*N_row+i
            LM[j+i*N_row,12]  =   -1  +  (i+0.5)*2/N_col        #Node 5, xi
            LM[j+i*N_row,13]  =   LM[j+i*N_row,1]               #Node 5, eta
            LM[j+i*N_row,14]  =   LM[j+i*N_row,11]+3+2*N_row  +NodeBeginNum  #Node 5, #   
            LM[j+i*N_row,15]  =   LM[j+i*N_row,3]               #Node 6, xi
            LM[j+i*N_row,16]  =   1  -  (j+0.5)*2/N_row         #Node 6, eta
            LM[j+i*N_row,17]  =   LM[j+i*N_row,8]+1 +NodeBeginNum            #Node 6, #
            LM[j+i*N_row,18]  =   LM[j+i*N_row,12]              #Node 7, xi
            LM[j+i*N_row,19]  =   LM[j+i*N_row,7]               #Node 7, eta
            LM[j+i*N_row,20]  =   LM[j+i*N_row,11]+1+2*N_row+NodeBeginNum    #Node 7, #
            LM[j+i*N_row,21]  =   LM[j+i*N_row,0]               #Node 8, xi
            LM[j+i*N_row,22]  =   LM[j+i*N_row,16]              #Node 8, eta
            LM[j+i*N_row,23]  =   LM[j+i*N_row,11]+1 +NodeBeginNum           #Node 8, #
            LM[j+i*N_row,24]  =   LM[j+i*N_row,12]              #Node 9, xi
            LM[j+i*N_row,25]  =   LM[j+i*N_row,16]              #Node 9, eta
            LM[j+i*N_row,26]  =   LM[j+i*N_row,11]+2+2*N_row  +NodeBeginNum  #Node 9, #
            
            #fluxes
            LM[j+i*N_row,27]  =0
            LM[j+i*N_row,28]  =0  
            LM[j+i*N_row,29]  =0     
            LM[j+i*N_row,30]  =0 
            LM[j+i*N_row,31]  =0 
            LM[j+i*N_row,32]  =0     
            LM[j+i*N_row,33]  =0       
            LM[j+i*N_row,34]  =0  
            
            
            LM[j+i*N_row,35]  =1          #source term
    
            LM[j+i*N_row,36]  =j+i*N_row+ElemBeginNum               #element #
        
        
            #Moving the node to the global coordinate system
            LM[j+i*N_row,0],LM[j+i*N_row,1]    =    XY@QBF.N(LM[j+i*N_row,0],LM[j+i*N_row,1])  #-1  +  i*2/N_col              #Node 1, xi
            LM[j+i*N_row,3],LM[j+i*N_row,4]    =    XY@QBF.N(LM[j+i*N_row,3],LM[j+i*N_row,4])          #Node 2, xi
            LM[j+i*N_row,6],LM[j+i*N_row,7]    =    XY@QBF.N(LM[j+i*N_row,6],LM[j+i*N_row,7])
            LM[j+i*N_row,9],LM[j+i*N_row,10]   =    XY@QBF.N(LM[j+i*N_row,9],LM[j+i*N_row,10])
            LM[j+i*N_row,12],LM[j+i*N_row,13]  =    XY@QBF.N(LM[j+i*N_row,12],LM[j+i*N_row,13])
            LM[j+i*N_row,15],LM[j+i*N_row,16]  =    XY@QBF.N(LM[j+i*N_row,15],LM[j+i*N_row,16])
            LM[j+i*N_row,18],LM[j+i*N_row,19]  =    XY@QBF.N(LM[j+i*N_row,18],LM[j+i*N_row,19])
            LM[j+i*N_row,21],LM[j+i*N_row,22]  =    XY@QBF.N(LM[j+i*N_row,21],LM[j+i*N_row,22])
            LM[j+i*N_row,24],LM[j+i*N_row,25]  =    XY@QBF.N(LM[j+i*N_row,24],LM[j+i*N_row,25])
    return LM

def Nodes(GM):
    x=np.zeros(int(max(*GM[:,2],*GM[:,5],*GM[:,8],*GM[:,11],*GM[:,14],*GM[:,17],*GM[:,20],*GM[:,23],*GM[:,26]))+1)
    y=x.copy()
    for i in range(GM.shape[0]):
        for j in range(9):
            x[int(GM[i,2+j*3])]=GM[i,j*3]
            y[int(GM[i,2+j*3])]=GM[i,1+j*3]
    return x,y    

#PreProcessor
XDom=2
YDom=1

X=XDom*np.array([0, 1, 1, 0, 0.5 , 1   ,0.5 ,0   ,0.5],dtype=np.float64)
Y=YDom*np.array([0, 0, 1, 1, 0   , 0.5 ,1   ,0.5 ,0.5],dtype=np.float64)


r1=1
r2=1.6
r3=3
X=np.array([r1*np.cos(0.75*np.pi), r1*np.cos(0.25*np.pi), r3*np.cos(0.25*np.pi), r3*np.cos(0.75*np.pi), r1*np.cos(0.5*np.pi) , r2*np.cos(0.25*np.pi)   ,r3*np.cos(0.5*np.pi) ,r2*np.cos(0.75*np.pi)   ,r2*np.cos(0.5*np.pi)],dtype=np.float64)
Y=np.array([r1*np.sin(0.75*np.pi), r1*np.sin(0.25*np.pi), r3*np.sin(0.25*np.pi), r3*np.sin(0.75*np.pi), r1*np.sin(0.5*np.pi) , r2*np.sin(0.25*np.pi)   ,r3*np.sin(0.5*np.pi) ,r2*np.sin(0.75*np.pi)   ,r2*np.sin(0.5*np.pi)],dtype=np.float64)




r1=1
r2=2
r3=3
X=np.array([r1*np.cos(0.75*np.pi), -r3*np.cos(0.75*np.pi), -r1*np.cos(0.75*np.pi), r3*np.cos(0.75*np.pi), r1*np.cos(0.5*np.pi) , r2*np.cos(0.25*np.pi)   ,r3*np.cos(0.5*np.pi) ,r2*np.cos(0.75*np.pi)   ,r2*np.cos(0.5*np.pi)],dtype=np.float64)
Y=np.array([r1*np.sin(0.75*np.pi), r3+r1-r3*np.sin(0.25*np.pi), r3+r1-r1*np.sin(0.25*np.pi), r3*np.sin(0.75*np.pi), r1*np.sin(0.5*np.pi) , r3+r1-r2*np.sin(0.25*np.pi)   ,r3*np.sin(0.5*np.pi) ,r2*np.sin(0.75*np.pi)   ,r2*np.sin(0.5*np.pi)],dtype=np.float64)






XY=np.array([X,Y])


ElemBeginNum=0
NodeBeginNum=0


N_row=30
N_col=30

f=0
phi_0=0
    

def detJ(xi,eta,XY):
    return np.linalg.det(XY@QBF.B(xi, eta))

def invJ(XY):
    return np.array([[(XY@QBF.Ba)[1,1],-(XY@QBF.Ba)[0,1]],[-(XY@QBF.Ba)[1,0],(XY@QBF.Ba)[0,0]]])

if detJ(-0.5,0.5,XY)<0 or detJ(0.5,-0.5,XY)<0 or detJ(-0.5,-0.5,XY)<0 or detJ(0.5,0.5,XY)<0:
    print("Bad Macro Element Coordinates")

LM1=NodeArrange(N_row,N_col,XY,NodeBeginNum,ElemBeginNum)

   
k_mast=np.zeros((9,9,9,9))
for i in range(9):
    for j in range(9):
        for n in range(9):
            for m in range(9):
                temp=(QBF.Na[i].dx()*QBF.Na[j].dx()+QBF.Na[i].dy()*QBF.Na[j].dy())*(QBF.Na[n].dx()*QBF.Na[m].dy())
                temp=temp.intx(-1,1)
                temp=temp.inty(-1,1)
                k_mast[n,m,i,j]=temp.scalar()






GM=np.array(LM1)
xNodes,yNodes=Nodes(GM)
plt.figure(1)
plt.plot(xNodes,yNodes,'*')
#plt.axes().set_aspect('equal', 'datalim')

GlobalStiffMat=np.zeros((len(xNodes),len(yNodes)))

#stiffness matrix of each element
Xele=np.zeros(9)
Yele=np.zeros(9)
# =============================================================================
# detelem00=np.zeros(GM.shape[0])
# detelempp=np.zeros(GM.shape[0])
# detelemmm=np.zeros(GM.shape[0])
# detelempm=np.zeros(GM.shape[0])
# detelemmp=np.zeros(GM.shape[0])
# =============================================================================

for k in range(GM.shape[0]):
    for j in range(9):
        Xele[j]=GM[k,3*j] 
        Yele[j]=GM[k,3*j+1]
    XYele=np.array([Xele,Yele])
# =============================================================================
#     detelem00[k]=detJ(0,0,XYele)
#     detelempp[k]=detJ(1,1,XYele)
#     detelemmm[k]=detJ(-1,-1,XYele)
#     detelempm[k]=detJ(1,-1,XYele)
#     detelemmp[k]=detJ(-1,1,XYele)
# =============================================================================
    NodeXYantySymOuter=np.outer(Xele,Yele)-np.outer(Yele,Xele)

    EleStiffMat=np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            for n in range(9):
                for m in range(9):
                    EleStiffMat[i,j]+=k_mast[n,m,i,j]*NodeXYantySymOuter[n,m]
    for n in range(9):
        for m in range(9):
            GlobalStiffMat[int(GM[k,3*n+2]),int(GM[k,3*m+2])]+=EleStiffMat[n,m]
                
                
# =============================================================================
# k1=np.array([[28/45, -1/30,  -1/45,  -1/30,  -1/5,   1/9,    1/9,    -1/5,   -16/45],
#             [-1/30, 28/45,  -1/30,  -1/45,  -1/5,   -1/5,   1/9,    1/9,    -16/45],
#             [-1/45, -1/30,  28/45,  -1/30,  1/9,    -1/5,   -1/5,   1/9,    -16/45],
#             [-1/30, -1/45,  -1/30,  28/45,  1/9,    1/9,    -1/5,   -1/5,   -16/45],
#             [-1/5,  -1/5,   1/9,    1/9,    88/45,  -16/45, 0,      -16/45, -16/15],
#             [1/9,   -1/5,   -1/5,   1/9,    -16/45, 88/45,  -16/45, 0,      -16/15],
#             [1/9,   1/9,    -1/5,   -1/5,   0,      -16/45, 88/45,  -16/45, -16/15],
#             [-1/5,  1/9,    1/9,    -1/5,   -16/45, 0,      -16/45, 88/45,  -16/15],
#             [-16/45,-16/45, -16/45, -16/45, -16/15, -16/15, -16/15, -16/15, 256/45]])
# =============================================================================
 

#plt.figure(123)
#plt.imshow(GlobalStiffMat)
 
b=np.ones(xNodes.shape)*1e-10

BCnodes=np.array([*np.arange(0,2*N_row+1), 
                  *np.arange(1+2*N_row,1+2*N_row+(N_col)*(2+4*N_row),(2+4*N_row)),
                  *np.arange(2+4*N_row,2+4*N_row+(N_col)*(2+4*N_row),(2+4*N_row)),
                  *np.arange(2*(N_row-1)+3+2*N_row,2*(N_row-1)+(N_col)*(2+4*N_row)+3+2*N_row,(2+4*N_row)),
                  *np.arange(2*(N_row-1)+4+4*N_row,2*(N_row-1)+(N_col)*(2+4*N_row)+4+4*N_row,(2+4*N_row)),
                  *np.arange((N_col)*(2+4*N_row)-1,2*N_row+(N_col)*(2+4*N_row))
                  ])



plt.figure(3)
for i in BCnodes:
    GlobalStiffMat[i,:]=np.zeros((1,GlobalStiffMat.shape[1]))
    GlobalStiffMat[i,i]=1
    b[i]=0
    plt.plot(xNodes[i],yNodes[i],'*')
Sol=np.linalg.solve(GlobalStiffMat,b)


plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_trisurf(xNodes,yNodes,Sol,
                cmap='viridis')


# =============================================================================
# GlobalStiffMatInv=np.linalg.inv(GlobalStiffMat)
# plt.figure(124)
# plt.imshow(GlobalStiffMatInv)
# 
# =============================================================================

#electric field 
#Ex= QBF.N1.dx()/((QBF.Ba@XY)[0,0])+QBF.N1.dy()/((QBF.Ba@XY)[0,1])
#Ey= QBF.N1.dx()/((QBF.Ba@XY)[1,0])+QBF.N1.dy()/((QBF.Ba@XY)[1,1])




#plt.figure(7)
#plt.plot(detelem00,color='r')
#plt.plot(detelempp,color='g')
#plt.plot(detelemmm,color='b')
#plt.plot(detelempm,color='y')
#plt.plot(detelemmp,color='m')
