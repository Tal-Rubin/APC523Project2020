#This module contains meshing, and initial conditions


# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:41:28 2020

@author: trubin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#PreProcessor
XDom=3
YDom=2

X=XDom*np.array([0, 1, 1, 0, 0.5 , 1   ,0.5 ,0   ,0.5])
Y=YDom*np.array([0, 0, 1, 1, 0   , 0.5 ,1   ,0.5 ,0.5])

XY=np.array([X,Y]).T


ElemBeginNum=0
NodeBeginNum=0


N_row=2
N_col=3

f=0
phi_0=0



# %%

#basis funcitons


def N9(xi,eta):
    return (1-xi**2)*(1-eta**2)

def N8(xi,eta):
    return 0.5*(1-xi)*(1-eta**2)-0.5*N9(xi,eta)

def N7(xi,eta):
    return 0.5*(1-xi**2)*(1+eta)-0.5*N9(xi,eta)

def N6(xi,eta):
    return 0.5*(1+xi)*(1-eta**2)-0.5*N9(xi,eta)

def N5(xi,eta):
    return 0.5*(1-xi**2)*(1-eta)-0.5*N9(xi,eta)

def N4(xi,eta):
    return 0.25*(1-xi)*(1+eta)-0.5*N7(xi,eta)-0.5*N8(xi,eta)-0.25*N9(xi,eta)

def N3(xi,eta):
    return 0.25*(1+xi)*(1+eta)-0.5*N6(xi,eta)-0.5*N7(xi,eta)-0.25*N9(xi,eta)

def N2(xi,eta):
    return 0.25*(1+xi)*(1-eta)-0.5*N5(xi,eta)-0.5*N6(xi,eta)-0.25*N9(xi,eta)

def N1(xi,eta):
    return 0.25*(1-xi)*(1-eta)-0.5*N5(xi,eta)-0.5*N8(xi,eta)-0.25*N9(xi,eta)



def N9_1(xi,eta):
    return -2*xi*(1-eta**2)

def N8_1(xi,eta):
    return -0.5*(1-eta**2)-0.5*N9_1(xi,eta)

def N7_1(xi,eta):
    return -xi*(1+eta)-0.5*N9_1(xi,eta)

def N6_1(xi,eta):
    return 0.5*(1-eta**2)-0.5*N9_1(xi,eta)

def N5_1(xi,eta):
    return -xi*(1-eta)-0.5*N9_1(xi,eta)

def N4_1(xi,eta):
    return -0.25*(1+eta)-0.5*N7_1(xi,eta)-0.5*N8_1(xi,eta)-0.25*N9_1(xi,eta)

def N3_1(xi,eta):
    return 0.25*(1+eta)-0.5*N6_1(xi,eta)-0.5*N7_1(xi,eta)-0.25*N9_1(xi,eta)

def N2_1(xi,eta):
    return 0.25*(1-eta)-0.5*N5_1(xi,eta)-0.5*N6_1(xi,eta)-0.25*N9_1(xi,eta)

def N1_1(xi,eta):
    return -0.25*(1-eta)-0.5*N5_1(xi,eta)-0.5*N8_1(xi,eta)-0.25*N9_1(xi,eta)


def N9_2(xi,eta):
    return -2*(1-xi**2)*eta

def N8_2(xi,eta):
    return -(1-xi)*eta-0.5*N9_2(xi,eta)

def N7_2(xi,eta):
    return 0.5*(1-xi**2)-0.5*N9_2(xi,eta)

def N6_2(xi,eta):
    return -(1+xi)*eta-0.5*N9_2(xi,eta)

def N5_2(xi,eta):
    return -0.5*(1-xi**2)-0.5*N9_2(xi,eta)

def N4_2(xi,eta):
    return 0.25*(1-xi)-0.5*N7_2(xi,eta)-0.5*N8_2(xi,eta)-0.25*N9_2(xi,eta)

def N3_2(xi,eta):
    return 0.25*(1+xi)-0.5*N6_2(xi,eta)-0.5*N7_2(xi,eta)-0.25*N9_2(xi,eta)

def N2_2(xi,eta):
    return -0.25*(1+xi)-0.5*N5_2(xi,eta)-0.5*N6_2(xi,eta)-0.25*N9_2(xi,eta)

def N1_2(xi,eta):
    return -0.25*(1-xi)-0.5*N5_2(xi,eta)-0.5*N8_2(xi,eta)-0.25*N9_2(xi,eta)

# %%
    
def B(xi,eta):
    return np.array([[N1_1(xi,eta),N2_1(xi,eta),N3_1(xi,eta),N4_1(xi,eta),N5_1(xi,eta),N6_1(xi,eta),N7_1(xi,eta),N8_1(xi,eta),N9_1(xi,eta)],[N1_2(xi,eta),N2_2(xi,eta),N3_2(xi,eta),N4_2(xi,eta),N5_2(xi,eta),N6_2(xi,eta),N7_2(xi,eta),N8_2(xi,eta),N9_2(xi,eta)]])


def J(xi,eta):
    return np.linalg.det(XY@B(xi,eta))

if J(-0.5,0.5)<0 or J(0.5,-0.5)<0 or J(-0.5,-0.5)<0 or J(0.5,0.5)<0:
    print("Bad Macro Element Coordinates")
    
def N(xi,eta):
    return np.array([N1(xi,eta),N2(xi,eta),N3(xi,eta),N4(xi,eta),N5(xi,eta),N6(xi,eta),N7(xi,eta),N8(xi,eta),N9(xi,eta)])

X_xieta=N(0,0).T@X
Y_xieta=N(0,0).T@Y

#splitting macro-element into elements
LM=np.zeros((N_row*N_col,37))
for i in range(N_col):
    for j in range(N_row):
        LM[j+i*N_row,0]   =   -1  +  i*2/N_col              #Node 1, xi
        LM[j+i*N_row,1]   =   1  -  (j+1)*2/N_row           #Node 1, eta
        LM[j+i*N_row,2]   =   2*j+i*(2+4*N_row)+2+NodeBeginNum           #Node 1, #    j+i*N_row+i+1
        LM[j+i*N_row,3]   =   -1  +  (i+1)*2/N_col          #Node 2, xi
        LM[j+i*N_row,4]   =   LM[j+i*N_row,1]               #Node 2, eta  
        LM[j+i*N_row,5]   =   LM[j+i*N_row,8]+2    +NodeBeginNum         #Node 2, #
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
        LM[j+i*N_row,0],LM[j+i*N_row,1]    =    N(LM[j+i*N_row,0],LM[j+i*N_row,1]).T@XY   #-1  +  i*2/N_col              #Node 1, xi
        LM[j+i*N_row,3],LM[j+i*N_row,4]    =    N(LM[j+i*N_row,3],LM[j+i*N_row,4]).T@XY          #Node 2, xi
        LM[j+i*N_row,6],LM[j+i*N_row,7]    =    N(LM[j+i*N_row,6],LM[j+i*N_row,7]).T@XY
        LM[j+i*N_row,9],LM[j+i*N_row,10]   =    N(LM[j+i*N_row,9],LM[j+i*N_row,10]).T@XY
        LM[j+i*N_row,12],LM[j+i*N_row,13]  =    N(LM[j+i*N_row,12],LM[j+i*N_row,13]).T@XY
        LM[j+i*N_row,15],LM[j+i*N_row,16]  =    N(LM[j+i*N_row,15],LM[j+i*N_row,16]).T@XY
        LM[j+i*N_row,18],LM[j+i*N_row,19]  =    N(LM[j+i*N_row,18],LM[j+i*N_row,19]).T@XY
        LM[j+i*N_row,21],LM[j+i*N_row,22]  =    N(LM[j+i*N_row,21],LM[j+i*N_row,22]).T@XY
        LM[j+i*N_row,24],LM[j+i*N_row,25]  =    N(LM[j+i*N_row,24],LM[j+i*N_row,25]).T@XY






CM=np.zeros((N_row*N_col,4))


k=np.array([[28/45, -1/30,  -1/45,  -1/30,  -1/5,   1/9,    1/9,    -1/5,   -16/45],
            [-1/30, 28/45,  -1/30,  -1/45,  -1/5,   -1/5,   1/9,    1/9,    -16/45],
            [-1/45, -1/30,  28/45,  -1/30,  1/9,    -1/5,   -1/5,   1/9,    -16/45],
            [-1/30, -1/45,  -1/30,  28/45,  1/9,    1/9,    -1/5,   -1/5,   -16/45],
            [-1/5,  -1/5,   1/9,    1/9,    88/45,  -16/45, 0,      -16/45, -16/15],
            [1/9,   -1/5,   -1/5,   1/9,    -16/45, 88/45,  -16/45, 0,      -16/15],
            [1/9,   1/9,    -1/5,   -1/5,   0,      -16/45, 88/45,  -16/45, -16/15],
            [-1/5,  1/9,    1/9,    -1/5,   -16/45, 0,      -16/45, 88/45,  -16/15],
            [-16/45,-16/45, -16/45, -16/45, -16/15, -16/15, -16/15, -16/15, 256/45]])
 

 
 
 








def plotting():
    xp=np.linspace(-1,1)
    yp=np.linspace(-1,1)
    
    Xp, Yp = np.meshgrid(xp, yp)
    
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    
    ax.plot_surface(Xp,Yp,N9(Xp, Yp))
    ax.plot_surface(Xp,Yp,N8(Xp, Yp))
    ax.plot_surface(Xp,Yp,N7(Xp, Yp))
    ax.plot_surface(Xp,Yp,N6(Xp, Yp))
    ax.plot_surface(Xp,Yp,N5(Xp, Yp))
    ax.plot_surface(Xp,Yp,N4(Xp, Yp))
    ax.plot_surface(Xp,Yp,N3(Xp, Yp))
    ax.plot_surface(Xp,Yp,N2(Xp, Yp))
    ax.plot_surface(Xp,Yp,N1(Xp, Yp))
    
    
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
            Jgrid[i,j]=J(Xp[i,j], Yp[i,j])
    
    
    
    
    
    fig = plt.figure(4)
    ax = plt.axes(projection='3d')
    ax.plot_surface(Xp,Yp,Jgrid)
