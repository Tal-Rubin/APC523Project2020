# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:39:02 2020

@author: trubin
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Quadratic_Poly_Functions as QBF
import os
import numpy as np
import matplotlib.animation as animation

def unpickle(file):
    import pickle

    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res   

def initPlotDGelem(rho,GlobalElementMatrix,NodeList,DGorder,res=2):   
    xp=np.linspace(-.95,.95,res)
    yp=np.linspace(-.95,.95,res)
    
    Xnode=np.array([])
    Ynode=np.array([])
    rhonode=np.array([])
    for k in range(GlobalElementMatrix.shape[0]):
        for i in range(len(xp)):
            for j in range(len(yp)):
                
                rhonode=np.append(rhonode,(rho[0,DGorder*k:DGorder*(k+1)]@QBF.Ma[0:DGorder])(xp[i],yp[j]))
                Xnode=np.append(Xnode,(NodeList[1,GlobalElementMatrix[k,1:10]]@QBF.Na)(xp[i],yp[j]))
                Ynode=np.append(Ynode,(NodeList[2,GlobalElementMatrix[k,1:10]]@QBF.Na)(xp[i],yp[j]))

    for t in range(1,rho.shape[0]):
        rhonode2=np.array([])

        for k in range(GlobalElementMatrix.shape[0]):
            for i in range(len(xp)):
                for j in range(len(yp)):
                    rhonode2=np.append(rhonode2,(rho[t,DGorder*k:DGorder*(k+1)]@QBF.Ma[0:DGorder])(xp[i],yp[j]))
        rhonode=np.vstack((rhonode,rhonode2))
    return Xnode,Ynode,rhonode

def update_plot(frame_number, z, plot,x,y,fig,ax):
    plot[0].remove()
    plot[0] = ax.plot_trisurf(x,y,z[frame_number,:],cmap='viridis')


    


def PostProc(header='Den',DGorder=1):
    files=[f for f in os.listdir('./Data')] 
    times=np.array([])
    rho=np.array([])
    
    GlobalElementMatrix,NodeList=unpickle('./Data/Grid')


    for file in files:
        if file.startswith(header):
            trho,ttimes=unpickle('./Data/'+file)
            rho=np.concatenate((rho,trho))
            times=np.append(times,ttimes)
    rho=rho.reshape(len(times),-1)
    
    x,y,z=initPlotDGelem(rho,GlobalElementMatrix,NodeList,DGorder)
    print('Minimum density = '+str(z.min()))
    
    fps = 10 # frame per sec
    frn = len(times) 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot = [ax.plot_trisurf(x, y, z[0,:],cmap='viridis')]
    ax.set_zlim(0,1)
    ani=animation.FuncAnimation(fig, update_plot, frn, fargs=(z, plot,x,y,fig,ax), interval=1000/fps)
    
    fn = 'Density animation'
    #ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
    ani.save(fn+'.gif',writer='imagemagick',fps=fps)
    
    
    #return rho, times
