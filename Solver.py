# -*- coding: utf-8 -*-
"""
Created on Sat May  2 22:14:24 2020

@author: trubin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import PreProc


GlobalStiffMat,GlobalElementMatrix,NodeList,BCnodes=PreProc.PreProc()    


b=np.ones(NodeList.shape[1])*1e-10



#plt.imshow(GlobalStiffMat)

plt.figure(3)
for i in BCnodes:
    GlobalStiffMat[i,:]=np.zeros((1,GlobalStiffMat.shape[1]))
    GlobalStiffMat[i,i]=1
    b[i]=0
    plt.plot(NodeList[1,i],NodeList[2,i],'*')
Sol=np.linalg.solve(GlobalStiffMat,b)


plt.figure(2)
ax = plt.axes(projection='3d')
ax.plot_trisurf(NodeList[1,:],NodeList[2,:],Sol,
                cmap='viridis')