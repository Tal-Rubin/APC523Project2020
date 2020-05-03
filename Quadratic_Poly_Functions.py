# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:00:55 2020

@author: trubin
"""

import numpy as np
import polynomial2d as p2d

#quadratic basis funcitons


N9 = p2d.poly([1,0,-1])*p2d.poly([[1],[0],[-1]])
N8 = 0.5*p2d.poly([1,-1])*p2d.poly([[1],[0],[-1]])-0.5*N9
N7 = 0.5*p2d.poly([1,0,-1])*p2d.poly([[1],[1]])-0.5*N9
N6 = 0.5*p2d.poly([1,1])*p2d.poly([[1],[0],[-1]])-0.5*N9
N5 = 0.5*p2d.poly([1,0,-1])*p2d.poly([[1],[-1]])-0.5*N9
N4 = 0.25*p2d.poly([1,-1])*p2d.poly([[1],[1]])-0.5*N7-0.5*N8-0.25*N9
N3 = 0.25*p2d.poly([1,1])*p2d.poly([[1],[1]])-0.5*N6-0.5*N7-0.25*N9
N2 = 0.25*p2d.poly([1,1])*p2d.poly([[1],[-1]])-0.5*N5-0.5*N6-0.25*N9
N1 = 0.25*p2d.poly([1,-1])*p2d.poly([[1],[-1]])-0.5*N5-0.5*N8-0.25*N9

Na=np.array([N1,N2,N3,N4,N5,N6,N7,N8,N9])

Ba=np.array([[N1.dx(),N2.dx(),N3.dx(),N4.dx(),N5.dx(),N6.dx(),N7.dx(),N8.dx(),N9.dx()],
            [N1.dy(),N2.dy(),N3.dy(),N4.dy(),N5.dy(),N6.dy(),N7.dy(),N8.dy(),N9.dy()]]).T

def N(xi,eta):
    return np.array([N1(xi,eta),N2(xi,eta),N3(xi,eta),N4(xi,eta),N5(xi,eta),N6(xi,eta),N7(xi,eta),N8(xi,eta),N9(xi,eta)])



def B(xi,eta):
    return np.array([[N1.dx()(xi,eta),N2.dx()(xi,eta),N3.dx()(xi,eta),N4.dx()(xi,eta),N5.dx()(xi,eta),N6.dx()(xi,eta),N7.dx()(xi,eta),N8.dx()(xi,eta),N9.dx()(xi,eta)],
                     [N1.dy()(xi,eta),N2.dy()(xi,eta),N3.dy()(xi,eta),N4.dy()(xi,eta),N5.dy()(xi,eta),N6.dy()(xi,eta),N7.dy()(xi,eta),N8.dy()(xi,eta),N9.dy()(xi,eta)]]).T

