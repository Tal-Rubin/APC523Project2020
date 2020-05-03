# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:55:55 2020

@author: trubin
"""
import numpy as np
#polynomial operations:
class poly:
    def __init__(self, coefficients):
        """ input: coefficients are in a 2d matrix form
        """
        if isinstance(coefficients, list):
            if all(isinstance(i, list) for i in coefficients):
                self.coefficients = np.array(coefficients,dtype=np.float64) 
            else:
                self.coefficients = np.array([coefficients],dtype=np.float64)
        elif isinstance(coefficients, int) or isinstance(coefficients, float) or isinstance(coefficients, np.float) or isinstance(coefficients, np.float64):
            self.coefficients = np.array([[coefficients]],dtype=np.float64)
        elif isinstance(coefficients,np.ndarray):
            self.coefficients = coefficients
        for i in range(self.coefficients.shape[0]):
            for j in range(self.coefficients.shape[1]):
                if abs(self.coefficients[i,j])<5e-16:
                    self.coefficients[i,j]=0
    def __repr__(self):
        """
        method to return the canonical string representation 
        of a polynomial.
   
        """
        res=''
        for i,coeff in np.ndenumerate(self.coefficients):
            if coeff!=0:
                if len(res)==0:
                    if coeff!=1:
                        res+= str(coeff)
                        if i[1]!=0 and i[1]!=1:
                            res+='*x**'+str(i[1])
                        if i[1]==1:
                            res+='*x' 
                        if i[0]!=0 and i[0]!=1:
                            res+='*y**'+str(i[0])
                        if i[0]==1:
                            res+='*y'
                    else:
                        if i[1]!=0 and i[1]!=1 and i[0]!=0:
                            res+='x**'+str(i[1])+'*'
                        if i[1]==1 and i[0]!=0:
                            res+='x*' 
                        if i[1]!=0 and i[1]!=1 and i[0]==0:
                            res+='x**'+str(i[1])
                        if i[1]==1 and i[0]==0:
                            res+='x' 
                        if i[0]!=0 and i[0]!=1:
                            res+='y**'+str(i[0])
                        if i[0]==1:
                            res+='y'
                        if i[1]==0 and i[0]==0:
                            res+= str(coeff)
                else:
                    if coeff>0 and coeff!=1:
                        res+= '+'+str(coeff)+'*'
                    elif coeff==1:
                        res+= '+'
                    else:
                        res+= str(coeff)+'*'
                    if i[1]!=0 and i[1]!=1 and i[0]!=0:
                        res+='x**'+str(i[1])+'*'
                    if i[1]==1 and i[0]!=0:
                        res+='x*' 
                    if i[1]!=0 and i[1]!=1 and i[0]==0:
                        res+='x**'+str(i[1])
                    if i[1]==1 and i[0]==0:
                        res+='x' 
                    if i[0]!=0 and i[0]!=1:
                        res+='y**'+str(i[0])
                    if i[0]==1:
                        res+='y'
        if len(res)==0:
            res='0'
        return res
    
    def tex(self):
        """
        method to return the canonical string representation 
        of a polynomial.
   
        """
        res=''
        for i,coeff in np.ndenumerate(self.coefficients):
            if coeff!=0:
                if len(res)==0:
                    if coeff!=1:
                        res+= str(coeff)
                        if i[1]!=0 and i[1]!=1:
                            res+='x^{'+str(i[1])+'}'
                        if i[1]==1:
                            res+='x' 
                        if i[0]!=0 and i[0]!=1:
                            res+='y^{'+str(i[0])+'}'
                        if i[0]==1:
                            res+='y'
                    else:
                        if i[1]!=0 and i[1]!=1 and i[0]!=0:
                            res+='x^{'+str(i[1])+'}'
                        if i[1]==1 and i[0]!=0:
                            res+='x' 
                        if i[1]!=0 and i[1]!=1 and i[0]==0:
                            res+='x^{'+str(i[1])+'}'
                        if i[1]==1 and i[0]==0:
                            res+='x' 
                        if i[0]!=0 and i[0]!=1:
                            res+='y^{'+str(i[0])+'}'
                        if i[0]==1:
                            res+='y'
                        if i[1]==0 and i[0]==0:
                            res+= str(coeff)
                else:
                    if coeff>0 and coeff!=1:
                        res+= '+'+str(coeff)
                    elif coeff==1:
                        res+= '+'
                    else:
                        res+= str(coeff)
                    if i[1]!=0 and i[1]!=1 and i[0]!=0:
                        res+='x^{'+str(i[1])+'}'
                    if i[1]==1 and i[0]!=0:
                        res+='x' 
                    if i[1]!=0 and i[1]!=1 and i[0]==0:
                        res+='x^{'+str(i[1])+'}'
                    if i[1]==1 and i[0]==0:
                        res+='x' 
                    if i[0]!=0 and i[0]!=1:
                        res+='y^{'+str(i[0])+'}'
                    if i[0]==1:
                        res+='y'

# =============================================================================
#         for i,coeff in np.ndenumerate(self.coefficients):
#             if coeff !=0:
#                 if len(res)>0 and coeff>0 and coeff!=1:
#                     res+= '+'+str(coeff)
#                 elif len(res)>0 and coeff==1:
#                     res+= '+'
#                 elif len(res)==0 and coeff==1:
#                     res+= ''
#                 else:
#                     res+= str(coeff)
#                 if i[1]!=0 and i[1]!=1:
#                     res+='x^{'+str(i[1])+'}'
#                 if i[1]==1:
#                     res+='x' 
#                 if i[0]!=0 and i[0]!=1:
#                     res+='y^{'+str(i[0])+'}'
#                 if i[0]==1:
#                     res+='y'
#                 if len(res)==0 and i[0]==0 and i[1]==0:
#                     res+= str(coeff)
# =============================================================================

        return res
    def __call__(self, x=None,y=None):    
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            if x.any()!=None and y.any()!=None:
                res = 0
                for i,coeff in np.ndenumerate(self.coefficients):
                    res += coeff*x**i[1]*y**i[0]
            if x.any()!=None and y.any()==None:
                res=np.zeros((self.degreey(),1))
                for i,coeff in np.ndenumerate(self.coefficients):
                    res[i[0],0] += coeff*x**i[1]
                res=poly(res)
            if x.any()==None and y.any()!=None:
                res=np.zeros((1,self.degreex()))
                for i,coeff in np.ndenumerate(self.coefficients):
                    res[0,i[1]] += coeff*y**i[0]
                res=poly(res)
        else:
            if x!=None and y!=None:
                res = 0
                for i,coeff in np.ndenumerate(self.coefficients):
                    res += coeff*x**i[1]*y**i[0]
            if x!=None and y==None:
                res=np.zeros((self.degreey(),1))
                for i,coeff in np.ndenumerate(self.coefficients):
                    res[i[0],0] += coeff*x**i[1]
                res=poly(res)
            if x==None and y!=None:
                res=np.zeros((1,self.degreex()))
                for i,coeff in np.ndenumerate(self.coefficients):
                    res[0,i[1]] += coeff*y**i[0]
                res=poly(res)
        return res

    def degreex(self):
        return self.coefficients.shape[1]
    
    def degreey(self):
        return self.coefficients.shape[0]
    
    def degree(self):
        return max(self.coefficients.shape)
    
    def __add__(self,other):  
        if isinstance(other, int) or isinstance(other, float):
            other=poly(other)
        res=np.zeros((max(self.coefficients.shape[0],other.degreey()),max(self.coefficients.shape[1],other.degreex())),dtype=np.float64)
        for i in range(self.coefficients.shape[0]):
            for j in range(self.coefficients.shape[1]):
                res[i,j]+=self.coefficients[i,j]
        for i in range(other.degreey()):
            for j in range(other.degreex()):
                res[i,j]+=other.coefficients[i,j]            
        return poly(res)
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self,other):
        return self.__add__(-1*other)

    def __str__(self):
        return self.tex()
    
    def __mul__(self,other):
        if isinstance(other, int) or isinstance(other, float):
            other=poly(other)
        res=np.zeros((self.degreey()+other.degreey()-1,self.degreex()+other.degreex()-1),dtype=np.float64)
        for i in range(self.degreey()):
            for j in range(self.degreex()):
                res[i:i+other.degreey(),j:j+other.degreex()]+=self.coefficients[i,j]*other.coefficients
        return poly(res)
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def dx(self):
        res=np.zeros((self.coefficients.shape[0],self.coefficients.shape[1]-1),dtype=np.float64)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i,j]=self.coefficients[i,j+1]*(j+1)
        return poly(res)
    
    def dy(self):
        res=np.zeros((self.coefficients.shape[0]-1,self.coefficients.shape[1]),dtype=np.float64)
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i,j]=self.coefficients[i+1,j]*(i+1)
        return poly(res)    
    
    def intx(self,ax=None,bx=None):
        res=np.zeros((self.coefficients.shape[0],self.coefficients.shape[1]+1),dtype=np.float64)
        for i in range(res.shape[0]):
            for j in range(self.coefficients.shape[1]):
                res[i,j+1]=self.coefficients[i,j]/(j+1)
        res=poly(res)
        if ax==None:
            return res
        elif bx==None:
            return res(ax)
        else:
            return res(bx)-res(ax)

    
    def inty(self,ay=None,by=None):
        res=np.zeros((self.coefficients.shape[0]+1,self.coefficients.shape[1]),dtype=np.float64)
        for i in range(self.coefficients.shape[0]):
            for j in range(res.shape[1]):
                res[i+1,j]=self.coefficients[i,j]/(i+1)
        res=poly(res)
        if ay==None:
            return res
        elif by==None:
            return res(None,ay)
        else:
            return res(None,by)-res(None,ay)    
    
    def scalar(self):
        if self.coefficients.shape==(1,1):
            return self.coefficients[0,0]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    