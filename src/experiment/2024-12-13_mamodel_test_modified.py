# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:46:33 2023

@author: Nibaldo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#%%

def material_Birzle(material_state,material_parameters):
    
    C=material_state
    #F=material_state #3x3
    c,c1,c3,d1,d3,beta=material_parameters
    #C=F.T@F
    I1=np.trace(C)
    I = np.eye(3)
    I3=np.linalg.det(C)
    C_inv = np.linalg.inv(C)
        
    S=  c*I - c*(I3**(-beta)) * C_inv.T + \
        c1 * d1 * (I1 * I3**(-1/3) - 3)**(d1 - 1) * (I * I3**(-1/3) - (1/3) * I1 * I3**(-4/3) * C_inv.T) +  \
        c3 * d3 * (I3**(1/3) - 1)**(d3 - 1) * (1/3) * I3**(1/3) * C_inv.T
    
    return S



def material_MaExponential(material_state,material_parameters):
    C=material_state
    #F=material_state #3x3
    a,b,k=material_parameters
    #C=F.T@F
    I1=np.trace(C)
    I2=0.5*( (I1*I1-np.trace(C@C)))
    psi=k*np.exp(  a*(0.5*I1-3/2)*(0.5*I1-3/2)+b*(0.25*I2-0.5*I1+0.75)   )
    
    S=psi*(  (I1*a-3*a-b )*np.eye(3)+0.5*b*(I1*np.eye(3)-C))
  #  E=0.5*(C-np.eye(3))
    return psi,S


'''
def stress_bir2019(u, p,c=0.3567,beta=1.075):
    I = Identity(3)
    F = (I + grad(u))
    C=variable(F.T*F)
    I1C=variable(tr(C))
    I2C=0.5*((tr(C)**2)-tr(C*C))
    I3C=variable(det(C))
    
    J=variable(sqrt(I3C))
    I1raya=I1C*pow(I3C,-1/3)
    psi=c*(I1C-3)  +  (c/beta)*(pow(I3C,-beta)-1)+(278.2/1000)*pow((pow(I3C,-1/3)*I1C-3),3)+(5.766/1000)*pow((pow(I3C,1/3)-1),6)
    S=2*diff(psi,C)
    PK=F*S-p*J*inv(F).T 
    return PK          

'''

#%%Volumetric test

# Default constants
beta = 1.075
c = 0.3567
c1 = 0.2782
c3 = 5.766e-3 
d1 = 3
d3 = 6

test_factors = [0.5, 0.75, 0.90, 1.0, 1.10, 1.25, 1.5]
n_lines = len(test_factors)

mapname = 'coolwarm'
nlines=len(test_factors)
cmap = mpl.colormaps[mapname]
colors = cmap(np.linspace(0,1,n_lines))

fig,ax  = plt.subplots(figsize=(6,4),dpi=200)

# Range of lambdas for the current test
lams=np.arange(1,2.0,0.05)

for j,tf in enumerate(test_factors):
    
    # Apply test factor to the material parameter under analysis
    c_test = c  * tf
    c1_test = c1
    c3_test = c3
    beta_test = beta 
    material_parameters = c_test,c1_test,c3_test,d1,d3,beta_test
    
    for af in np.linspace(0.2,3.0,3):
        
        # Empty data holders
        Sxxs=[]
        Js=[]
        
        for i in np.arange(len(lams)):
            
            # Build 
            lam=lams[i]
            F=np.array([[ lam, 0., 0],
                      [   0., lam, 0],
                      [   0,    0,  lam]]) 
    
            J=np.linalg.det(F)
            C=F.T*F
            material_state=C
            Se=material_Birzle(material_state,material_parameters)
            Sxx=Se[0,0]#+Sgamma[0,0]
            Sxxs.append(Sxx)
            Js.append(J)
        
        ls = "--" if j == 3 else "-"
    ax.plot(Js,Sxxs,color=colors[j], ls=ls,label="Factor=%.2f"%tf)
    
x0, x1 = ax.get_xlim()
ax.set_xlim((x0,x0+(x1-x0)*1.10))        
ax.set_ylabel("Stress")
ax.set_xlabel("Jacobian")
#ax.legend()
        
      