# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:04:11 2025

@author: angus
"""

'''

Código para estudiar la ley de permeabilidad-porosidad vigente

'''

import matplotlib.pyplot as plt
import numpy as np


def permeability_function(knormal, kmin, phi, phi0, exp=2./3., phi_thr=0.2):
    kappa =  knormal*((phi/phi0)**(exp)) + kmin
    kappa[phi<phi_thr] = kmin
    return kappa

logplot = False
phi_space = np.linspace(0,1.,100)
phi0 = 0.5
knormal=1e3
kmin=10
exp = 2./3.
phi_thr=0.1

phi0s = [0.10,0.25,0.50,0.75]

fig,ax = plt.subplots(dpi=150,figsize=(5,5))

for phi0 in phi0s:
    
    k_space = permeability_function(knormal,kmin,phi_space,phi0,exp=exp,phi_thr=0.0)
    
    if logplot:
        ydata = np.log10(k_space)
    else:
        ydata = k_space
        
    ax.plot(phi_space,ydata, label='$\phi_0=$%.2f'%phi0, alpha=phi0, color="k")

ax.legend()
ax.set_title('Permeability and porosity relationship')
if logplot:
    ax.set_ylabel('Log$_10$(Permeability) $\kappa$')
else:
    ax.set_ylabel('Permeability $\kappa$')
ax.set_xlabel('Porosity $\phi$')