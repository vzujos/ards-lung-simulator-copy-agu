# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:46:31 2026

@author: angus
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Helvetica'

def c_phi(phi, c_tissue_min, c_tissue_max, k=80, phi_t=0.2):
    return (
        c_tissue_max
        - (c_tissue_max - c_tissue_min)
        * (1 - 1 / (1 + np.exp(-k * (phi_t - phi))))
    )


def evaluate_c_phi(c_tissue_min, c_tissue_max, k=80, phi_t=0.2, n=500):

    phi = np.linspace(0.0, 1.0, n)
    c = c_phi(phi, c_tissue_min, c_tissue_max, k, phi_t)

    return phi, c


# Example parameters
c_tissue_min = 1.0
c_tissue_max = 5.0

phi, c = evaluate_c_phi(c_tissue_min, c_tissue_max)

fs = 14
# Plot
fig,ax =  plt.subplots(dpi=300)
ax.plot(phi, c)
ax.set_xlabel("Porosity $\phi$",size=fs)
ax.axhline(c_tissue_max, alpha=0.2, ls="--",color='k')
ax.axhline(c_tissue_min, alpha=0.2, ls="--",color='k')
ax.text(0.1,1.0+.05,"$c_\min$",size=fs)
ax.text(0.6,5.0-0.15,"$c_\max$"+"$=5 c_\min$",size=fs)
ax.set_ylim(0.0,5.5)
ax.set_ylabel("Tissue stiffness - $c(\phi)$",size=fs)
