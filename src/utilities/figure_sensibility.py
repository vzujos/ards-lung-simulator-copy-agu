# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:01:28 2024

@author: angus
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']


def retrieve_fields(path_to_data, model=""):
    
        pressures = np.load(path_to_data+"%spresionestodas.npy"%model).flatten()*10.1972 # cmH2O
        times = np.load(path_to_data+"%seffectivetimes.npy"%model).flatten() # s
        fluxes = -np.load(path_to_data+"%sfluxes.npy"%model).flatten() # L/s
        fluxes[0] = fluxes[1]
        volumes = np.load(path_to_data+"%svolumenes.npy"%model).flatten() # 
        volumes -= volumes[0]
        
        return times, pressures, fluxes, volumes

root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/2026-03-SENSITIVITY-PIG2/"
signal_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/PIG%i-ARDSnet.npz"
pign=2

include_signals=True
os.path.isdir(root)

cases = ["1_Ctis", "2_kcw","3_kd","4_palv","5_gammainsp","6_gammaexp"]
titles = ["Tissue \n stiffness - $C$", "Chest-wall\nstiffness - $K_{cw}$",
          "Diaphragm\nstiffness - $K_d$", "Alveolar\npressure - $P_{alv}$",
          "Inspiratory\ngamma - $\gamma_{I}$", "Expiratory\ngamma - $\gamma_{E}$"]

baseline = root+"0_baseline/output/Signals/"

fig,axes = plt.subplots(nrows=2, ncols=6,figsize=(12,4),dpi=200)

color1 = 'navy'
color2 = 'royalblue'
color3 = 'royalblue'

for e,(case,title) in enumerate(zip(cases,titles)):
    
    plus = root+case+"-plus/output/Signals/"
    minus = root+case+"-minus/output/Signals/"
        
    t, p, f, v = retrieve_fields(baseline)
    tp, pp, fp, vp = retrieve_fields(plus)
    tm, pm, fm, vm = retrieve_fields(minus)
    
    
    mat = np.load(signal_path%(pign,pign))
    mv = mat['volume'].flatten()
    mf = mat['flow'].flatten()
    mp = mat['pressure'].flatten()
    mt = mat['time'].flatten()
    
    lw = 1.0
    
    # Pressure
    ax = axes[0,e]
    ax.plot(t,p, color=color1, ls='--',lw=lw)
    ax.plot(tm,pm, color=color2, ls='-',lw=lw,alpha=0.75)
    ax.plot(tp,pp, color=color3,ls='-', lw=lw,alpha=0.75)
    ax.fill_between(t,pm,pp,color=color1, alpha=0.5)
    #ax.plot(mt,mp, ls='--')
    ax.set_ylim((0.0,35))
    if e>0:
        ax.set_yticks([])
    else:
        ax.set_ylabel("Airway pressure\n(cmH$_{2}$O)")
        
    ax.set_title(title, size=10)
    
    
    ax = axes[1,e] 
    ax.plot(t,f, color=color1,ls='--',lw=lw)
    ax.plot(tm,fm, color=color2, ls='-',lw=lw,alpha=0.75)
    ax.plot(tp,fp, color=color3,ls='-',lw=lw,alpha=0.75)
    ax.fill_between(t,fm,fp, color=color1, alpha=0.2)
    ax.set_ylim(-1.25,0.75)
    ax.axhline(0.0, ls='--',lw=1.0,color='k',alpha=0.25)
    
    if e>0:
        ax.set_yticks([])
    else:
        ax.set_ylabel("Airway flow\n(L/s)")
    
    ax.set_xlabel("Time (s)")
    
# Arrows and text
# First block
x0,y0 = (0.5,15) ; x1,y1 = (1.0,25)
axes[0,0].annotate('',                    # no text
                xy=(x1, y1),            # arrow tip
                xytext=(x0, y0),        # arrow start
                arrowprops=dict(arrowstyle='->',linewidth=1.0))
axes[0,0].text(x1,y1,"$C$")

x0,y0 = (0.90,0.3) ; x1,y1 = (1.05,0.74)
axes[1,0].annotate('',                    # no text
                xy=(x1, y1),            # arrow tip
                xytext=(x0, y0),        # arrow start
                arrowprops=dict(arrowstyle='->',linewidth=1.0))
axes[1,0].text(x1,y1-0.2,"$C$")


x0,y0 = (1.8,0.3) ; x1,y1 = (1.70,-0.15)
axes[1,0].annotate('',                    # no text
                xy=(x1, y1),            # arrow tip
                xytext=(x0, y0),        # arrow start
                arrowprops=dict(arrowstyle='->',linewidth=1.0))
axes[1,0].text(x1+-0.2,y1+0.1,"$C$")

# Fourth block
x0,y0 = (0.5,15) ; x1,y1 = (1.0,25)
axes[0,3].annotate('',                    # no text
                xy=(x1, y1),            # arrow tip
                xytext=(x0, y0),        # arrow start
                arrowprops=dict(arrowstyle='->',linewidth=1.0))
axes[0,3].text(x1,y1,"$P_{alv}$")

x0,y0 = (1.3,0.1) ; x1,y1 = (1.5,0.6)
axes[1,3].annotate('',                    # no text
                xy=(x1, y1),            # arrow tip
                xytext=(x0, y0),        # arrow start
                arrowprops=dict(arrowstyle='->',linewidth=1.0))
axes[1,3].text(x1+0.05,y1-0.2,"$P_{alv}$")

# Fifth block
x0,y0 = (0.20,20) ; x1,y1 = (0.15,33)
axes[0,4].annotate('',                    # no text
                xy=(x1, y1),            # arrow tip
                xytext=(x0, y0),        # arrow start
                arrowprops=dict(arrowstyle='->',linewidth=1.0))
axes[0,4].text(x1-0.2,y1-5,"$\gamma_{I}$")

# Sixth block
x1,y1 = (0.90,0.3) ; x0,y0 = (1.05,0.74)
axes[1,5].annotate('',                    # no text
                xy=(x1, y1),            # arrow tip
                xytext=(x0, y0),        # arrow start
                arrowprops=dict(arrowstyle='->',linewidth=1.0))
axes[1,5].text(x1,y1-0.1,"$\gamma_{E}$")


x1,y1 = (1.8,0.4) ; x0,y0 = (1.70,-0.05)
axes[1,5].annotate('',                    # no text
                xy=(x1, y1),            # arrow tip
                xytext=(x0, y0),        # arrow start
                arrowprops=dict(arrowstyle='->',linewidth=1.0))
axes[1,5].text(x1+-0.2,y1+0.1,"$\gamma_{E}$")




#plt.tight_layout()
#fig.suptitle(title)
plt.savefig('./figures/sensibility.pdf',dpi=300, bbox_inches='tight')
