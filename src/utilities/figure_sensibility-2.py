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
        try:
            pressures = np.load(path_to_data+"%spresionestodas.npy"%model).flatten()*10.1972 # cmH2O
            times = np.load(path_to_data+"%seffectivetimes.npy"%model).flatten() # s
            fluxes = -np.load(path_to_data+"%sfluxes.npy"%model).flatten() # L/s
            fluxes[0] = fluxes[1]
            volumes = np.load(path_to_data+"%svolumenes.npy"%model).flatten() # 
            volumes -= volumes[0]
        except:
            return False, (None,None,None,None)
            
        return True, (times, pressures, fluxes, volumes)

root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/sensitivity-PIG5/"
signal_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/PIG%i-ARDSnet.npz"
pign=5

include_signals=True
os.path.isdir(root)

cases = ["1_Ctis", "2_kcw","3_kd","4_palv","5_gammainsp","6_gammaexp"]
titles = ["Tissue \n stiffness - $C$", "Chest-wall\nstiffness - $K_{cw}$",
          "Diaphragm\nstiffness - $K_d$", "Alveolar\npressure - $\\bar{p}$",
          "Inspiratory\ngamma - $\gamma_{I}$", "Expiratory\ngamma - $\gamma_{E}$"]

baseline = root+"0_baseline/output/Signals/"

fig,axes = plt.subplots(nrows=3, ncols=6,figsize=(12,6),dpi=200)

color1 = 'navy'
color2 = 'royalblue'
color3 = 'darkred'

# Alpha value for color bands
band1 = 0.75
band2 = 0.50
band3 = 0.25

# Line width for plots 
lw = 0.5

exceptions = []

for e,(case,title) in enumerate(zip(cases,titles)):
    
    plus10 = root+case+"-plus/output/Signals/"
    minus10 = root+case+"-minus/output/Signals/"
    plus05 = root+case+"-plus-5/output/Signals/"
    minus05 = root+case+"-minus-5/output/Signals/"
    plus20 = root+case+"-plus-20/output/Signals/"
    minus20 = root+case+"-minus-20/output/Signals/"
    plus50 = root+case+"-plus-50/output/Signals/"
    minus50 = root+case+"-minus-50/output/Signals/"
    plus90 = root+case+"-plus-90/output/Signals/"
    minus90 = root+case+"-minus-90/output/Signals/"
    
    # Read data
    flag, (t, p, q, v) = retrieve_fields(baseline)
    
    fm05, (tm05, pm05, qm05, vm05) = retrieve_fields(minus05)
    fm10, (tm10, pm10, qm10, vm10) = retrieve_fields(minus10)
    fm20, (tm20, pm20, qm20, vm20) = retrieve_fields(minus20)
    
    fp05, (tp05, pp05, qp05, vp05) = retrieve_fields(plus05)
    fp10, (tp10, pp10, qp10, vp10) = retrieve_fields(plus10)
    fp20, (tp20, pp20, qp20, vp20) = retrieve_fields(plus20)       
    
    
    if e in [1,2]:
        fm50, (tm50, pm50, qm50, vm50) = retrieve_fields(minus50)
        fm90, (tm90, pm90, qm90, vm90) = retrieve_fields(minus90)

        fp50, (tp50, pp50, qp50, vp50) = retrieve_fields(plus50)
        fp90, (tp90, pp90, qp90, vp90) = retrieve_fields(plus90)
    
    # Read signals
    mat = np.load(signal_path%(pign,pign))
    mv = mat['volume'].flatten()
    mf = mat['flow'].flatten()
    mp = mat['pressure'].flatten()
    mt = mat['time'].flatten()
     
    for fcn, fl in enumerate([flag, fm05, fm10, fm20, fp05, fp10, fp20]):
        if not fl:
            exceptions += [case, fl]
        
    if e ==1:
        data = {'p':{'pm90':pm90,
                    'pm50':pm50,
                    'pm20':pm20,
                    'pm10':pm10,
                    'pm05':pm05,
                    'base':p,
                    'pp05':pp05,
                    'pp10':pp10,
                    'pp20':pp20,
                    'pp50':pp50,
                    'pp90':pp90,

                    }
                }

    
    # Pressure
    ax = axes[0,e]
    if flag: ax.plot(t,p, color='k', ls='-',lw=lw)
    if fm05: ax.plot(tm05,pm05, color=color2, ls='-',lw=lw,alpha=0.5)
    if fm10: ax.plot(tm10,pm10, color=color2, ls='-',lw=lw,alpha=0.5)
    if fm20: ax.plot(tm20,pm20, color=color2, ls='-',lw=lw,alpha=0.5)

    
    if fp05: ax.plot(tp05,pp05, color=color3,ls='-', lw=lw,alpha=0.5)
    if fp10: ax.plot(tp10,pp10, color=color3,ls='-', lw=lw,alpha=0.5)
    if fp20: ax.plot(tp20,pp20, color=color3,ls='-', lw=lw,alpha=0.5)
    
    
    if fp20 and fp10: ax.fill_between(t,pp20,pp10,color=color3, alpha=band3)
    if fp10 and fp05: ax.fill_between(t,pp10,pp05,color=color3, alpha=band2)
    if fp05 and flag: ax.fill_between(t,pp05,p,color=color3, alpha=band1)
    if flag and fm05: ax.fill_between(t,p,pm05,color=color2, alpha=band1)
    if fm10 and fm05: ax.fill_between(t,pm05,pm10,color=color2, alpha=band2)
    if fm20 and fm10: ax.fill_between(t,pm10,pm20,color=color2, alpha=band3) 
    
    if e in [1,2]:
        if fp90 and fp50: ax.fill_between(t,pp90,pp50,color=color3, alpha=band3)
        if fp50 and fp20: ax.fill_between(t,pp50,pp20,color=color3, alpha=band2)
        if fm20 and fm50: ax.fill_between(t,pm20,pm50,color=color2, alpha=band2)
        if fm50 and fm90: ax.fill_between(t,pm50,pm90,color=color2, alpha=band3) 
    
    #ax.plot(mt,mp, ls='--')
    ax.set_ylim((0.0,35))
    if e>0:
        ax.set_yticks([])
    else:
        ax.set_ylabel("Airway pressure\n(cmH$_{2}$O)")
        
    ax.set_title(title, size=10)
    
    
    ax = axes[1,e] 
    # Flow
    if flag: ax.plot(t,q, color='k', ls='-',lw=lw)
    if fm05: ax.plot(tm05,qm05, color=color2, ls='-',lw=lw,alpha=0.5)
    if fm10: ax.plot(tm10,qm10, color=color2, ls='-',lw=lw,alpha=0.5)
    if fm20: ax.plot(tm20,qm20, color=color2, ls='-',lw=lw,alpha=0.5)

    
    if fp05: ax.plot(tp05,qp05, color=color3,ls='-', lw=lw,alpha=0.5)
    if fp10: ax.plot(tp10,qp10, color=color3,ls='-', lw=lw,alpha=0.5)
    if fp20: ax.plot(tp20,qp20, color=color3,ls='-', lw=lw,alpha=0.5)
    
    
    if fp20 and fp10: ax.fill_between(t,qp20,qp10,color=color3, alpha=band3)
    if fp10 and fp05: ax.fill_between(t,qp10,qp05,color=color3, alpha=band2)
    if fp05 and flag: ax.fill_between(t,qp05,q,color=color3, alpha=band1)
    if flag and fm05: ax.fill_between(t,q,qm05,color=color2, alpha=band1)
    if fm10 and fm05: ax.fill_between(t,qm05,qm10,color=color2, alpha=band2)
    if fm20 and fm10: ax.fill_between(t,qm10,qm20,color=color2, alpha=band3) 
    
    if e in [1,2]:
   
        if fp90 and fp50: ax.fill_between(t,qp90,qp50,color=color3, alpha=band3)
        if fp50 and fp20: ax.fill_between(t,qp50,qp20,color=color3, alpha=band2)
        if fm20 and fm50: ax.fill_between(t,qm20,qm50,color=color2, alpha=band2)
        if fm50 and fm90: ax.fill_between(t,qm50,qm90,color=color2, alpha=band3) 

    ax.axhline(0.0, ls='--',lw=1.0,color='k',alpha=0.25)
    
    if e>0:
        ax.set_yticks([])
    else:
        ax.set_ylabel("Airway flow\n(L/s)")
        
    # Volume
    ax = axes[2,e] 
    if flag: ax.plot(t,v, color='k', ls='-',lw=lw)
    if fm05: ax.plot(tm05,vm05, color=color2, ls='-',lw=lw,alpha=0.5)
    if fm10: ax.plot(tm10,vm10, color=color2, ls='-',lw=lw,alpha=0.5)
    if fm20: ax.plot(tm20,vm20, color=color2, ls='-',lw=lw,alpha=0.5)

    
    if fp05: ax.plot(tp05,vp05, color=color3,ls='-', lw=lw,alpha=0.5)
    if fp10: ax.plot(tp10,vp10, color=color3,ls='-', lw=lw,alpha=0.5)
    if fp20: ax.plot(tp20,vp20, color=color3,ls='-', lw=lw,alpha=0.5)
    
    
    if fp20 and fp10: ax.fill_between(t,vp20,vp10,color=color3, alpha=band3)
    if fp10 and fp05: ax.fill_between(t,vp10,vp05,color=color3, alpha=band2)
    if fp05 and flag: ax.fill_between(t,vp05,v,color=color3, alpha=band1)
    if flag and fm05: ax.fill_between(t,v,vm05,color=color2, alpha=band1)
    if fm10 and fm05: ax.fill_between(t,vm05,vm10,color=color2, alpha=band2)
    if fm20 and fm10: ax.fill_between(t,vm10,vm20,color=color2, alpha=band3) 


    if e in [2,3]:
   
        if fp90 and fp50: ax.fill_between(t,vp90,vp50,color=color3, alpha=band3)
        if fp50 and fp20: ax.fill_between(t,vp50,vp20,color=color3, alpha=band2)
        if fm20 and fm50: ax.fill_between(t,vm20,vm50,color=color2, alpha=band2)
        if fm50 and fm90: ax.fill_between(t,vm50,vm90,color=color2, alpha=band3) 

    ax.axhline(0.0, ls='--',lw=1.0,color='k',alpha=0.25)
    
    if e>0:
        ax.set_yticks([])
    else:
        ax.set_ylabel("Tidal volume \n(L)")
    

    
    ax.set_xlabel("Time (s)")

if pign == 5:
    # Arrows and text
    # First block
    x0,y0 = (0.5,15) ; x1,y1 = (1.0,25)
    axes[0,0].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,0].text(x1,y1,"$c$")
    
    x0,y0 = (1.5,5) ; x1,y1 = (1.5,16)
    axes[0,0].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,0].text(x1,y1,"$c$")    
    
    x0,y0 = (0.90,0.3) ; x1,y1 = (1.05,0.68)
    axes[1,0].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[1,0].text(x1,y1-0.15,"$c$")
    
    
    x0,y0 = (1.8,0.3) ; x1,y1 = (1.70,-0.15)
    axes[1,0].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[1,0].text(x1+-0.2,y1+0.1,"$c$")
    
    # Second block; K_cw
    x0,y0 = (0.5,15) ; x1,y1 = (1.0,25)
    axes[0,1].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,1].text(x1,y1,"$K_{cw}$")
    
    x0,y0 = (1.5,5) ; x1,y1 = (1.5,16)
    axes[0,1].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,1].text(x1,y1,"$K_{cw}$")    

    # Third block; K_d
    x0,y0 = (0.5,15) ; x1,y1 = (1.0,25)
    axes[0,2].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,2].text(x1,y1,"$K_{d}$")
    

    x0,y0 = (1.5,5) ; x1,y1 = (1.5,16)
    axes[0,2].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,2].text(x1,y1,"$K_{d}$")    

    # Fourth block
    x0,y0 = (0.5,15) ; x1,y1 = (1.0,25)
    axes[0,3].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,3].text(x1,y1,"$\\bar{p}$")
    

    x0,y0 = (1.5,5) ; x1,y1 = (1.5,16)
    axes[0,3].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,3].text(x1,y1,"$\\bar{p}$")

    
    x0,y0 = (1.3,0.1) ; x1,y1 = (1.5,0.6)
    axes[1,3].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[1,3].text(x1+0.05,y1-0.2,"$\\bar{p}$")
    
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
    
    # Volume tags
    x0,y0 = (1.7,0.0) ; x1,y1 = (2.0,0.1)
    axes[2,0].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[2,0].text(x1-0.2,y1,"$c$")
    
    x0,y0 = (1.7,0.0) ; x1,y1 = (2.0,0.1)
    axes[2,1].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[2,1].text(x1-0.3,y1,"$k_{cw}$")
    

    x0,y0 = (1.7,0.0) ; x1,y1 = (2.0,0.1)
    axes[2,2].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[2,2].text(x1-0.3,y1,"$k_{d}$")
        

    # Fourth block
    x1,y1 = (1.55,-0.04) ; x0,y0 = (2.0,0.1)
    axes[2,3].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[2,3].text(x1-0.2,y1+0.05,"$\\bar{p}$")
    
    # Sixth block
    x0,y0 = (1.25,0.07) ; x1,y1 = (1.75,0.17)
    axes[2,5].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[2,5].text(x1-0.2,y1,"$\gamma_{E}$")
    #plt.tight_layout()
    #fig.suptitle(title)
elif pign == 2:
    
    # Arrows and text
    # First block
    x0,y0 = (0.5,15) ; x1,y1 = (1.0,30)
    axes[0,0].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,0].text(x1,y1,"$c$")
    
    x0,y0 = (1.20,0.2) ; x1,y1 = (1.50,0.55)
    axes[1,0].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[1,0].text(x1,y1-0.2,"$c$")
    
    
    x0,y0 = (2.2,0.3) ; x1,y1 = (1.70,-0.15)
    axes[1,0].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[1,0].text(x1+-0.2,y1+0.1,"$c$")
    
    # Fourth block
    x0,y0 = (0.5,15) ; x1,y1 = (1.0,25)
    axes[0,3].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,3].text(x1,y1,"$\\bar{p}$")
    
    x0,y0 = (1.3,0.1) ; x1,y1 = (1.5,0.6)
    axes[1,3].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[1,3].text(x1+0.05,y1-0.2,"$\\bar{p}$")
    
    # Fifth block
    x0,y0 = (0.20,20) ; x1,y1 = (0.15,33)
    axes[0,4].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,4].text(x1-0.2,y1-5,"$\gamma_{I}$")
    
    # Sixth block
    d = 0.4
    x1,y1 = (0.90+d,0.3) ; x0,y0 = (1.05+d,0.74)
    axes[1,5].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[1,5].text(x1,y1-0.1,"$\gamma_{E}$")
    
    
    x1,y1 = (1.8+d,0.4) ; x0,y0 = (1.70+d,-0.05)
    axes[1,5].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[1,5].text(x1+-0.2,y1+0.1,"$\gamma_{E}$")
    
    # Volume tags
    x0,y0 = (2.1,-0.02) ; x1,y1 = (2.5,0.1)
    axes[2,0].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[2,0].text(x1-0.2,y1,"$c$")

    # First block
    x0,y0 = (0.5,15) ; x1,y1 = (1.0,30)
    axes[0,1].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[0,1].text(x1,y1,"$k_{cw}$")

    x0,y0 = (2.1,-0.02) ; x1,y1 = (2.5,0.1)
    axes[2,1].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[2,1].text(x1-0.2,y1,"$k_{cw}$")
    
    # Fourth block
    x1,y1 = (1.55,-0.04) ; x0,y0 = (2.0,0.1)
    axes[2,3].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[2,3].text(x1-0.2,y1+0.05,"$\\bar{p}$")
    
    # Sixth block
    x0,y0 = (1.25,0.07) ; x1,y1 = (1.75,0.17)
    axes[2,5].annotate('',                    # no text
                    xy=(x1, y1),            # arrow tip
                    xytext=(x0, y0),        # arrow start
                    arrowprops=dict(arrowstyle='->',linewidth=1.0))
    axes[2,5].text(x1-0.2,y1,"$\gamma_{E}$")
    #plt.tight_layout()
    #fig.suptitle(title)



plt.savefig('./figures/sensibility.pdf',dpi=300, bbox_inches='tight')

print(exceptions)