# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:01:28 2024

@author: angus
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress


def exp_decay(t,A,tau,C):
    return A*np.exp(-t/tau)+C

def determine_tau(q,t,tcut):
    
    # Isolate data for the exponential decay
    tm = t>tcut
    qq = q[tm][:-1]
    tt = t[tm][:-1]-tcut

    # Initial guesses
    A0 = qq.max() - qq.min()
    tau0 = (tt.max()-tt.min())/2
    C0 = qq.min()
    p0 = [A0, tau0, C0]
    
    # Fit
    params, covariance = curve_fit(exp_decay, tt, qq, p0=p0)
    
    A_fit, tau_fit, C_fit = params
    # Return time constant
    return tau_fit

def tau_guttmann(q,v,t,nwindows=5):
    
    # Remove last 0.001 (s) of evaluated time (avoid artifacts)
    tt = t.max()-0.001
    tm = t<tt
    q = q[tm]; t=t[tm]; v=v[tm]
    
    # Find peak expiratory flow rate (PEFR)
    arg_pefr = np.argmax(q)
    # Time associated to PEFR
    t_pefr = t[arg_pefr]
    # Ensure volume is normalized 
    v_ = v-v[0]

    # Reduce the data to the relevant domain
    mask = t>t_pefr
    q_ = q[mask]; v_ = v_[mask]
    
    # Overall volume change
    dv =  v_.max() - v_.min()
    
   # print("Overall volume: %.2f"%dv)
    slopes = []; intercepts = []
    
    ddv = dv/nwindows
    
    for i in range(nwindows):
        
        # Volume windows
        vwin0 = ddv*i; vwin1 = ddv*(i+1)
        # Masking the data
        vmask = np.logical_and(v_>vwin0, v_<vwin1)
        x = q_[vmask]; y = v_[vmask]
        # Determine the slope through least-squares regression
        res = linregress(x, y)
        m = res.slope
        b = res.intercept
        slopes += [m]
        intercepts += [b]
 
    return np.mean(slopes), slopes

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']

parameters = {"PIG2-ARDSnet":{"Tsyr":0.435,
                           "Tpausa":0.560,
                           "Texp":1.870,
                           "Pplat":21.7,
                           "PEEP":11.0,
                           "Ppeak":26.8,
                           "Signal path":'model2/signals/PIG2-ARDSnet.npz',
                           "Target volume":0.3527,
                           },
           "PIG3-ARDSnet":{"Tsyr":0.455,
                           "Tpausa":0.435,
                           "Texp":1.100,
                           "Pplat":26.3,
                           "PEEP":11.3 ,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG3-ARDSnet.npz',
                           "Target volume":0.3870,
                           },
           "PIG4-ARDSnet":{"Tsyr":0.48,
                           "Tpausa":0.28,
                           "Texp":1.73,
                           "Pplat":18.5,
                           "PEEP": 10.7,
                           "Ppeak":24.8,
                           "Signal path":'model2/signals/PIG4-ARDSnet.npz',
                           "Target volume":0.411,
                           },
           "PIG5-ARDSnet":{"Tsyr":0.375,
                           "Tpausa":0.375,
                           "Texp":1.25,
                           "Pplat":20.6,
                           "PEEP": 10.8,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG5-ARDSnet.npz',
                           "Target volume":0.4011,
                           },
           "PIG6-ARDSnet":{"Tsyr":0.540,
                           "Tpausa":0.265,
                           "Texp":1.340,
                           "Pplat":22.7,
                           "PEEP": 10.7,
                           "Ppeak":25.7,
                           "Signal path":'model2/signals/PIG6-ARDSnet.npz',
                           "Target volume":0.2995,
                           }
           }

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

def plot_row(data_manager, ax_rows, cases, ticker=5.0, autolim=True, lim=None,
             ylabel=None, xlabel=None, horizontal_line=True, fill_between=False,
             titles=None, keys = ["$c$", "$k_{cw}$", "$k_d$",  "$\\bar{p}$",
                                  "$\gamma_{I}$","$\gamma_{E}$","$\\kappa$"],
             color1='navy'):
    
    for e,case in enumerate(cases):
        
        ax = ax_rows[e]
        
        # Transform into numpy array
        data = np.array(data_manager[case])
        # Determine medium value: Fast fix for odd nonlinear behavior, something in
        # the sensibility function
        mid = data[3]#(data[2]+data[4])*0.5
        # Sample for nan values and clean values
        isnan = np.isnan(data)
        data_norm = (data-mid)/mid*100
    
        # The first group generates the array, the following groups are compared to
        # that value to determine the figure limits
        if autolim:
            if e == 0:
                lim = np.max(np.abs(data_norm[~isnan]))
            else:
                lim = np.max([lim, np.max(np.abs(data_norm[~isnan]))])
        
        # Scatter points 
        for ee, (x, y) in enumerate(zip(percentages, data_norm)):
            # The central point is intervened (percentage == 0)
            if ee == 3:
                ax.scatter(np.abs(x),0.0,color=color1,marker='o')
            else:
                if not isnan[ee]:
                    ax.scatter(np.abs(x),y,color =color1,marker='o')
        
        # Plus x%
        xdata = np.array([0,5,10,20])
        ydata = np.array([0.0,data_norm[4],data_norm[5],data_norm[6]])
        isnan = np.isnan(ydata)
        ax.plot(xdata[~isnan],ydata[~isnan],color=color1)
        ax.fill_between(xdata[~isnan],ydata[~isnan],alpha=0.20, color=color1)
        
        # Minus x%
        ydata = np.array([0.0,data_norm[2],data_norm[1],data_norm[0]])
        isnan = np.isnan(ydata)
        ax.plot(xdata[~isnan],ydata[~isnan],color=color1)
        ax.fill_between(xdata[~isnan],ydata[~isnan],alpha=0.20, color=color1)
        
        # Clean ticks
        if e!=0:
            ax.set_yticks([])
        else:
            ax.set_yticks([])
        
        if autolim:
            lim = np.ceil(lim/ticker)*ticker
            
        if horizontal_line:
            ax.axhline(0.0, color='k',ls='--',alpha=0.25)
    
    # Clean limits for the graph
    for e in range(len(cases)):
        ax_rows[e].set_ylim((-lim,lim)) # Fix limits
        if not xlabel is None: ax_rows[e].set_xlabel(xlabel) # Place xlabel
        
    # Ensure proper xticks
    ax_rows[0].set_yticks(np.linspace(-lim,lim,int(lim/ticker*2+1)))
    
    if not ylabel is None: # Place ylabel
        ax_rows[0].set_ylabel(ylabel)
    
    if not titles is None:
        for ax, title in zip(ax_rows,titles):
            ax.set_title(title)
    
    # Each case
    for e,case in enumerate(cases):
        
        ax = ax_rows[e]
        
        # Transform into numpy array
        data = np.array(data_manager[case])
        # Determine medium value: Fast fix for odd nonlinear behavior, something in
        # the sensibility function
        mid = data[3]
        # Sample for nan values and clean values
        isnan = np.isnan(data)
        data_norm = (data-mid)/mid*100
        isnan = np.isnan(data_norm)
      #  try:
        ampl = data_norm[~isnan].max() - data_norm[~isnan].min()
    #    except:
     #       ampl=1
        lims = ax.get_ylim()
        
        # Define arrow direction
        if data_norm[1] > data_norm[5]:
            # In this case, arrow points downwards
            x1,y1 = (12.5,-ampl/2) ; x0,y0 = (12.5,ampl/2)
        else:
            # Arrow points upwards
            x0,y0 = (12.5,-ampl/2) ; x1,y1 = (12.5,ampl/2)
        
        # (Required amplitude for arrow correction) and (minimal amplitude)
        if np.abs(y0-y1)<0.10*(lims[1]-lims[0]) and np.abs(y1-y0)>1e-2:
            y1 += (y1-y0)/np.abs(y1-y0)*(lims[1]-lims[0])*0.1
            y0 -= (y1-y0)/np.abs(y1-y0)*(lims[1]-lims[0])*0.1
        
        # Arrows
        if ampl > (lims[1]-lims[0])/50:

            ax.annotate('',                    # no text
                            xy=(x1, y1),            # arrow tip
                            xytext=(x0, y0),        # arrow start
                            arrowprops=dict(arrowstyle='->',linewidth=1.0))
            
            h = 0.03
            ax.text(14.5,(lims[1]-lims[0])*h,keys[e])
            

root = "C:/Users/Test/Documents/Databases/sens-p3/"
root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/sens-p3/"

signal_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/PIG%i-ARDSnet.npz"
pign=3
# %%
color1 = 'navy'


os.path.isdir(root)

cases = ["1_Ctis", "2_kcw","3_kd","4_palv","5_gammainsp","6_gammaexp","7_perm"]
titles = ["Tissue \n stiffness - $C$", "Chest-wall\nstiffness - $K_{cw}$",
          "Diaphragm\nstiffness - $K_d$", "Basal alveolar\npressure - $\\bar{p}$",
          "Inspiratory\ngamma - $\gamma_{I}$", "Expiratory\ngamma - $\gamma_{E}$", 
          "Permeability - \n$\\kappa$"]

baseline = root+"0_baseline/output/Signals/"


# Alpha value for color bands
band1 = 0.75
band2 = 0.50
band3 = 0.25

# Line width for plots 
lw = 0.5

exceptions = []

Tsyr = parameters['PIG%i-ARDSnet'%pign]['Tsyr']
Tpausa = parameters['PIG%i-ARDSnet'%pign]['Tpausa']
Tinsp = Tsyr+Tpausa

flag, (t, p, q, v) = retrieve_fields(baseline)
arg_ppeak = np.argmin(np.abs(t-Tsyr*0.995))
arg_pplat = np.argmin(np.abs(t-Tsyr-Tpausa*0.995))

percentages = [-20,-10,-5,0,5,10,20]
ticker = 5

data_manager = {"pplat":{},
                "ppeak":{},
                "vt":{},
                "tau":{},
                "peep":{}}

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

    ppeaks = []; pplats = []; vts = []; bools = []; taus = []; peeps = []

    for address in [minus20, minus10, minus05, baseline, plus05, plus10, plus20]:
        
        flag, (t, p, q, v) = retrieve_fields(address)
        bools += [flag]
        
        if flag:
            pplats += [p[arg_pplat]]; ppeaks+= [p[arg_ppeak]]; vts += [np.max(v)]
            tau = tau_guttmann(-q,v,t)
            taus += [tau[0]]
            peeps += [p[0]]
        else:
            pplats += [np.nan]; ppeaks+= [np.nan]
            vts += [np.nan]; taus += [np.nan]
            peeps += [np.nan]
    
    
    data_manager['pplat'].update({case:pplats})
    data_manager['ppeak'].update({case:ppeaks})
    data_manager['vt'].update({case:vts})
    data_manager['tau'].update({case:taus})
    data_manager['peep'].update({case:peeps})
    
    
# %%

plot_curves=False

case = "7_perm" # "7_perm"
curves_of_interest = [(case,0),
                      (case,3),
                      (case,6)]

if plot_curves: 
    
    if len(curves_of_interest)>0:
        
        fig,ax = plt.subplots(nrows=2,ncols=1, dpi=200)
        
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
    
            ppeaks = []; pplats = []; vts = []; bools = []; taus = []; peeps = []
    
            for a, address in enumerate([minus20, minus10, minus05, baseline, plus05, plus10, plus20]):
                
                flag, (t, p, q, v) = retrieve_fields(address)
                
                key = (case, a )
                
                if a<3:
                    color='b'; ls='-'
                elif a>3:
                    color='r'; ls='-'
                else:
                    color='k'; ls='--'
                if key in curves_of_interest:
                    lab = address.split("/")[-4]
                    ax[0].plot(t,q,color=color,ls=ls)
                    ax[1].plot(t,p,label=lab,color=color, ls=ls)
            
        ax[1].legend()
        
        
# %%
fig,axes = plt.subplots(nrows=4, ncols=7,figsize=(12,8),dpi=200)


plot_row(data_manager['ppeak'], axes[0,:], cases, ylabel="Peak pressure\n variation (%)",
         titles=titles,autolim=False, lim=15, ticker=5)
plot_row(data_manager['pplat'], axes[1,:], cases, autolim=False, lim=15,  ticker=5,
         ylabel="Plateau pressure\nvariation (%)")
plot_row(data_manager['peep'], axes[2,:], cases, autolim=False, lim=30,  ticker=15,
         ylabel="PEEP\nvariation (%)")
#plot_row(data_manager['vt'], axes[2,:], cases, ylabel="Tidal volume\nvariation (%)")
plot_row(data_manager['tau'], axes[3,:], cases,autolim=False, lim=50, ticker=25,
         xlabel="Parameter\nvariation (%)", ylabel="Expiration time\nconstant variation (%)")

if not os.path.isdir('./figures/'):
    os.mkdir('./figures/')
plt.savefig('./figures/sensibility-2.pdf',dpi=300, bbox_inches='tight')