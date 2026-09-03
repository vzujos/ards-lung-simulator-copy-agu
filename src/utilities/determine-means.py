# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:49:53 2026

@author: Academic
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress


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

def tau_guttmanm_raw(q,t,nwindows=5):
    
    # Find peak expiratory flow rate (PEFR)
    arg_pefr = np.argmin(q)
    # Time associated to PEFR
    t_pefr = t[arg_pefr]
    # Total dt 
    exp_dt = t.max() - t_pefr
    guttmann_dt = exp_dt/nwindows
    
    taus = []
    print("Evaluating Guttmann's tau: ")
    print("Maximum t: %.2f"%t.max())
    print("PEFT t: %.2f"%t_pefr)
    print("Window dt: %.2f"%(guttmann_dt))
    
    for i in range(nwindows):
        tstart=t_pefr+guttmann_dt*i; tend=t_pefr+guttmann_dt*(i+1)
        tmask = np.logical_and(t>tstart, t<tend)
        print("  (%i) %.2f - %.2f"%(i,tstart,tend))

        gt = t[tmask]; gq = q[tmask]
        taus += [determine_tau(gq,gt,gt.min())]
        print("  tau = %.2f"%taus[-1])
    return np.mean(taus),taus

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
        
      #  print(" > Evaluating window %.2f - %.2f"%(vwin0, vwin1))
      #  print(" > Number of points: %i"%np.count_nonzero(vmask))
      #  print(" > Identified slope: %.3f"%slopes[-1])
    
    
 #   plt.plot(q_,v_)
 #   plt.show()
    
    
    return np.mean(slopes), slopes
        

for param in ["Pplat", "Ppeak", "PEEP", "Target volume"]:
    data = []
    print("\nParameter: %s"%param)
    for subject in [2,3,4,5,6]:
        data += [parameters["PIG%i-ARDSnet"%subject][param]]
        if param != "Target volume":
            print(subject, "%.1f"%parameters["PIG%i-ARDSnet"%subject][param])
        else:
            print(subject, "%.2f"%parameters["PIG%i-ARDSnet"%subject][param])

    
    if not param in ["Target volume","Tau"]:
        print(param+": %.1f (%.1f)"%(np.mean(data),np.std(data)))
    else:
        print(param+": %.2f (%.2f)"%(np.mean(data),np.std(data)))



# %%


subjmesh = {5:'m',4:'m',2:'mf',3:'mf',6:'mf'}
root_path = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"

cases = ['PIG2-mf-per','PIG3-mf-per','PIG4-m-per','PIG5-m-per','PIG6-mf-per',]


data = {pig:{field:{"Exp":None,"Sim":None,"Rel":None} for field in ["Pplat","Ppeak","PEEP","Tau"]} for pig in [2,3,4,5,6]}

for pig,case in zip([2,3,4,5,6],cases):
    
    key = "PIG%i"%pig
    key2 = "PIG%i-ARDSnet"%pig
    
    pig_path = root_path+case+"/"
    pig_path+= "/Signals/"
    
    exp_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/%s/"%(key)
    signal_path = exp_path+key2+".npz"
    signals = np.load(signal_path)
    
    tt = signals['time']
    qq = -signals['flow']
    vv = signals['volume']
    pp = signals['pressure']
    
    t = np.load(pig_path+'effectivetimes.npy').flatten()
    p = np.load(pig_path+'presionestodas.npy').flatten()*10.1972
    q = np.load(pig_path+'fluxes.npy').flatten()
    v = np.load(pig_path+'volumenes.npy').flatten()
        
    Tsyr = parameters[key2]['Tsyr']
    Tpausa = parameters[key2]['Tpausa']
    Texp = parameters[key2]['Texp']
    Tinsp = Tsyr+Tpausa
    Tcycle = Tinsp+Texp
    
    Ppeak_sim = np.max(p)
    
    tplat = np.logical_and(t<Tinsp,t>(Tinsp-0.05))
    
    Pplat_sim = np.median(p[tplat]) 
    
    PEEP_init_sim = p[0]
    tpeep = np.logical_and(t<(Tcycle),t>(Tcycle-0.05))
    PEEP_end_sim = np.median(p[tpeep])

    tau_sim, _= tau_guttmann(-q,v,t)
    tau_exp, _ = tau_guttmann(-qq,vv,tt)

    PEEP_exp = parameters[key2]['PEEP']
    Pplat_exp = parameters[key2]['Pplat']
    Ppeak_exp = parameters[key2]['Ppeak']
    
    print("\nPIG%i"%pig)
    print(" > Ppeak (exp): %.1f | (sim): %.1f | (rel) %.1f"%(Ppeak_exp, Ppeak_sim, 100*np.abs(Ppeak_exp-Ppeak_sim)/Ppeak_exp)+"(%)")
    print(" > Pplat (exp): %.1f | (sim): %.1f | (rel) %.1f"%(Pplat_exp, Pplat_sim, 100*np.abs(Pplat_exp-Pplat_sim)/Pplat_exp)+"(%)")
    print(" > PEEPs (exp): %.1f | (sim): %.1f | (rel) %.1f"%(PEEP_exp, PEEP_init_sim, 100*np.abs(PEEP_exp-PEEP_init_sim)/PEEP_exp)+"(%)")
    print(" > Tau   (exp): %.2f | (sim): %.2f | (rel) %.1f"%(tau_exp, tau_sim, 100*np.abs(tau_exp-tau_sim)/tau_exp)+"(%)")
    
    data[pig]["Ppeak"]["Exp"] = Ppeak_exp
    data[pig]["Ppeak"]["Sim"] = Ppeak_sim
    data[pig]["Ppeak"]["Rel"] = 100*np.abs(Ppeak_exp-Ppeak_sim)/Ppeak_exp
    
    data[pig]["Pplat"]["Exp"] = Pplat_exp
    data[pig]["Pplat"]["Sim"] = Pplat_sim
    data[pig]["Pplat"]["Rel"] = 100*np.abs(Pplat_exp-Pplat_sim)/Pplat_exp
    
    data[pig]["PEEP"]["Exp"] = PEEP_exp
    data[pig]["PEEP"]["Sim"] = PEEP_init_sim
    data[pig]["PEEP"]["Rel"] = 100*np.abs(PEEP_exp-PEEP_init_sim)/PEEP_exp

    data[pig]["Tau"]["Exp"] = tau_exp
    data[pig]["Tau"]["Sim"] = tau_sim
    data[pig]["Tau"]["Rel"] = 100*np.abs(tau_exp-tau_sim)/tau_exp

    if False:
        plt.plot(tt,pp,label="Experiment")
        plt.plot(t,p,label="Simulation")
        plt.legend()
        plt.axvline(Tinsp)
        plt.title(key)
        plt.ylabel("Pressure (L/s)")
        plt.xlabel("Time (s)")
        plt.ylim((0,40))
        plt.show()
    
        plt.plot(tt,qq,label="Experiment")
        plt.plot(t,q,label="Simulation")
        plt.legend()
        plt.axvline(Tinsp)
        plt.title(key)
        plt.ylabel("Flow (L/s)")
        plt.xlabel("Time (s)")
        plt.show()

for field in ["Ppeak","Pplat","PEEP","Tau"]:
    f1=""; f2=""; f3=""
    d1=[]; d2=[]; d3=[]
    
    fmt ="%.1f" if field != "Tau" else "%.2f"
    print(field)
    for pig in [2,3,4,5,6]:
        f1 += fmt%data[pig][field]["Exp"]
        d1 += [data[pig][field]["Exp"]]
        f2 += fmt%data[pig][field]["Sim"]
        d2 += [data[pig][field]["Sim"]]
        f3 += fmt%data[pig][field]["Rel"]
        d3 += [data[pig][field]["Rel"]]
        
        if pig != 6:
            f1 += " & "; f2 += " & "; f3 += " & ";
        
   # f1 += " \\"; f2 += " \\"; f3 += " \\"
    print(f1)
    print(f2)
    print(f3)
    
    print(field+": %.2f (%.2f) [exp]"%(np.mean(d1),np.std(d1)))
    print(field+": %.2f (%.2f) [sim]"%(np.mean(d2),np.std(d2)))
    print(field+": %.1f (%.1f) [rel]"%(np.mean(d3),np.std(d3)))
