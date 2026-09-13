# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 17:25:49 2025

@author: angus
"""

# signal_profiles.py

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def fit_segment(time, signal, kind):
    def linear(x, a, b): return a * x + b
    def exp(x, a, b): return -a * (1 - np.exp(-b * x))

    if kind == "linear":
        return curve_fit(linear, time, signal)[0], linear
    elif kind == "exp":
        return curve_fit(exp, time, signal)[0], exp
    elif kind == "zero":
        return (), lambda x: 0.0
    else:
        raise ValueError(f"Unknown fit type: {kind}")


def create_flow_function(time_arr, flow, segments, t_cycle=2.0):
    segment_fits = []

    for seg in segments:
        mask = (time_arr >= seg["t_start"]) & (time_arr < seg["t_end"])
        t_seg = time_arr[mask]
        f_seg = flow[mask]

        params, fn = fit_segment(t_seg, f_seg, seg["type"])
        segment_fits.append({
            "t_start": seg["t_start"],
            "t_end": seg["t_end"],
            "fn": fn,
            "params": params
        })

    def flow_func(t):
        t_mod = t % t_cycle
        for seg in segment_fits:
            if seg["t_start"] <= t_mod < seg["t_end"]:
                return seg["fn"](t_mod, *seg["params"])
        return 0.0

    return flow_func


def create_pressure_function(time_arr, pressure, segments, t_cycle=2.0):
    segment_fits = []

    for seg in segments:
        mask = (time_arr >= seg["t_start"]) & (time_arr < seg["t_end"])
        t_seg = time_arr[mask]
        p_seg = pressure[mask]

        if len(p_seg) < 2:
            raise ValueError(f"Segment {seg} has too few data points for fitting.")

        params, fn = fit_segment(t_seg, p_seg, seg["type"])
        segment_fits.append({
            "t_start": seg["t_start"],
            "t_end": seg["t_end"],
            "fn": fn,
            "params": params
        })

    plateau_measured = pressure[time_arr >= segments[0]["t_start"]][0]

    def pressure_func(t, plateau_pressure_sim):
        t_mod = t % t_cycle
        scale = plateau_pressure_sim / plateau_measured
    
        for i, seg in enumerate(segment_fits):
            is_last = (i == len(segment_fits) - 1)
    
            if seg["t_start"] <= t_mod < seg["t_end"]:
                return seg["fn"](t_mod, *seg["params"]) * scale
    
            # Special case: t == Tcycle → t_mod == 0.0, use extrapolation from last segment
            if is_last and np.isclose(t_mod, 0.0, atol=1e-8):
                return seg["fn"](seg["t_end"], *seg["params"]) * scale
    
        raise ValueError(f"Time {t_mod:.4f} is outside all defined segments.")

    return pressure_func




if __name__ == "__main__":

    
    manager = {"PIG2-ARDSnet":{"Tsyr":0.435,
                               "Tpausa":0.560,
                               "Texp":1.870,
                               "Pplat":21.7,
                               "PEEP":11.0,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG2/PIG2-ARDSnet.npz',
                               "Target volume":0.3527,
                               },
               "PIG3-ARDSnet":{"Tsyr":0.455,
                               "Tpausa":0.435,
                               "Texp":1.100,
                               "Pplat":26.3,
                               "PEEP":11.3 ,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG3/PIG3-ARDSnet.npz',
                               "Target volume":0.3870,
                               },
               "PIG4-ARDSnet":{"Tsyr":0.48,
                               "Tpausa":0.28,
                               "Texp":1.73,
                               "Pplat":18.5,
                               "PEEP": 10.7,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG4/PIG4-ARDSnet.npz',
                               "Target volume":0.411,
                               },
               "PIG5-ARDSnet":{"Tsyr":0.375,
                               "Tpausa":0.375,
                               "Texp":1.25,
                               "Pplat":20.6,
                               "PEEP": 10.8,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/PIG5-ARDSnet.npz',
                               "Target volume":0.4011,
                               },
               "PIG6-ARDSnet":{"Tsyr":0.540,
                               "Tpausa":0.265,
                               "Texp":1.340,
                               "Pplat":22.7,
                               "PEEP": 10.7,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/PIG6-ARDSnet.npz',
                               "Target volume":0.2995,
                               },               
               "Template":{"Tsyr":None,
                               "Tpausa":None,
                               "Texp":None,
                               "Pplat":None,
                               "Signal path":None,
                               "Target volume":None,
                               },
               }
    
   
    '''
    
# =============================================================================
#     PIG 3
# =============================================================================
    flow_segments = [
        {"t_start": 0.0,   "t_end": 0.150, "type": "exp"},
        {"t_start": 0.150, "t_end": 0.250, "type": "linear"},
        {"t_start": 0.250, "t_end": 0.350, "type": "linear"},
        {"t_start": 0.350, "t_end": 0.400, "type": "linear"},
        {"t_start": 0.400, "t_end": 0.450, "type": "linear"},
        {"t_start": 0.450, "t_end": 0.540, "type": "linear"},
        {"t_start": 0.810, "t_end": 1.99, "type": "zero"},
    ]
    
    pressure_segments = [
        {"t_start": 0.890, "t_end": 0.920, "type": "linear"},
        {"t_start": 0.920, "t_end": 0.950, "type": "linear"},
        {"t_start": 0.950, "t_end": 1.020, "type": "linear"},
        {"t_start": 1.020, "t_end": 1.300, "type": "linear"},
        {"t_start": 1.300, "t_end": 1.99, "type": "linear"},
    ]

# =============================================================================
#     PIG 4
# =============================================================================
    flow_segments = [
        {"t_start": 0.0,   "t_end": 0.075, "type": "linear"},
        {"t_start": 0.075, "t_end": 0.120, "type": "linear"},
        {"t_start": 0.120, "t_end": 0.440, "type": "linear"},
        {"t_start": 0.440, "t_end": 0.465, "type": "linear"},
        {"t_start": 0.465, "t_end": 0.490, "type": "linear"},
        {"t_start": 0.490, "t_end": 0.550, "type": "linear"},
        {"t_start": 0.550, "t_end": 0.810, "type": "zero"},
        {"t_start": 0.810, "t_end": 2.000, "type": "linear"},
    ]
    
    pressure_segments = [
        {"t_start": 0.760, "t_end": 0.785, "type": "linear"},
        {"t_start": 0.785, "t_end": 0.800, "type": "linear"},

        {"t_start": 0.800, "t_end": 0.850, "type": "linear"},
        {"t_start": 0.850, "t_end": 0.950, "type": "linear"},
        {"t_start": 0.950, "t_end": 1.000, "type": "linear"},
        {"t_start": 1.000, "t_end": Tcycle, "type": "linear"},
    ]    
    
# =============================================================================
#     PIG 5
# =============================================================================
    flow_segments = [
        {"t_start": 0.0,   "t_end": 0.375, "type": "exp"},
        {"t_start": 0.375, "t_end": 0.470, "type": "linear"},
        {"t_start": 0.470, "t_end": 0.750, "type": "zero"},
        {"t_start": 0.750, "t_end": 0.810, "type": "linear"},
        {"t_start": 0.810, "t_end": 2.000, "type": "linear"},
    ]
    
    pressure_segments = [
        {"t_start": 0.750, "t_end": 0.800, "type": "linear"},
        {"t_start": 0.800, "t_end": 0.850, "type": "linear"},
        {"t_start": 0.850, "t_end": 0.900, "type": "linear"},
        {"t_start": 0.900, "t_end": 0.950, "type": "linear"},
        {"t_start": 0.950, "t_end": 2.000, "type": "linear"},
    ]
# =============================================================================
#     PIG 6
# =============================================================================
    flow_segments = [
        {"t_start": 0.0,   "t_end": 0.075, "type": "exp"},
        {"t_start": 0.075, "t_end": 0.470, "type": "linear"},
        {"t_start": 0.470, "t_end": 0.485, "type": "linear"},
        {"t_start": 0.485, "t_end": 0.510, "type": "linear"},
        {"t_start": 0.510, "t_end": 0.570, "type": "linear"},
        {"t_start": 0.575, "t_end": 0.810, "type": "zero"},
        {"t_start": 0.810, "t_end": 2.000, "type": "linear"},
    ]
    
    pressure_segments = [
        {"t_start": 0.805, "t_end": 0.825, "type": "linear"},
        {"t_start": 0.825, "t_end": 0.850, "type": "linear"},
        {"t_start": 0.850, "t_end": 0.900, "type": "linear"},
        {"t_start": 0.900, "t_end": 0.940, "type": "linear"},
        {"t_start": 0.940, "t_end": 1.150, "type": "linear"},
        {"t_start": 1.150, "t_end": 2.000, "type": "linear"},
    ]

    
    '''
    
    
    pig_no = 6
    codename = "PIG%i-ARDSnet"%pig_no
    # Import relevant function
    signalpath = manager[codename]['Signal path']

    # Load the experimental signal
    npz = np.load(signalpath)
    Paw = npz['pressure']
    flow = npz['flow']
    vol = npz['volume']
    time = npz['time']

    # Definition of times
    Tsyr = manager[codename]["Tsyr"]
    Tpausa = manager[codename]["Tpausa"]
    Texp = manager[codename]["Texp"]
    Tinsp = Tsyr+Tpausa
    Tcycle = Texp+Tinsp
    
    interpolator_type = 'linear'
    
    # Determine the number of cycles
    Tcycle = Tsyr+Tpausa+Texp

    fPaw = interp1d(time,Paw,kind=interpolator_type) # Signal
    fflow = interp1d(time,flow,kind=interpolator_type) # Signal
    fvol = interp1d(time,vol,kind=interpolator_type) # Signal
    
    function_manager = {"PIG2":{"flow_segments":[
                                {"t_start": 0.0,   "t_end": 0.145, "type": "exp"},
                                {"t_start": 0.145, "t_end": 0.240, "type": "linear"},
                                {"t_start": 0.240, "t_end": 0.360, "type": "linear"},
                                {"t_start": 0.360, "t_end": 0.400, "type": "linear"},
                                {"t_start": 0.400, "t_end": 0.490, "type": "linear"},
                                {"t_start": 0.490, "t_end": 0.520, "type": "linear"},
                                {"t_start": 0.810, "t_end": 2.865, "type": "zero"},],
                                "pressure_segments":[{"t_start": 0.995, "t_end": 1.050, "type": "linear"},
                                {"t_start": 1.050, "t_end": 1.100, "type": "linear"},
                                {"t_start": 1.100, "t_end": 1.150, "type": "linear"},
                                {"t_start": 1.150, "t_end": 1.290, "type": "linear"},
                                {"t_start": 1.290, "t_end": 1.500, "type": "linear"},
                                {"t_start": 1.150, "t_end": 1.700, "type": "linear"},
                                {"t_start": 1.700, "t_end": 2.865, "type": "linear"},]
                                },
                        "PIG3":{"flow_segments": [
                                {"t_start": 0.0,   "t_end": 0.150, "type": "exp"},
                                {"t_start": 0.150, "t_end": 0.250, "type": "linear"},
                                {"t_start": 0.250, "t_end": 0.350, "type": "linear"},
                                {"t_start": 0.350, "t_end": 0.400, "type": "linear"},
                                {"t_start": 0.400, "t_end": 0.450, "type": "linear"},
                                {"t_start": 0.450, "t_end": 0.540, "type": "linear"},
                                {"t_start": 0.810, "t_end": 1.99, "type": "zero"},],
                                "pressure_segments":[
                                {"t_start": 0.890, "t_end": 0.920, "type": "linear"},
                                {"t_start": 0.920, "t_end": 0.950, "type": "linear"},
                                {"t_start": 0.950, "t_end": 1.020, "type": "linear"},
                                {"t_start": 1.020, "t_end": 1.300, "type": "linear"},
                                {"t_start": 1.300, "t_end": 1.99, "type": "linear"},]
                                },
                        "PIG4":{"flow_segments":[
                                {"t_start": 0.0,   "t_end": 0.075, "type": "linear"},
                                {"t_start": 0.075, "t_end": 0.120, "type": "linear"},
                                {"t_start": 0.120, "t_end": 0.440, "type": "linear"},
                                {"t_start": 0.440, "t_end": 0.465, "type": "linear"},
                                {"t_start": 0.465, "t_end": 0.490, "type": "linear"},
                                {"t_start": 0.490, "t_end": 0.550, "type": "linear"},
                                {"t_start": 0.550, "t_end": 0.810, "type": "zero"},
                                {"t_start": 0.810, "t_end": 2.49, "type": "linear"},],
                                "pressure_segments":[
                                {"t_start": 0.760, "t_end": 0.785, "type": "linear"},
                                {"t_start": 0.785, "t_end": 0.800, "type": "linear"},
                                {"t_start": 0.800, "t_end": 0.850, "type": "linear"},
                                {"t_start": 0.850, "t_end": 0.950, "type": "linear"},
                                {"t_start": 0.950, "t_end": 1.000, "type": "linear"},
                                {"t_start": 1.000, "t_end": 2.49, "type": "linear"},]
                                },
                        "PIG5":{"flow_segments":[
                                {"t_start": 0.0,   "t_end": 0.375, "type": "exp"},
                                {"t_start": 0.375, "t_end": 0.470, "type": "linear"},
                                {"t_start": 0.470, "t_end": 0.750, "type": "zero"},
                                {"t_start": 0.750, "t_end": 0.810, "type": "linear"},
                                {"t_start": 0.810, "t_end": 2.000, "type": "linear"},],
                                "pressure_segments":[
                                {"t_start": 0.750, "t_end": 0.800, "type": "linear"},
                                {"t_start": 0.800, "t_end": 0.850, "type": "linear"},
                                {"t_start": 0.850, "t_end": 0.900, "type": "linear"},
                                {"t_start": 0.900, "t_end": 0.950, "type": "linear"},
                                {"t_start": 0.950, "t_end": 2.000, "type": "linear"},
                                ]
                                },
                        "PIG6":{"flow_segments":[
                                {"t_start": 0.0,   "t_end": 0.075, "type": "exp"},
                                {"t_start": 0.075, "t_end": 0.470, "type": "linear"},
                                {"t_start": 0.470, "t_end": 0.485, "type": "linear"},
                                {"t_start": 0.485, "t_end": 0.510, "type": "linear"},
                                {"t_start": 0.510, "t_end": 0.570, "type": "linear"},
                                {"t_start": 0.575, "t_end": 0.810, "type": "zero"},
                                {"t_start": 0.810, "t_end": 2.145, "type": "linear"},],
                                "pressure_segments":[
                                {"t_start": 0.805, "t_end": 0.825, "type": "linear"},
                                {"t_start": 0.825, "t_end": 0.850, "type": "linear"},
                                {"t_start": 0.850, "t_end": 0.900, "type": "linear"},
                                {"t_start": 0.900, "t_end": 0.940, "type": "linear"},
                                {"t_start": 0.940, "t_end": 1.150, "type": "linear"},
                                {"t_start": 1.150, "t_end": 2.145, "type": "linear"},
                        ]}
                        
                }
    
    flow_segments = function_manager["PIG%i"%pig_no]["flow_segments"]
    pressure_segments = function_manager["PIG%i"%pig_no]["pressure_segments"]


    flow_fn = create_flow_function(time, flow, flow_segments)
    pressure_fn = create_pressure_function(time, Paw, pressure_segments, t_cycle=Tcycle)
    
    figure = True
    nsamples = 250
    err = 1e-3
    nsamples_inter = 40
    ttime = np.linspace(0+err,Tcycle-err,200)
    ttime2 = np.linspace(err,Tinsp-err,60)
    ttime3 = np.linspace(err+Tinsp,Tcycle-err,int(60*Texp/Tinsp))
    
    if figure:
        fig,axes = plt.subplots(nrows=2,figsize=(8,8),dpi=200)
        
        exp_mask = np.logical_and(time>Tinsp,time<Tcycle)
        exp_mask2 = np.logical_and(ttime>Tinsp,ttime<Tcycle)
        
        for e,ax in enumerate(axes):
            if e == 0: # Pressure
                ax.plot(ttime[ttime<Tcycle],fPaw(ttime)[ttime<Tcycle],color="tab:blue",label="Response")
                ax.plot(ttime[exp_mask2],fPaw(ttime)[exp_mask2],color="tab:red",label="Imposed")     
                ax.scatter(ttime3,[pressure_fn(t, manager[codename]["Pplat"] ) for t in ttime3],color="k",s=8)
                ax.fill_between(time[time<Tcycle],Paw[time<Tcycle],0,alpha=0.10,color="tab:blue")
                ax.fill_between(time[exp_mask],Paw[exp_mask],0,alpha=0.20,color="tab:red")
                ax.set_ylim((0,np.ceil(Paw.max()*1.05/5.0)*5.0))
                ax.set_xticks([])
                ax.set_ylabel("Pressure (cmH2O)")
                ax.legend(loc='upper center')
                
            if e == 1: # Flow
                ax.plot(ttime[ttime<Tinsp],fflow(ttime)[ttime<Tinsp],color="tab:red",label="Imposed")
                ax.plot(ttime[ttime>Tinsp],fflow(ttime)[ttime>Tinsp],color="tab:blue",label="Response")
                ax.scatter(ttime2,[flow_fn(t) for t in ttime2],color="k",s=8)
                ax.fill_between(time[time<Tinsp],flow[time<Tinsp],0,alpha=0.20,color="tab:red")
                ax.fill_between(time[time<Tcycle],flow[time<Tcycle],0,alpha=0.10,color="tab:blue")
                ax.set_ylabel("Flow (L/s)")
                ax.set_xticks([])
                ax.legend()
                
            
            if e == 1:
               # ax.plot(ttime,fvol(ttime),color="tab:blue")
                #ax.fill_between(time[time<Tcycle],vol[time<Tcycle],0,alpha=0.10,color="tab:blue")
                ax.set_ylabel("Flow (L/s)")
                ax.set_xlabel("Time (s)")
            
            for cy in [0]:
                
                ax.axvline(Tsyr+cy*Tcycle, ls = "--", color="k",alpha=0.25)
                ax.axvline(Tsyr+Tpausa+cy*Tcycle, ls = "--", color="k",alpha=0.25)
                ax.axvline((cy+1)*Tcycle, ls = "-", color="k",alpha=0.45)
            '''
            if e == 0:
                ax.axvline(1.500)
            if e == 1:
                ax.axvline(0.520)
            '''