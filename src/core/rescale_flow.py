# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:04:58 2025

@author: angus
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


if __name__ == '__main__':

    manager = {"PIG2-ARDSnet":{"Tsyr":0.435,
                               "Tpausa":0.580,
                               "Texp":1.850,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG2/PIG2-ARDSnet.mat',
                               "red":3,
                               "Target volume":0.3527,
                               "Flow correction":{"activate":True,
                                                  "Tcorrection_start":2.4,  # Time to start fitting
                                                  "Tcorrection_end":2.6,   # Time to end fitting
                                                  "Treplace_start":2.6,    # Start replacing here
                                                  "Treplace_end":None,
                                                  "force_zero":False,
                                                  }
                               },
               "PIG3-ARDSnet":{"Tsyr":0.455,
                               "Tpausa":0.435,
                               "Texp":1.100,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG3/PIG3-ARDSnet.mat',
                               "red":0,
                               "Target volume":0.3870,
                               "Flow correction":{"activate":True,
                                                  "Tcorrection_start":1.80,  # Time to start fitting
                                                  "Tcorrection_end":1.90,   # Time to end fitting
                                                  "Treplace_start":1.90,    # Start replacing here
                                                  "Treplace_end":None,
                                                  "force_zero":False,
                                                  }},
               "PIG4-ARDSnet":{"Tsyr":0.48,
                               "Tpausa":0.28,
                               "Texp":1.73,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG4/PIG4-ARDSnet.mat',
                               "red":26,
                               "Target volume":0.411,
                               "Flow correction":{"activate":True,
                                                  "Tcorrection_start":2.30,  # Time to start fitting
                                                  "Tcorrection_end":2.40,   # Time to end fitting
                                                  "Treplace_start":2.40,    # Start replacing here
                                                  "Treplace_end":None,
                                                  "force_zero":True,
                                                  }
                               },
               "PIG5-ARDSnet":{"Tsyr":0.375,
                               "Tpausa":0.375,
                               "Texp":1.25,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/PIG5-ARDSnet.mat',
                               "red":0,
                               "Target volume":0.4011,
                               "Flow correction":{"activate":True,
                                                  "Tcorrection_start":1.75,  # Time to start fitting
                                                  "Tcorrection_end":1.90,   # Time to end fitting
                                                  "Treplace_start":1.850,    # Start replacing here
                                                  "Treplace_end":None,
                                                  "force_zero":False
                                                  }
                               },
               "PIG6-ARDSnet":{"Tsyr":0.540,
                               "Tpausa":0.265,
                               "Texp":1.340,
                               "Signal path":'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/PIG6-ARDSnet.mat',
                               "red":21,
                               "Target volume":0.2995,
                               "Flow correction":{"activate":True,
                                                  "Tcorrection_start":1.90,  # Time to start fitting
                                                  "Tcorrection_end":2.00,   # Time to end fitting
                                                  "Treplace_start":2.00,    # Start replacing here
                                                  "Treplace_end":None,
                                                  "force_zero":True
                                                  }
                               },               
               "Template":{"Tsyr":None,
                               "Tpausa":None,
                               "Texp":None,
                               "Signal path":None,
                               "red":None,
                               "Target volume":None,
                               "Flow correction":{"activate":False,
                                                  "Tcorrection_start":None,  # Time to start fitting
                                                  "Tcorrection_end":None,   # Time to end fitting
                                                  "Treplace_start":None,    # Start replacing here
                                                  "Treplace_end":None,
                                                  "force_zero":False
                                                  }
                               },
               }
    
    case = "PIG6-ARDSnet"
    
    # Definition of times for posterior analysis of the signal
    Tsyr = manager[case]["Tsyr"]
    Tpausa = manager[case]["Tpausa"]
    Texp = manager[case]["Texp"]
    Tinsp = Tsyr+Tpausa
    Tcycle = Texp+Tinsp
    
    # Visualize intermediate
    visualize_intermediate = False
    
    # Not used
    interpolator_type = 'linear'
    
    # Path to the signal value
    signalpath = manager[case]["Signal path"]
    export_path = signalpath.split(".mat")[0]+".npz"
    save_new = False
    # Signal reduction (tailor to the signal)
    red = manager[case]["red"]
    
    # Load the experimental signal
    mat = loadmat(signalpath) 
    Paw = mat['Paw_rdata'].flatten()[red:] # Airway pressure (in cmH2O)
    flow = -mat['flow_rdata'].flatten()[red:] # Flow data (in L/s)
    vol = mat['volume'].flatten()[red:] # Volume (in L)
    time = mat['time'].flatten()[red:]
    
    # Time starts at zero
    time = time-time[0]
    
    # Target volume
    target_vol = manager[case]["Target volume"] # (L)
    
    # ---- Define time intervals (adjust as needed) ----
    activate_Tcorrection = manager[case]['Flow correction']['activate']  # Activate flow correction

    Tcorrection_start = manager[case]['Flow correction']['Tcorrection_start']  # Time to start fitting
    Tcorrection_end = manager[case]['Flow correction']['Tcorrection_end']   # Time to end fitting
    Treplace_start = manager[case]['Flow correction']['Treplace_start']    # Start replacing here
    force_zero = manager[case]['Flow correction']['force_zero']    # Start replacing here

    Treplace_end = Tcycle    # End replacing at end of cycles
    
# %%
    
    # Visualize the flow signal
    mask = np.logical_and(time<=Tcycle,time>=0.0)
    
    if visualize_intermediate:
        # Generate figure frame
        fig, axes = plt.subplots(2,1, figsize=(6,6),dpi=150)
        # Select axis [Flow]
        ax = axes[0]
        ax.plot(time[mask], flow[mask], color="k") # Original signal
        ax.set_xticks([])
        ax.set_ylabel('Flow (L/s)')
        ax.axhline(0.0, ls="--",color="gray", alpha=0.2)
        # Select axis [Vol]
        ax = axes[1]
        ax.plot(time[mask], vol[mask], color="k") # Original volume
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Volume (L)')
        ax.axhline(0.0, ls="--",color="gray", alpha=0.2)
        plt.tight_layout()
# %%
    # PEEP
    peeps = [Paw[mask][0],Paw[mask][-1]]
    peep = np.mean(peeps)
    print("PEEP: %.1f (cmH2O)"%peep)
    
    # Plateau pressure
    tpplat = np.argmin(np.abs(time-Tsyr-Tpausa*0.95))
    pplat = Paw[tpplat]
    print("Plateau pressure: %.1f (cmH2O)"%pplat)
    
    # Peak pressure
    tppeak = np.argmin(np.abs(time-Tsyr))
    ppeak = Paw[tppeak]
    print("Peak pressure: %.1f (cmH2O)"%ppeak)
    print("Peak pressure: %.1f (cmH2O)"%np.max(Paw))

    
# %%

from scipy.signal import savgol_filter, medfilt, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumtrapz

# Set smoothing method: choose from 'savgol', 'moving_avg', 'gaussian', 'butterworth', 'median'
smoothing_method = 'moving_avg'  # Change as needed

# Apply selected smoothing method
if smoothing_method == 'savgol':
    window_length = 51  # Must be odd
    polyorder = 3
    flow_proc = savgol_filter(flow, window_length=window_length, polyorder=polyorder)

elif smoothing_method == 'moving_avg':
    window_size = 31
    kernel = np.ones(window_size) / window_size
    flow_proc = np.convolve(flow, kernel, mode='same')

elif smoothing_method == 'gaussian':
    sigma = 8  # adjust based on signal characteristics
    flow_proc = gaussian_filter1d(flow, sigma=sigma)

elif smoothing_method == 'butterworth':
    fs = 1.0 / np.mean(np.diff(time))  # Sampling frequency
    cutoff = 10.0  # Hz
    order = 4
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    flow_proc = filtfilt(b, a, flow)

elif smoothing_method == 'median':
    kernel_size = 31  # Must be odd
    flow_proc = medfilt(flow, kernel_size=kernel_size)

else:
    raise ValueError(f"Unknown smoothing method: {smoothing_method}")

# Apply mask and proceed with volume integration
flow_masked = flow_proc[mask]
time_masked = time[mask]
dt = np.mean(np.diff(time_masked))



# Integrate and rescale
vol_from_flow = -cumtrapz(flow_masked, dx=dt, initial=0.0)
measured_volume = np.max(vol_from_flow)
scale_factor = target_vol / measured_volume
flow_scaled = flow_masked * scale_factor
vol_scaled = -cumtrapz(flow_scaled, dx=dt, initial=0.0)

if visualize_intermediate:
    # Plot rescaled flow and volume
    fig, axes = plt.subplots(2,1, figsize=(6,6),dpi=150)
    
    # Flow
    ax = axes[0]
    ax.plot(time_masked, flow_masked, color='gray', label=f'{smoothing_method} flow', alpha=0.5)
    ax.plot(time_masked, flow_scaled, color='b', label='Scaled flow')
    ax.set_ylabel('Flow (L/s)')
    ax.set_xticks([])
    ax.axhline(0.0, ls="--",color="gray", alpha=0.2)
    ax.legend()
    
    # Volume
    ax = axes[1]
    ax.plot(time_masked, vol_scaled, color='b', label='Integrated volume (scaled)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Volume (L)')
    ax.axhline(0.0, ls="--",color="gray", alpha=0.2)
    ax.axhline(target_vol, ls="--",color="green", alpha=0.5, label='Target Vol')
    ax.legend()
    
    plt.tight_layout()

# %%

# ---- Extract fitting range ----
if activate_Tcorrection:
    
    fit_mask = np.logical_and(time_masked >= Tcorrection_start, time_masked <= Tcorrection_end)
    replace_mask = np.logical_and(time_masked >= Treplace_start, time_masked <= Treplace_end)
    
    if np.sum(fit_mask) < 2:
        raise ValueError("Not enough points to perform linear fitting in correction interval.")
    
    # Linear fit: flow = m*t + b
    coeffs = np.polyfit(time_masked[fit_mask], flow_scaled[fit_mask], deg=1)
    flow_linear_fit = np.polyval(coeffs, time_masked[replace_mask])
    
    # ---- Replace section of flow_scaled with linear fit ----
    flow_scaled_corrected = flow_scaled.copy()
    flow_scaled_corrected[0] = 0.0
    if not force_zero:
        flow_scaled_corrected[replace_mask] = flow_linear_fit
    else:
        flow_scaled_corrected[replace_mask] = 0.0
        
    # ---- Recompute volume from corrected flow ----
    vol_scaled_corrected = -cumtrapz(flow_scaled_corrected, dx=dt, initial=0.0)
    
    # ---- Plot ----
    if visualize_intermediate:
        fig, axes = plt.subplots(2,1, figsize=(6,6),dpi=150)
        
        # Flow
        ax = axes[0]
        ax.plot(time_masked, flow_masked, color='gray', label=f'{smoothing_method} flow', alpha=0.5)
        ax.plot(time_masked, flow_scaled, color='b', label='Scaled flow')
        ax.plot(time_masked, flow_scaled_corrected, color='r', linestyle='--', label='Corrected flow')
        ax.set_ylabel('Flow (L/s)')
        ax.set_xticks([])
        ax.axhline(0.0, ls="--",color="gray", alpha=0.2)
        ax.legend()
        
        # Volume
        ax = axes[1]
        ax.plot(time_masked, vol_scaled_corrected, color='r', linestyle='--', label='Corrected volume')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Volume (L)')
        ax.axhline(0.0, ls="--",color="gray", alpha=0.2)
        ax.axhline(target_vol, ls="--",color="green", alpha=0.5, label='Target Vol')
        ax.legend()
        
        plt.tight_layout()

else:
    flow_scaled_corrected = flow_scaled.copy()
    vol_scaled_corrected = vol_scaled.copy()


# %%

# ---- Final visualization: Flow, Volume, and Pressure (Paw) ----

# Mask everything to one cycle
final_mask = np.logical_and(time >= 0.0, time <= Tcycle)

# Time and final signals for one cycle
time_final = time[final_mask]
flow_final = flow_scaled_corrected
vol_final = vol_scaled_corrected
paw_final = Paw[final_mask]  # Raw Paw (optionally smooth if noisy)

ts = 0.4
fill_color = 'orange'

# Plot all signals
fig, axes = plt.subplots(3, 1, figsize=(7, 7), dpi=250, sharex=True)

for ax in axes:
    ax.axvline(Tsyr, color="k",alpha=0.3)
    ax.axvline(Tinsp, color="k",alpha=0.3)

xticks = np.linspace(0,np.ceil(Tcycle/ts)*ts,np.ceil(Tcycle/ts).astype(int)+1)

# Flow
axes[0].plot(time_final, flow_final, color='k')
axes[0].set_ylabel("Flow (L/s)")
axes[0].axhline(0.0, ls="--", color="gray", alpha=0.3)
axes[0].fill_between(time_final,0,flow_final, color=fill_color,alpha=0.5)
axes[0].set_xticks([])

# Volume
axes[1].plot(time_final, vol_final, color='k')
axes[1].set_ylabel("Volume (L)")
axes[1].axhline(target_vol, ls="--", color="green", alpha=0.4, label="Target Volume")
axes[1].axhline(0.0, ls="--", color="gray", alpha=0.4, label="Zero-value")
axes[1].fill_between(time_final,0,vol_final, color=fill_color,alpha=0.5)
axes[1].legend()
x0,x1 = axes[1].get_xlim(); dx = x1-x0;
y0,y1 = axes[1].get_ylim(); dy = y1-y0;
axes[1].text(x0+dx*0.95,y0+dy*0.85,"$VT$: %.3f (L)"%target_vol,ha='right')


# Pressure (Paw)
axes[2].plot(time_final, paw_final, color='k')
axes[2].set_ylabel("Paw (cmH₂O)")
axes[2].set_xlabel("Time (s)")
axes[2].axhline(0.0, ls="--", color="gray", alpha=0.3)
axes[2].fill_between(time_final,0,paw_final, color=fill_color,alpha=0.5)
axes[2].set_xticks(xticks)
axes[2].set_xticklabels(["%.1f"%xt for xt in xticks])
ax.axhline(peep,ls="--",color="k",alpha=0.1)
ax.axhline(pplat,ls="--",color="k",alpha=0.1)
ax.axhline(ppeak,ls="--",color="k",alpha=0.1)

x0,x1 = axes[2].get_xlim(); dx = x1-x0;
y0,y1 = axes[2].get_ylim(); dy = y1-y0;
axes[2].text(x0+dx*0.95,y0+dy*0.65,"PEEP: %.1f (cmH2O)"%peep,ha='right')
axes[2].text(x0+dx*0.95,y0+dy*0.75,"$P_{plat}$: %.1f (cmH2O)"%pplat,ha='right')
axes[2].text(x0+dx*0.95,y0+dy*0.85,"$P_{peak}$: %.1f (cmH2O)"%ppeak,ha='right')

plt.suptitle("Final Corrected Signals for One Cycle [%s]"%case, y=1.02)
plt.tight_layout()

# %%

# Save data into a compressed .npz file
if save_new:
    np.savez(export_path,
         time=time_final,
         flow=flow_final,
         volume=vol_final,
         pressure=paw_final)

print(f"Final signals saved to: {export_path}")