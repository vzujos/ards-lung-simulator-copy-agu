# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:50:03 2024

@author: angus
"""

import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def determine_changes_of_sign(xsig,ysig,xcutoff,dt=0.0025,
                              positive=True,visualize=True,negative_cutoff=-0.1,
                              xlabel="Time (s)", ylabel="Volume (L)", 
                              dylabel="Flow (L/s)",
                              fill_color="g"):
    
    # Determine gradient of signal
    dysig = np.gradient(ysig,dt)
    
    # Reduce the length of the studied signal
    m = time<xcutoff
    
    # Change of sign of the flow function
    if positive:
        signs = dysig[m]>0.0
    else:
        signs = dysig[m]<negative_cutoff
        
    changes_of_sign = []
    
    # Detect whenever the flow signal changes signs
    for i in range(1, len(xsig[m])):
        
        # Counters
        prev = i-1; curr = i
        # If sign changes, 
        change_of_sign = (signs[prev]==False and signs[curr]==True) or \
                         (signs[prev]==True and signs[curr]==False) 
                         
        if change_of_sign:
            #print("Change of sign at: %i (prev)"%prev)
            changes_of_sign += [prev]
    
    if visualize:
        
        # Generate a figure
        s=16

        if positive: 
            fill = dysig[m]>0.0
        else:
            fill = dysig[m]<negative_cutoff
    
        fig,axes = plt.subplots(nrows=2,figsize=(8,6))
        
        axes[0].plot(xsig[m],ysig[m],alpha=0.5)
        axes[0].fill_between(xsig[m],ysig[m],0.0,fill,alpha=0.5, color=fill_color)
        axes[0].scatter(xsig[m][changes_of_sign],ysig[m][changes_of_sign],s=s,color="k")
        axes[0].set_xticks([])
        axes[0].set_ylabel(ylabel)
        
        axes[1].plot(xsig[m],dysig[m],alpha=0.5)
        axes[1].fill_between(xsig[m],dysig[m],0.0,fill,alpha=0.5, color=fill_color)
        axes[1].scatter(xsig[m][changes_of_sign],dysig[m][changes_of_sign],s=s,color="k")
        axes[1].set_ylabel(dylabel)
        axes[1].set_xlabel(xlabel)
        
    return changes_of_sign


def determine_windows(wfilter, minimum_window_length):

    window_candidates = []
    windows = []
    count=0
    open_window =False
    for i in range(1,len(wfilter)):
        
        # Open a window whenever we find a change of sign
        if not wfilter[i-1] and wfilter[i]:
            # Open window
            window_start = i
            
            # If we just changed signs, open the window
            open_window = True
            
        # As long as the window is open, augment the counter
        if open_window:
            count += 1
            
        # If the sign becomes positive, keep the last negative as the end
        if wfilter[i-1] and (not wfilter[i]) and open_window:
             window_end = i-1
             # Store info
             window_candidates += [(window_start, window_end, count)]
             # Reset
             open_window=False; count=0
    
    # Demand a minimum length to deem the window acceptable
    for window in window_candidates:
        start, end, count = window
        if count>minimum_window_length: 
            windows += [window]

    return windows

# %%
    
root_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/"

subject = "PIG3"
case = "APRV"
dt = 0.0025
path_to_signal = root_path+"%s/%s-%s.mat"%(subject,subject,case)
median_volume_peak = 0.20
all_signals_plot = False

if os.path.isfile(path_to_signal):
    print("File found")
else:
    print("File not found")
    

# Load the Matlab signal
sig = loadmat(path_to_signal)

# Load each field
time = sig['time'].flatten()
Paw = sig['Paw_rdata'].flatten()
Peso = sig['Peso_sdata'].flatten()
flow = sig['flow_rdata'].flatten()
vol = sig['volume'].flatten()

print("Maximum time registered: %.2f"%time.max())


time_cutoff = 20
m = time<time_cutoff


substract_peep = True

if substract_peep:
    peep = np.min(Paw[m])
    print("Using PEEP: %.2f (cmH2O)"%peep)
else:
    peep = 0.0

if all_signals_plot:
    
    fig,axes = plt.subplots(nrows=3,figsize=(5,5),dpi=300)
    
    ax = axes[0]
    ax.plot(time[m],vol[m],alpha=0.3)
    ax.set_xticks([])
    ax.set_ylabel("Volume (L)")
    
    
    ax = axes[1]
    ax.plot(time[m],flow[m],alpha=0.3)
    ax.set_xticks([])
    ax.set_ylabel("Flow (L/s)")
    
    
    ax = axes[2]
    
    ax.plot(time[m],Paw[m]-peep,alpha=0.3)
    ax.set_ylabel("Pressure (cmH2O)")
    ax.set_xlabel("Time (s)")

# %%





# %%

dt = 0.0025
minimum_cycle_duration = 1.80
lower_threshold_for_p_high = 300

xsig = time[m]
ysig = Paw[m] - peep
zsig = vol[m]

dysig =  np.gradient(ysig,dt)
dzsig = np.gradient(zsig,dt)

peaks, properties = find_peaks(dysig, height=lower_threshold_for_p_high,
                               distance=minimum_cycle_duration)

fill = dzsig<0.
fill2 = dysig<-50
fill3 = dysig>40

# Determine the 'outer window', for which we will look for regions with negative flow
outer_windows = determine_windows(fill,100)
# Determine the 'inner window' for which we will look for negative pressure slopes
inner_windows = determine_windows(fill2,20)
# Idem for start windows
start_windows = determine_windows(fill3,20)

decay_windows = []
release_a = []; release_b = []; pump = []; Pmed = []

for owin in outer_windows:
    
    ostr, oend, _ = owin
    
    for iwin in inner_windows:
        
        _, iend, _ = iwin
        
        if iend < oend and iend > ostr:
            
            decay_windows += [(ostr, iend, oend)]
            
            release_a += [(iend-ostr)*dt]
            release_b += [(oend-iend)*dt]
            Pmed += [ysig[iend]]

for swin in start_windows:
    _, _, count = swin
    
    pump += [count*dt]


pressures_high = ysig[np.logical_or(dysig>25,dysig>-25)]



xlabel = "Time (s)"
ylabel = "Pressure (cmH2O)"
dylabel = "Pressure rate (cmH2O/s)"
s = 20
fill_color ="tab:orange"


# %%
fig, axes = plt.subplots(nrows=2,figsize=(6,6),dpi=150)

axes[0].plot(xsig,ysig,alpha=0.5)
axes[0].fill_between(xsig,ysig,0.0,fill,alpha=0.5, color=fill_color)
axes[0].fill_between(xsig,ysig,0.0,fill2,alpha=0.5, color='b')
axes[0].fill_between(xsig,ysig,0.0,fill3,alpha=0.5, color='g')

axes[0].scatter(xsig[peaks],ysig[peaks],s=s,color="k")
axes[0].set_xticks([])
axes[0].set_ylabel(ylabel)


axes[1].plot(xsig,dysig,alpha=0.5)
axes[1].fill_between(xsig,dysig,0.0,fill,alpha=0.5, color=fill_color)
axes[1].fill_between(xsig,dysig,0.0,fill2,alpha=0.5, color='b')
axes[1].fill_between(xsig,dysig,0.0,fill3,alpha=0.5, color='g')
axes[1].scatter(xsig[peaks],dysig[peaks],s=s,color="k")
axes[1].set_ylabel(dylabel)
axes[1].set_xlabel(xlabel)

# Decay windows
for window in decay_windows:
    start,mid,end= window
    
    axes[0].scatter(xsig[start],ysig[start],s=s,color="r",marker="v")
    axes[0].scatter(xsig[end],ysig[end],s=s,color="r",marker="^")
    axes[0].scatter(xsig[mid],ysig[mid],s=s,color="r",marker="<")
    
    axes[1].scatter(xsig[start],dysig[start],s=s,color="r",marker="v")
    axes[1].scatter(xsig[end],dysig[end],s=s,color="r",marker="^")
    axes[1].scatter(xsig[mid],dysig[mid],s=s,color="r",marker="<")
    
# Initial slope
for window in start_windows:
    start,end,count = window
    end+=5
    axes[0].scatter(xsig[start],ysig[start],s=s,color="b",marker="v")
    axes[0].scatter(xsig[end],ysig[end],s=s,color="b",marker="^")
    
    axes[1].scatter(xsig[start],dysig[start],s=s,color="b",marker="v")
    axes[1].scatter(xsig[end],dysig[end],s=s,color="b",marker="^")
    
dts = []
for i in range(1,len(peaks)):
    dts += [(peaks[i]-peaks[i-1])*dt]
    
    
    
# %%

Tcycle = np.median(dts)
Tstart = np.median(pump)
Trela = np.median(release_a)
Trelb = np.median(release_b)
Tplat = Tcycle-(Tstart+Trela+Trelb)

print("Pressures:")
print(" > Phigh: %.2f"%np.median(pressures_high))
print(" > Pmed: %.2f"%np.median(Pmed))
print("Times: ")
print(" > Tcycle: %.2f"%Tcycle)
print(" > Tplat: %.2f"%Tplat)
print(" > Tstart: %.2f"%Tstart)
print(" > Trelease_a: %.2f"%Trela)
print(" > Trelease_b: %.2f"%Trelb)