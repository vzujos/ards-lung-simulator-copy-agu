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

# %%
    
root_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/"

subject = "PIG6"
case = "ARDSnet"
dt = 0.0025
path_to_signal = root_path+"%s/%s-%s.mat"%(subject,subject,case)

if os.path.isfile(path_to_signal):
    print("File found")
else:
    print("File not found")
    

# %%

# Load the Matlab signal
sig = loadmat(path_to_signal)

# Load each field
time = sig['time'].flatten()
Paw = sig['Paw_rdata'].flatten()
Peso = sig['Peso_sdata'].flatten()
flow = sig['flow_rdata'].flatten()
vol = sig['volume'].flatten()

print("Maximum time registered: %.2f"%time.max())

# %%

lower_threshold_for_volume_peak = 0.16
minimum_cycle_duration = 1.0
time_cutoff = 10.0


peaks, properties = find_peaks(vol,
                               height=lower_threshold_for_volume_peak,
                               distance=minimum_cycle_duration)

dt=0.0025
plt.plot(time,vol,alpha=0.3)
plt.scatter(time[peaks],vol[peaks],color="r",s=8)
plt.show()
print("Median volume peak: %.3f"%np.median(vol[peaks]))

median_volume_peak = np.median(vol[peaks])


# %% Determine Tsyr

changes_of_sign = determine_changes_of_sign(time,vol,time_cutoff,)

if changes_of_sign[0]>100 and flow[0]>0:
    print("Note: Adding an artificial change of sign at the first wave")
    changes_of_sign.insert(0,0)

m = time<time_cutoff
dysig = np.gradient(vol,0.0025)
xsig=time


vol_thr = 0.90 * median_volume_peak
Tsyrs = []
Tsyrs_points = []

for i in range(1, len(changes_of_sign)):
    
    start = changes_of_sign[i-1]
    end = changes_of_sign[i]
    
    area = np.trapz(y=dysig[m][start:end],x=xsig[m][start:end])
    #print(area/median_volume_peak)
    if area > vol_thr:
        Tsyrs += [(end-start)*dt]
        Tsyrs_points += [(start, end)]
    
Tsyr = np.median(Tsyrs)
print("Tsyr = %.4f"%Tsyr)


# %% Determine tentative expiratory points

changes_of_sign = determine_changes_of_sign(time,vol,time_cutoff,positive=False,
                                            negative_cutoff=-0.01,fill_color="tab:orange")



m = time<time_cutoff
dysig = np.gradient(vol,0.0025)
xsig=time


vol_thr = 0.90 * median_volume_peak
Texps_points_temp = []

for i in range(1, len(changes_of_sign)):
    
    # Take the bounds of the region
    start = changes_of_sign[i-1]
    end = changes_of_sign[i]
    # Accept them if the area between the curve
    area = -(np.trapz(y=dysig[m][start:end],x=xsig[m][start:end]))
    #print(area/median_volume_peak)
    if area > vol_thr:
        print(area)
        Texps_points_temp += [(start, end)]

# %% 

Texps = []
Texps_points = []
Twaves_points = []
Twaves = []

for i in range(min(len(Tsyrs_points),len(Texps_points_temp))):
    # i is each wave number
    # The start of the second wave is the end of the first
    try:
        expiration_end = Tsyrs_points[i+1][0]-1
    except:
        print("Not enough data to continue. Wave #%i"%(i+1))
        break
    
    wave_start = Tsyrs_points[i][0]
    wave_end = expiration_end
    
    expiration_start = Texps_points_temp[i][0]

    Texps_points += [(expiration_start,expiration_end)]
    Texp = (expiration_end-expiration_start)*dt
    Texps += [Texp]
    
    Twave = (wave_end-wave_start)*dt
    Twaves += [Twave]
    Twaves_points += [(wave_start,wave_end)]
    
Texp = np.median(Texps)
print("Texp = %.4f"%Texp)
Twave = np.median(Twaves)
print("Twave = %.4f"%Twave)

# %%

m_exp = np.zeros_like(time,dtype=bool)[m]
for (start, end) in Texps_points:
    m_int = np.zeros_like(time,dtype=bool)[m]
    m_int[start:end]=True
    m_exp = np.logical_or(m_exp, m_int)
    
m_syr = np.zeros_like(time,dtype=bool)[m]
for (start, end) in Tsyrs_points:
    m_int = np.zeros_like(time,dtype=bool)[m]
    m_int[start:end]=True
    m_syr = np.logical_or(m_syr, m_int)
    
    
fig,axes = plt.subplots(nrows=2,figsize=(8,6))
        
xsig = time
ysig = vol
xlabel="Time (s)"
ylabel="Volume (L)"
dylabel="Flow (L/s)"
s=36
place_Texp_points = True

axes[0].plot(xsig[m],ysig[m],alpha=0.5)
fill_color="g"
axes[0].fill_between(xsig[m],ysig[m],0.0,m_syr,alpha=0.5, color=fill_color)
fill_color="tab:orange"
axes[0].fill_between(xsig[m],ysig[m],0.0,m_exp,alpha=0.5, color=fill_color)


axes[0].set_xticks([])
axes[0].set_ylabel(ylabel)
        
axes[1].plot(xsig[m],dysig[m],alpha=0.5)
fill_color="g"
axes[1].fill_between(xsig[m],dysig[m],0.0,m_syr,alpha=0.5, color=fill_color)

fill_color="tab:orange"
axes[1].fill_between(xsig[m],dysig[m],0.0,m_exp,alpha=0.5, color=fill_color)

axes[1].set_ylabel(dylabel)
axes[1].set_xlabel(xlabel)

if place_Texp_points:
    
    t = 1
    
    for (wave_start,wave_end) in Twaves_points:
        
        xlims = axes[0].get_xlim(); Dx = xlims[1]-xlims[0]; dx = Dx/100; 
        
        ylims = axes[0].get_ylim(); Dy = ylims[1]-ylims[0]; dy = Dy/20
        
        axes[0].scatter(xsig[m][wave_start],ysig[m][wave_start],s=s,color="b", marker="v")
        axes[0].text(xsig[m][wave_start]+dx,ysig[m][wave_start]-dy,"%i"%t,va='bottom')
        axes[0].scatter(xsig[m][wave_end],ysig[m][wave_end],s=s,color="b",marker="^")

        ylims = axes[1].get_ylim(); Dy = ylims[1]-ylims[0]; dy = Dy/20

        axes[1].scatter(xsig[m][wave_start],dysig[m][wave_start],s=s,color="b", marker="v")
        axes[1].text(xsig[m][wave_start]+dx,dysig[m][wave_start]-dy,"%i"%t,va='bottom')
        axes[1].scatter(xsig[m][wave_end],dysig[m][wave_end],s=s,color="b",marker="^")        

        t+=1

# %%

Tinsp = Twave-Texp
Tpause = Tinsp-Tsyr

print("Wave analysis: %s "%subject)
print(" > Tsyr: %.2f"%Tsyr)
print(" > Tpause: %.2f"%Tpause)
print(" > Tinsp: %.2f"%Tinsp)
print(" > Texp: %.2f"%Texp)
print(" > Twave: %.2f"%Twave)