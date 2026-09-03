# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:29:44 2024

@author: angus
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io.matlab import loadmat


# %%

def determine_frequencies(time, signal):
    # Ensure inputs are numpy arrays
    time = np.array(time)
    signal = np.array(signal)
    
    # Calculate the time step (assuming uniform sampling)
    dt = time[1] - time[0]  # Time interval between samples
    
    # Perform FFT on the signal
    fft_result = np.fft.fft(signal)
    
    # Get the number of samples
    n = len(signal)
    
    # Compute the frequency axis (corresponding to the FFT result)
    frequencies = np.fft.fftfreq(n, d=dt)
    
    # Only take the positive half of the frequencies and magnitudes
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    magnitudes = np.abs(fft_result)[positive_freq_idx] / n
    
    return frequencies, magnitudes

# %%


subjects = [2,3,4,5,6]
states = ["APRV","ARDSnet"]

targets = ["PIG%i-%s.mat"%(subject,state) for subject in subjects for state in states]

target = targets[9]
key = target.split(".")[0]
protocol = key.split("-")[1]

root = "../raw-data/Signals/"

matpath = root+target
matdata = loadmat(matpath)

time_dict = {"Tcycle":{  "PIG2-ARDSnet":2.8852,
                         "PIG3-ARDSnet":2.0004,
                         "PIG4-ARDSnet":2.5005,
                         "PIG5-ARDSnet":2.0010,
                         "PIG6-ARDSnet":2.1743,
                         "PIG2-APRV":1.8754,
                         "PIG3-APRV":1.9235,
                         "PIG4-APRV":1.9235,
                         "PIG5-APRV":1.9235,
                         "PIG6-APRV":2.3342,
                      },
             }

    
# Read the signals

time = matdata['time'].flatten()
Peso = matdata['Peso_sdata'].flatten()
Paw = matdata['Paw_rdata'].flatten()
flow = matdata['flow_rdata'].flatten()
volume = matdata['volume'].flatten()

# Determine frequencies using FFT
frequencies, magnitudes = determine_frequencies(time,flow)

# Filter for the 5 maximum magnitudes
threshold = np.quantile(magnitudes,0.998)
mag_mask = magnitudes>threshold
sel_freqs = frequencies[mag_mask]
sel_mags = magnitudes[mag_mask]
mag_sorter = np.flip(np.argsort(sel_mags))
freqs = sel_freqs[mag_sorter][:5]

# Determine periods
periods = 1/freqs

# %%

plt.plot(time,flow)
for i in range(20):
    t = i*time_dict["Tcycle"][key]
    if t<max(time):
        plt.axvline(t, color="r",alpha=0.5,ls="--")


# %% Scouter

if False:
    plt.plot(time,flow)
    for e, period in enumerate(periods):
        for i in range(200):
            t =  period*i
            if t<max(time) and i%2==0:
                if e == 1:
                    plt.axvline(t, color="r",alpha=0.5,ls="--")
                

        
