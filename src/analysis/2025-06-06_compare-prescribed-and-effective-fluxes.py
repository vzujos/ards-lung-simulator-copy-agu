import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz

# --- Define paths ---
root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/%s/"
case = "Calibrated-PIG5-ARDSnet-10"
path = root % case + "/Signals/"

# --- Load signals ---
q_prescribed = np.load(path + "prescribedfluxes.npy")  # [uL/ms]
q_resulting = np.load(path + "fluxes.npy")             # [L/s]
time = np.load(path + 'effectivetimes.npy')            # [s]

# --- Convert prescribed flux to L/s ---
q_prescribed_lps = -q_prescribed * 1e-6  # [uL/ms] → [L/s] (sign flipped for consistency)

# --- Compute dt and integrate to get volume ---
dt = np.mean(np.diff(time))  # Assume uniform sampling

vol_prescribed = cumtrapz(q_prescribed_lps, dx=dt, initial=0.0)  # [L]
vol_resulting = cumtrapz(q_resulting, dx=dt, initial=0.0)        # [L]

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=150, sharex=True)

# Plot fluxes
axes[0].plot(time, q_prescribed_lps, label="Prescribed Flux", color='r', alpha=0.7)
axes[0].plot(time, q_resulting, label="Resulting Flux", color='b', alpha=0.7)
axes[0].axhline(0.0, color='gray', linestyle='--', alpha=0.3)
axes[0].set_ylabel("Flow (L/s)")
axes[0].legend()
axes[0].set_title("Flux Signals")

# Plot volumes
axes[1].plot(time, vol_prescribed, label="Integrated Prescribed Volume", color='r', linestyle='--')
axes[1].plot(time, vol_resulting, label="Integrated Resulting Volume", color='b')
axes[1].axhline(0.0, color='gray', linestyle='--', alpha=0.3)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Volume (L)")
axes[1].legend()
axes[1].set_title("Volume Derived by Integration")

plt.tight_layout()
plt.show()

# Optional: Save volume arrays
#np.save(path + "volume_prescribed.npy", vol_prescribed)
#np.save(path + "volume_resulting.npy", vol_resulting)
