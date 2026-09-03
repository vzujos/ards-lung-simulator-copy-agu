
import os
import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# %%

def determine_misfits_from_signal(simulation_path, signal_path, wave_config,
                                  nsamples=250, 
                                  include_volume_signal = False, w_volume_signal=1.0,
                                  include_pressure_signal = False, w_pressure_signal=1.0,
                                  include_flux_signal = False, w_flux_signal=1.0,
                                  include_ppeak = False, w_ppeak=1.0,
                                  include_pplat = False, w_pplat=1.0,
                                  include_peep = False, w_peep=1.0,
                                  include_vpeak = False, w_vpeak=1.0,
                                  normalize_misfit=True,): 
                                                                    
        
    # Simulation parameters (mostly for normalization)
    Tsyr = wave_config["Tsyr"]
    Tpausa = wave_config["Tpausa"]
    Texp = wave_config["Texp"]
    ppeak = wave_config["Ppeak"]
    pplat = wave_config["Pplat"]
    peep = wave_config["PEEP"]
    
    # Load the experimental signal
    mat = np.load(signal_path)
    Paw = mat['pressure']; flow = mat['flow']; 
    vol = mat['volume']; time = mat['time']
    time -= time[0]
    
    # Determine the number of cycles
    Tcycle = Tsyr+Tpausa+Texp
   
    #  Load a sample calibration
    #simulation_path += "/Signals/"
    stime = np.load(simulation_path+"effectivetimes.npy").flatten()
    sPaw = np.load(simulation_path+"presionestodas.npy").flatten()*10.1972
    sflow = np.load(simulation_path+"fluxes.npy").flatten()
    svol = np.load(simulation_path+"volumenes.npy").flatten()
    svol = svol - svol[0]
    
    maximum_simulated_time = np.max(stime)
    if maximum_simulated_time < (Tsyr+Tpausa):
        print("The simulation finished before inspiratory pause")
        print("Returning artificially high cost")
        return 999.9
        
    # Generate piecewise linear functions
    fsPaw  = interp1d(stime, sPaw,  kind='linear', fill_value="extrapolate")
    fsflow = interp1d(stime, sflow, kind='linear', fill_value="extrapolate")
    fsvol  = interp1d(stime, svol,  kind='linear', fill_value="extrapolate")
    
    fPaw  = interp1d(time, Paw,  kind='linear')
    fflow = interp1d(time, flow, kind='linear')
    fvol  = interp1d(time, vol,  kind='linear')
    
    # Sample  measurements
    simulated_pplat = fsPaw(Tsyr+Tpausa*(0.999)) # Plateau pressure
    simulated_ppeak = fsPaw(Tsyr*0.999) # Peak pressure
    simulated_peep  = sPaw[0] # PEEP value
    simulated_vpeak = fsvol(Tsyr+Tpausa*(0.95)) # Maximum volume
    
    # Signal VT
    signal_vt = fvol(Tsyr+Tpausa*(0.95))

    err = 1e-8
    ttime = np.linspace(0+err,Tcycle-err,nsamples)
    
    # Determine misfit
    pmisfit = 0.0; fmisfit = 0.0; vmisfit = 0.0; misfit = 0.0
    
    # Counters
    npm = 0; nfm = 0; nvm = 0
    
    misfit_info = {}
    
    # Determine signal misfit
    for t in ttime:
        T = np.floor(t/Tcycle).astype(int)
        
        # Skip these measurements if beyond the simulated time
        if t>maximum_simulated_time:
            continue
            
        if (t>T*Tcycle) and (t<(T*Tcycle+Tsyr+Tpausa)):
            # Inspiratory phases
            pmisfit += (fPaw(t)-fsPaw(t))**2
            npm += 1
        elif (t>(T*Tcycle+Tsyr+Tpausa)) and (t<(T+1)*Tcycle):
            # Expiratory phase
            fmisfit += (fflow(t)-fsflow(t))**2
            nfm += 1
        # Thoughout the whole cycle
        vmisfit += (fvol(t)-fsvol(t))**2
        nvm +=1
    
    # Normalize (Suggested: True)
    if normalize_misfit:
        pmisfit /= (npm*pplat**2)
        fmisfit /= (nfm*(signal_vt/Tsyr)**2)
        vmisfit /= (nvm*signal_vt**2)
    
    if include_pressure_signal: misfit += pmisfit*w_pressure_signal
    if include_flux_signal: misfit += fmisfit*w_flux_signal
    if include_volume_signal: misfit += vmisfit*w_volume_signal
    if include_ppeak:
        misfit_ppeak = (simulated_ppeak-ppeak)**2/ppeak**2
        misfit += misfit_ppeak*w_ppeak
        
    if include_pplat:
        misfit_pplat = (simulated_pplat-pplat)**2/pplat**2
        misfit += misfit_pplat*w_pplat
        misfit_info.update({"Pplat":misfit_pplat*w_pplat})
        
        #print(" PPLATEAU MISFIT:")
        #print(" > Simulated Pplat: %.3f"%simulated_pplat)
        #print(" > Signal Pplat: %.3f"%pplat)
        #print(" > Misfit Pplat: %.3f"%misfit_pplat)
    
        
    if include_peep:
        misfit_peep = (simulated_peep-peep)**2/peep**2
        
        misfit += misfit_peep*w_peep
        #print(" PEEP MISFIT:")
        #print(" > Simulated PEEP: %.3f"%simulated_peep)
        #print(" > Signal PEEP: %.3f"%peep)
        #print(" > Misfit PEEP: %.3f"%misfit_peep)
        misfit_info.update({"PEEP":misfit_peep*w_peep})

    if include_vpeak:

        misfit_vpeak = ((simulated_vpeak-signal_vt)/signal_vt)**2
        #print(" VOLUME MISFIT: ")
        print(" > Simulated Vpeak: %.3f"%simulated_vpeak)
        print(" > Signal VT: %.3f"%signal_vt)
        #print(" > Misfit Vpeak: %.3f"%misfit_vpeak)
        misfit_info.update({"Vpeak": misfit_vpeak*w_vpeak})

        misfit += misfit_vpeak*w_vpeak                       
        
    outpath = os.path.join(simulation_path, "./misfit_report.txt")
    with open(outpath, 'w') as f:
        f.write("\n")
        if include_pressure_signal: f.write("%18s %.2f (%4.1f%%)\n" % ("Pressure misfit:", pmisfit, pmisfit/misfit*100))
        if include_flux_signal:     f.write("%18s %.2f (%4.1f%%)\n" % ("Flow misfit:", fmisfit, fmisfit/misfit*100))
        if include_volume_signal:   f.write("%18s %.2f (%4.1f%%)\n" % ("Volume misfit:", vmisfit, vmisfit/misfit*100))
        if include_pplat:           f.write("%18s %.2f (%4.1f%%)\n" % ("Pplat misfit:", misfit_pplat, misfit_pplat/misfit*100))
        if include_ppeak:           f.write("%18s %.2f (%4.1f%%)\n" % ("Ppeak misfit:", misfit_ppeak, misfit_ppeak/misfit*100))
        if include_peep:            f.write("%18s %.2f (%4.1f%%)\n" % ("PEEP misfit:", misfit_peep, misfit_peep/misfit*100))
        if include_vpeak:           f.write("%18s %.2f (%4.1f%%)\n" % ("Vpeak misfit:", misfit_vpeak, misfit_vpeak/misfit*100))
        f.write("-"*30 + "\n")
        f.write("%18s %.1f\n" % ("Overall misfit:", misfit))
    
    return misfit, misfit_info

# %%

def add_Signals(simpath, axes, color, tag):
    
    # Load data
    time = np.load(simpath+'effectivetimes.npy')
    pres = np.load(simpath+'presionestodas.npy')*10.1972
    flux = np.load(simpath+'fluxes.npy')
    vols = np.load(simpath+'volumenes.npy')
    vols -= vols[0]
    
    # Generate curves
    axes[0].plot(time[:-1],pres[:-1], color=color,ls="-",lw=1.5,label=tag,alpha=0.8)
    axes[1].plot(time[:-1],flux[:-1], color=color,ls="-",lw=1.5,label=tag,alpha=0.8)
    axes[2].plot(time[:-1],vols[:-1], color=color,ls="-",lw=1.5,label=tag,alpha=0.8)
    print("Maximum time at %s: %.2f"%(tag,time.max()))
    
    # Fill between curves for easier definition
    axes[0].fill_between(time[:-1].flatten(),pres[:-1].flatten(),np.zeros_like(pres[:-1]).flatten(),alpha=0.20, color=color)
    axes[1].fill_between(time[:-1].flatten(),flux[:-1].flatten(),np.zeros_like(flux[:-1]).flatten(),alpha=0.20, color=color)
    axes[2].fill_between(time[:-1].flatten(),vols[:-1].flatten(),np.zeros_like(vols[:-1]).flatten(),alpha=0.20, color=color)
    

parameters = {"PIG2-ARDSnet":{"Tsyr":0.435,
                           "Tpausa":0.560,
                           "Texp":1.870,
                           "Pplat":21.7,
                           "PEEP":11.0,
                           "Ppeak":26.8,
                           "Signal path":'model2/signals/PIG2-ARDSnet.npz',
                           "Target volume":0.3527,
                           "EELV":1936.33,
                           },
           "PIG3-ARDSnet":{"Tsyr":0.455,
                           "Tpausa":0.435,
                           "Texp":1.100,
                           "Pplat":26.3,
                           "PEEP":11.3 ,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG3-ARDSnet.npz',
                           "Target volume":0.3870,
                           "EELV":1745.23 ,
                           },
           "PIG4-ARDSnet":{"Tsyr":0.48,
                           "Tpausa":0.28,
                           "Texp":1.73,
                           "Pplat":18.5,
                           "PEEP": 10.7,
                           "Ppeak":24.8,
                           "Signal path":'model2/signals/PIG4-ARDSnet.npz',
                           "Target volume":0.411,
                           "EELV":2407.44,
                           },
           "PIG5-ARDSnet":{"Tsyr":0.375,
                           "Tpausa":0.375,
                           "Texp":1.25,
                           "Pplat":20.6,
                           "PEEP": 10.8,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG5-ARDSnet.npz',
                           "Target volume":0.4011,
                           "EELV":2072.99,
                           },
           "PIG6-ARDSnet":{"Tsyr":0.540,
                           "Tpausa":0.265,
                           "Texp":1.340,
                           "Pplat":22.7,
                           "PEEP": 10.7,
                           "Ppeak":25.7,
                           "Signal path":'model2/signals/PIG6-ARDSnet.npz',
                           "Target volume":0.2995,
                           "EELV":2314.06,
                           }
           }

pig_num = 4

codename = "PIG%i-ARDSnet"%pig_num

# Unpack information
Tsyr = parameters[codename]["Tsyr"]
Tpausa = parameters[codename]["Tpausa"]
Texp = parameters[codename]["Texp"]
Tinsp = Tsyr+Tpausa
Tcycle = Tinsp+Texp



textsize = 12

path = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/PIG%i/"%pig_num

colors = ['tab:blue','tab:orange','tab:green','tab:purple', "silver"]

# Load the signal data
signalpath = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/PIG%i-ARDSnet.npz'%(pig_num,pig_num)
signals = np.load(signalpath)
svol = signals['volume']
spres = signals['pressure']
sflow = -signals['flow']
stime = signals['time']

# Generate the canvas
fig,axes = plt.subplots(nrows=3, dpi=200, figsize=(5,6))

# Add signal data
axes[0].plot(stime,spres, color="k",ls="--",lw=1.5,label="Experiment",alpha=.9)
axes[1].plot(stime,sflow, color="k",ls="--",lw=1.5,label="Signal",alpha=.9)
axes[2].plot(stime,svol, color="k",ls="--",lw=1.5,label="Signal",alpha=.9)

#for ax,ydata in zip(axes,[spres,sflow,svol]):
#     ax.fill_between(stime,ydata,np.zeros_like(ydata),alpha=0.3)
    
root = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/'

simdata = {5:[#("Simulation",root+"PIG5/sim_058/Signals/"),
              ("Simulation",root+"MFSIMS/PIG5/output/Signals/"),

              #("mc-bir-best",root+"PIG5/sim_054/Signals/"),
          #    ("mc-ma-prev" ,root+"PIG5/sim_057/Signals/"),
              ("mc-ma-best" ,root+"PIG5/sim_011/Signals/"),

            #  ("mc-bir-wide" ,root+"PIG5/sim_096/Signals/"),
             # ("mf-sim",root+"MFSIMS/PIG5/output/Signals/")
              ],
           3:[#("mf-bir-best",None),
              ("Simulation",root+"PIG3/sim_017/Signals/"),
              #("mc-ma-best" ,root+"PIG3/sim_106/Signals/")
              ],
           4:[#("mf-bir-best",None),
              #("mc-bir-best",root+"PIG4/sim_101/Signals/"),
              ("Birzle-BEST",root+"MFSIMS/PIG4-redo/output/Signals/"),
              ("Ma-BEST" ,root+"PIG4/sim_047/Signals/"), # OK
              ("Ma-ALT1" ,root+"PIG4/sim_041/Signals/"), # OK
              ("Ma-ALT2" ,root+"PIG4/sim_027/Signals/"), # OK
              ],
           6:[("mf-bir-best",None),
              ("mc-bir-best",root+"PIG6/sim_103/Signals/"),
              ("mc-bir-wide",root+"PIG6/sim_086/Signals/"),
              ("mc-ma-best" ,None)],
           2:[#("mf-bir-best",None),
              ("m-bir-good",root+"PIG2/sim_000/Signals/"),
              ("m-bir-wide",root+"PIG2/sim_006/Signals/"),
              ("m-bir-best",root+"PIG2/sim_026/Signals/"),

              ("mc-ma-best" ,None)]
                         }


   
simpaths = [("signal",root+'/MFSIMS/PIG%i/output/Signals/'%pig_num)]+simdata[pig_num]


vertical_color = "darkred"

if True:
     
    if False:
        for ax in axes:
            # Inspiratory time
            ax.axvline(Tinsp,color=vertical_color)
            # Tsyringe
            ax.axvline(Tsyr,color=vertical_color)
            # Mark the 0.0 baseline
            ax.axhline(0.0, ls="-",color="darkred",lw=0.5)

    # Pick the color
    color = colors[0]
    tag = "Simulation"
    
    #add_Signals(simpaths[0][1],axes,colors[1],tag="Simulation")

    if len(simpaths)>1 and True:
        for simpath,color in zip(simpaths[1:],colors[1:]):
            if simpath[1] is None: continue
            add_Signals(simpath[1],axes,color,tag=simpath[0])

    # Add names
    axes[0].set_ylabel("Pressure\n(cmH2O)",size=textsize)
    axes[1].set_ylabel("Flux\n(L/s)",size=textsize)
    axes[2].set_ylabel("Volume\n(L)",size=textsize)
    axes[2].set_xlabel("Time (s)", size=textsize)    
    
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    
    
axes[0].legend()

if pig_num == 5:
    xticks = [0.0,0.5,1.0,1.5,2.0]
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(["%.1f"%x for x in xticks ])
    
    axes[0].set_yticks([0,20,40])
    axes[1].set_yticks([-1.0, 0.0, 1.0])
    axes[2].set_yticks([0.0,0.2,0.4])
    
if pig_num == 3:
    xticks = [0.0,0.5,1.0,1.5,2.0]
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(["%.1f"%x for x in xticks ])
    
    axes[0].set_yticks([0,20,40])
    axes[1].set_yticks([-1.0, 0.0, 1.0])
    axes[2].set_yticks([0.0,0.2,0.4])

if pig_num == 4:
    xticks = [0.0,0.5,1.0,1.5,2.0,2.5]
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(["%.1f"%x for x in xticks ])
    
    axes[0].set_yticks([0,15,30])
    axes[1].set_yticks([-1.0, 0.0, 1.0])
    axes[2].set_yticks([0.0,0.2,0.4])


plt.tight_layout()

# %%
for simpath in simpaths:
 #   print("Codename: %s"%simpath[0])
    if simpath[0] == 'signal': 
        continue
    value, info = determine_misfits_from_signal(simpath[1],signalpath,parameters[codename], 
                                  include_vpeak=True,
                                  include_pplat=True, 
                                  include_peep=True,
                                  w_vpeak=20.0,
                                  normalize_misfit=True)
    
    print("Code: %11s || Overall misfit: %.3f"%(simpath[0],value))
    print(" >>> Composition:")
    for key in info:
        print("   > %s: %.3f (%.1f"%(key,info[key],info[key]/value*100)+"%)")
    print()