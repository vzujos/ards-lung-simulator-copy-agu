
import os
import numpy as np
import meshio
import matplotlib.pyplot as plt

def add_Signals(simpath, axes, color, tag):
    
    # Load data
    time = np.load(simpath+'effectivetimes.npy')
    pres = np.load(simpath+'presionestodas.npy')*10.1972
    flux = np.load(simpath+'fluxes.npy')
    flux[0] = flux[1]
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

pig_num = 3

codename = "PIG%i-ARDSnet"%pig_num

# Unpack information
Tsyr = parameters[codename]["Tsyr"]
Tpausa = parameters[codename]["Tpausa"]
Texp = parameters[codename]["Texp"]
Tinsp = Tsyr+Tpausa
Tcycle = Tinsp+Texp

textsize = 12

path = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/Codes/DeleteMe/PIG%i/"%pig_num

colors = ['tab:blue','tab:orange','tab:green','tab:purple']

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
root2 = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/'

simdata = {5:[#("Simulation",root+"PIG5/sim_058/Signals/"),
             # ("mc-calibration",root+"PIG5/sim_285/Signals/"),
              ("Birzle's CM" ,root2+"PIG5-mf-bir/Signals/"),
              ("Novel CM" ,root2+"PIG5-mf-per/Signals/"),
            #  ("mc-bir-wide" ,root+"PIG5/sim_096/Signals/"),
             # ("mf-sim",root+"MFSIMS/PIG5/output/Signals/")
              ],
           3:[#("mf-bir-best",None),
              ("Simulation",root+"PIG3/sim_017/Signals/"),
              #("mc-ma-best" ,root+"PIG3/sim_106/Signals/")
              ],
           4:[#("mf-bir-best",None),
              #("mc-bir-best",root+"PIG4/sim_101/Signals/"),
              ("mc-calibration",root+"PIG4/sim_285/Signals/"),
              ("mc-ma-best" ,root+"PIG4/sim_018/Signals/"),
              ],
           6:[("mc-calibration",root+"PIG6/sim_125/Signals/"),
              ("mf-simulation" ,root2+"PIG6-mf-bir/Signals/"),],
           2:[#("mf-bir-best",None),
              ("mc-calibration",root+"PIG2/sim_040/Signals/"),
              ("mf-simulation" ,root2+"PIG2-mf-bir/Signals/"),]
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
