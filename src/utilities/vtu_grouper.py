# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:58:16 2023

@author: angus
"""

import meshio as io
import os
import numpy as np
import re, json


pig_num = 5

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
root = os.path.join(repo_root, "outputs", "PIG5-mc-per")
path = os.path.join(root, "VTK")
filelist = os.listdir(path)

dest = os.path.join(root, "post")
os.makedirs(dest, exist_ok=True)

use_initial_porosity = True
alpha_correction = True


initial_porosity_path = os.path.join(
    repo_root, "Geometries", "PIG%i" % pig_num, "ARDSnet", "medium",
    "FEniCS", "Porosity_Visualization.vtu"
)
if use_initial_porosity and os.path.isfile(initial_porosity_path):
    phi0_mesh = io.read(initial_porosity_path)
    phi0_celldata = phi0_mesh.cell_data['Porosity_EE'][0]
    phi0_pointdata = phi0_mesh.point_data['Porosity_EE']
else:
    use_initial_porosity = False
    phi0_pointdata = None


vtu_files = [filename for filename in filelist
             if filename.lower().endswith(".vtu")]

heads = ["Displacement","HYD","HYD_tissue",
         "Jacobian","Pressure","VM","VM_tissue", "QQint"]

sorter = {}

dummy = list(sorted(filter(lambda x: x[0]=="D", vtu_files)))
dummy = list(sorted(map(lambda x: x.split("_")[1],dummy)))
dummy = list(sorted(map(lambda x: x.split(".vtu")[0],dummy)))


longs = list(filter(lambda x: len(x.split("_"))==3,vtu_files))
shorts = list(filter(lambda x: len(x.split("_"))!=3,vtu_files))

files = {"Displacement":list(sorted(filter(lambda x: x[0]=="D", vtu_files))),
         "Pressure":list(sorted(filter(lambda x: x[0]=="P", vtu_files))),
         "Jacobian":list(sorted(filter(lambda x: x[0]=="J", vtu_files))),
         "HYD_tissue":list(sorted(filter(lambda x: x[:3]=="HYD", longs))),
         "HYD":list(sorted(filter(lambda x: x[:3]=="HYD", shorts))),
         "VM_tissue":list(sorted(filter(lambda x: x[:2]=="VM", longs))),
         "QQint":list(sorted(filter(lambda x: x[:2]=="QQ", shorts))),
         "VM":list(sorted(filter(lambda x: x[:2]=="VM", shorts))),
         "time":dummy,
         }

heads = [field for field in heads if files[field]]


alpha_data = {2:1.0347,
                 3:1.0325,
                 4:1.0458,
                 # 5:1.0357, # Birzle
                 #5:1.0311, # Ma
                 6:1.0376}

if alpha_correction:
    alpha = alpha_data.get(pig_num, 1.0)
    if pig_num in alpha_data:
        print("Note that we are employing alpha-correction for the Jacobian")
        print("*** Current value for subject 'PIG%i' is %.4f ***"%(pig_num,alpha))
        print("*** MAKE SURE IT IS THE RIGHT VALUE ***")
    else:
        print("No alpha-correction value is available; using alpha=1.0")


##end_tag = "3.250000000000"
end_tag = dummy[-1]

outlist = []  
for i in range(len(dummy)):
    
    # first field and mesh
    field = heads[0]
    repath = path+files[field][i]
    msh = io.read(repath)
    # other fields
    for field in heads[1:]:
        
        # path to secondary mesh
        repath = path+files[field][i]
        # load
        msh2 = io.read(repath)
        # extract data
        key = list(msh2.point_data.keys())[0]
        msh.point_data.update({field:msh2.point_data[key]})
        
        
        if field == "Pressure":
            pres = msh.point_data["Pressure"]
            msh.point_data.update({"Pressure (cmH2O)":pres*10.1972})
        
        if field == "Jacobian":
            jac = msh.point_data["Jacobian"]
            jac_full = jac.copy()
            jac_partial = jac.copy()/alpha**3
            data = jac_partial - 1 + 0.5
            
            if use_initial_porosity and phi0_pointdata is not None and jac.shape != phi0_pointdata.shape:
                use_initial_porosity = False
                print("Warning: Shape mismatch while reading the initial porosity")
            
            if use_initial_porosity and phi0_pointdata is not None and jac.shape == phi0_pointdata.shape:
                print("Using the initial porosity from provided field.")
                data = jac_partial-1+phi0_pointdata
            
            msh.point_data.update({"Jacobian Partial":jac_partial})
            msh.point_data.update({"Jacobian Full":jac_full})        
            msh.point_data.update({"Lagrangian Porosity":data})
            msh.point_data.update({"Eulerian Porosity":data/jac_partial})
            msh.point_data.update({"Delta Porosity":(data/jac_partial-phi0_pointdata)})
        
    # save
    tag = files["time"][i]
    
    finalname = "%sfull_%s.vtu"%(dest,tag)
    msh.write(finalname)
    outlist += [finalname]
    if i%10==0: print("i: %4i  |   t = %s"%(i,tag))
    
    if tag == end_tag:
        print("Early break due to 'end_tag'='%s'"%end_tag)
        break
# %%
heads = ["Displacement","HYD","HYD_tissue",
         "Jacobian","Pressure","VM","VM_tissue",
         "Eulerian Porosity", "Lagrangian Porosity", "QQint",
         "Jacobian Partial", "Jacobian Full"]

# Generate t = 0.000000000000
tag = "0.000000000000"
msh = io.read(finalname)
for field in heads:
    if not field in ["Jacobian", "Displacement", "Eulerian Porosity","Lagrangian Porosity","QQint"]:
        msh.point_data.update({field:np.zeros_like(msh.point_data[field])})
    elif field == "Jacobian":
        msh.point_data.update({'Jacobian':np.ones_like(msh.point_data[field])})
        msh.point_data.update({'Jacobian Partial':np.ones_like(msh.point_data[field])})
        msh.point_data.update({'Jacobian Complete':np.full_like(msh.point_data[field],alpha**3)})

    elif field == "Displacement":
        msh.point_data.update({"u":np.zeros_like(msh.point_data["u"])})
    elif field == "Eulerian Porosity":
        msh.point_data.update({field:phi0_pointdata})
    elif field == "Lagrangian Porosity":
        msh.point_data.update({field:phi0_pointdata})
    elif field == "QQint":
        msh.point_data.update({"qq":np.zeros_like(msh.point_data["u"])})

    
    finalname = "%sfull_%s.vtu"%(dest,tag)
    msh.write(finalname) 


# create json for time series

reout = [os.path.basename(filename) for filename in outlist]

series_data = {
    "file-series-version": "1.0",
    "files": []
}

# t=0 frame
series_data["files"].append({"name": "full_0.000000000000.vtu", "time": 0})

# rest of frames
for e, out in enumerate(reout):
    series_data["files"].append({"name": out, "time": e+1})

with open(dest + "simulation.vtu.series", "w") as f:
    json.dump(series_data, f, indent=2)

# --- Post-processing: generate physical-time series file ---


original_series = dest + "simulation.vtu.series"
fixed_series    = dest + "simulation_physical.vtu.series"

# Regex to extract the number from filenames like full_0.019700000000.vtu
number_re = re.compile(r"full_([0-9\.]+)\.vtu")

# Load the existing series JSON
with open(original_series, "r") as f:
    data = json.load(f)

# Update "time" entries with the float parsed from filenames
for entry in data["files"]:
    fname = entry["name"]
    m = number_re.search(fname)
    if m:
        entry["time"] = float(m.group(1))
    else:
        print(f"⚠️ Warning: could not extract time from {fname}")

# Write the fixed series file
with open(fixed_series, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Physical-time series file written to {fixed_series}")
