# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:58:16 2023

@author: angus
"""

import argparse
import json
import re
from pathlib import Path

import meshio as io
import numpy as np

parser = argparse.ArgumentParser(description="Group simulation VTU fields into time-series files.")
parser.add_argument("--case", default="PIG5-mc-per", help="Simulation directory below outputs/.")
parser.add_argument("--mesh-quality", default="medium-coarse", choices=("coarse", "medium-coarse", "medium", "medium-fine", "fine"))
parser.add_argument("--data-root", type=Path, default=None, help="Directory containing outputs/ and testing-data/.")
args = parser.parse_args()

project_root = Path(__file__).resolve().parents[2]
data_root = args.data_root.resolve() if args.data_root else project_root
case_root = data_root / "outputs" / args.case
mesh_quality = args.mesh_quality

# VTK path
vtk_path = case_root / "VTK"

# Output for postprocessed paths
dest = case_root / "post"
dest.mkdir(parents=True, exist_ok=True)

# Path to the reference (pre-stressed, end-expiratory) porosity
initial_porosity_path = (data_root / "testing-data" / "PIG5" / "ARDSnet" /
                         mesh_quality / "FEniCS" / "Porosity_Visualization.vtu")

if not vtk_path.is_dir():
    raise FileNotFoundError("VTK directory not found: %s" % vtk_path)
if not initial_porosity_path.is_file():
    raise FileNotFoundError("Initial porosity file not found: %s" % initial_porosity_path)

# %% Retrieve inverse jacobian for the unloaded geometry

# Associated to the inverse analysis
j0_mesh = io.read(case_root / 'InverseAnalysis' / 'Jacobian000000.vtu')
j_inv = j0_mesh.point_data['Jacobian']

# Associated to the deformation previous to the forward simulation
j_prev = j_inv

# %% Load initial porosity 

# Load the mesh with the porosity at end-expiration 
phi_ee_mesh = io.read(initial_porosity_path)
phi_ee_celldata = phi_ee_mesh.cell_data['Porosity_EE'][0]
phi_ee_pointdata = phi_ee_mesh.point_data['Porosity_EE']
# This would be a material porosity according to AR&H(2022)

# %% Process the data associated to the different timesteps

vtu_files = [path.name for path in vtk_path.iterdir()
             if path.is_file() and path.suffix == ".vtu"]

heads = ["Displacement","HYD","Jacobian","Pressure","VM", "QQint"]

sorter = {}

dummy = list(sorted(filter(lambda x: x[0]=="D", vtu_files)))
dummy = list(sorted(map(lambda x: x.split("_")[1],dummy)))
dummy = list(sorted(map(lambda x: x.split(".vtu")[0],dummy)))


longs = list(filter(lambda x: len(x.split("_"))==3,vtu_files))
shorts = list(filter(lambda x: len(x.split("_"))!=3,vtu_files))

files = {"Displacement":list(sorted(filter(lambda x: x[0]=="D", vtu_files))),
         "Pressure":list(sorted(filter(lambda x: x[0]=="P", vtu_files))),
         "Jacobian":list(sorted(filter(lambda x: x[0]=="J", vtu_files))),
         "HYD":list(sorted(filter(lambda x: x[:3]=="HYD", shorts))),
         "QQint":list(sorted(filter(lambda x: x[:2]=="QQ", shorts))),
         "VM":list(sorted(filter(lambda x: x[:2]=="VM", shorts))),
         "time":dummy,
         }

end_tag = dummy[-1]

outlist = []  
for i in range(len(dummy)):
    
    # first field and mesh
    field = heads[0]
    repath = vtk_path / files[field][i]
    msh = io.read(repath)
    # other fields
    for field in heads[1:]:
        
        # path to secondary mesh
        repath = vtk_path / files[field][i]
        # load
        msh2 = io.read(repath)
        # extract data
        key = list(msh2.point_data.keys())[0]
        msh.point_data.update({field:msh2.point_data[key]})
        
        
        if field == "Pressure":
            pres = msh.point_data["Pressure"]
            msh.point_data.update({"Pressure (cmH2O)":pres*10.1972})
        
        if field == "Jacobian":
            
            # Effective (complete) Jacobian
            # Determinant of the effective deformation tensor
            j = msh.point_data["Jacobian"]
            
            # Forward Jacobian; Effective Jacobian =  Forward Jacobian * Inverse Jacobian
            j_forward = j.copy()/j_prev
                    
            # Material coordinates
            # Porosity associated with the reference geometry
            material_porosity = j_forward - 1 + phi_ee_pointdata
            
            # Eulerian porosity
            # Porosity associated with the deformed geometry
            spatial_porosity = material_porosity/j_forward
            
            # Save all the relevant fields
            msh.point_data.update({"Jacobian Forward":j_forward})
            msh.point_data.update({"Jacobian":j})        
            msh.point_data.update({"Lagrangian Porosity":material_porosity})
            msh.point_data.update({"Eulerian Porosity":spatial_porosity})
            msh.point_data.update({"Delta Porosity":(spatial_porosity-phi_ee_pointdata)})
            
            # Note that Delta Porosity involves a difference between spatial and material 
            # vectors. This might be controversial (?). But it is akin to what has been
            # always done in the Biomechanical Analysis, where image intensities are 
            # transformed into porosities directly, each in their own respective geometry.
        
    # save
    tag = files["time"][i]
    
    finalname = dest / ("full_%s.vtu" % tag)
    msh.write(finalname)
    outlist += [finalname]
    if i%10==0: print("i: %4i  |   t = %s"%(i,tag))
    
    if tag == end_tag:
        print("Early break due to 'end_tag'='%s'"%end_tag)
        break
    
# %%
heads = ["Displacement","HYD",
         "Jacobian","Pressure","VM",
         "Eulerian Porosity", "Lagrangian Porosity", "QQint",
         "Jacobian", "Jacobian Forward"]

# Generate t = 0.000000000000
tag = "0.000000000000"
msh = io.read(finalname)
for field in heads:
    if not field in ["Jacobian", "Displacement", "Eulerian Porosity","Lagrangian Porosity","QQint"]:
        msh.point_data.update({field:np.zeros_like(msh.point_data[field])})
    elif field == "Jacobian":
        msh.point_data.update({'Jacobian':j_prev})
        msh.point_data.update({'Jacobian Forward':j_prev})

    elif field == "Displacement":
        msh.point_data.update({"u":np.zeros_like(msh.point_data["u"])})
    elif field == "Eulerian Porosity":
        msh.point_data.update({field:phi_ee_pointdata})
    elif field == "Lagrangian Porosity":
        msh.point_data.update({field:phi_ee_pointdata})
    elif field == "QQint":
        msh.point_data.update({"qq":np.zeros_like(msh.point_data["u"])})

    
    finalname = dest / ("full_%s.vtu" % tag)
    msh.write(finalname) 


# create json for time series

reout = [path.name for path in outlist]

series_data = {
    "file-series-version": "1.0",
    "files": []
}

# t=0 frame
series_data["files"].append({"name": "full_0.000000000000.vtu", "time": 0})

# rest of frames
for e, out in enumerate(reout):
    series_data["files"].append({"name": out, "time": e+1})

with open(dest / "simulation.vtu.series", "w") as f:
    json.dump(series_data, f, indent=2)

# --- Post-processing: generate physical-time series file ---


original_series = dest / "simulation.vtu.series"
fixed_series = dest / "simulation_physical.vtu.series"

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
