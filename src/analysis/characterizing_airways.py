# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:01:47 2025

@author: angus
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib
import AirwayManager as awm

root = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/"
ids = [2,3,4,5,6]

skel_name = "skel.vtu"

manager = {"PIG2":{"path":"C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG2/ARDSnet/MESH-15/",
                   "new_nii_name":"NEW_Airways_Exp.nii.gz",
                   "raw_nii_name":"RAW_Airways_Exp.nii.gz"},
             "PIG3":{"path":"C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG3/ARDSnet/MESH-15/",
                     "new_nii_name":"NEW_Airways_Exp.nii.gz",
                     "raw_nii_name":"RAW_Airways_Exp.nii.gz"},
             "PIG4":{"path":"C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG4/ARDSnet/MESH-15/",
                     "new_nii_name":"NEW_Airways_Exp.nii.gz",
                     "raw_nii_name":"RAW_Airways_Exp.nii.gz"},
             "PIG5":{"path":"C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/MESH/",
                     "new_nii_name":"Airway_Mask_Exp.nii.gz",
                     "raw_nii_name":"Airway_Mask_Exp.nii.gz"},
             "PIG6":{"path":"C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/ARDSnet/MESH/",
                     "new_nii_name":"NEW_Airways_Exp.nii.gz",
                     "raw_nii_name":"RAW_Airways_Exp.nii.gz"},}

# %%

print("\nChecking folder availability:")
for iid in ids:
    if os.path.isdir(manager["PIG%i"%iid]["path"]): 
        print(" > 'PIG%i' found"%iid)
    else:
        print(" > 'PIG%i' not found"%iid)

print("\nChecking for airway availability:")
for iid in ids:
    if os.path.isfile(manager["PIG%i"%iid]["path"]+skel_name): 
        print(" > 'PIG%i' found"%iid)
    else:
        print(" > 'PIG%i' not found"%iid)

print("\nDetermining equivalent airway resistance:")
for iid in ids:
    target = manager["PIG%i"%iid]["path"]+skel_name
    R = awm.determine_equivalent_resistance(target,verbose=False)
    print(" > 'PIG%i': %.2e"%(iid,R))
    manager["PIG%i"%iid] = {"Absolute resistance":R}
    if iid == 5:
        reference_resistance = R

print("\nDetermining relative airway resistance:")

for iid in ids:
    R = manager["PIG%i"%iid]["Absolute resistance"]
    relative_resistance = R/reference_resistance
    print(" > 'PIG%i': %.2f"%(iid,relative_resistance))

    manager["PIG%i"%iid]["Relative resistance"] = {"Relative resistance":relative_resistance}
    
# %%

print("\nAre airways images availables?")
for iid in ids:
    nifti = root%iid + "NIFTI/"
    tag = "PIG%i"%iid
    raw_file = nifti+manager[tag]["raw_nii_name"]
    new_file = nifti+manager[tag]["new_nii_name"]

    if os.path.isfile(raw_file):
        print("'%s' Raw file found!"%tag)
        raw = nib.load(raw_file)
        nraw = np.count_nonzero(np.array(raw.dataobj)==1)
    else:
        print("'%s' Raw file not found"%tag)
        
    if os.path.isfile(new_file):
        print("'%s' New file found!"%tag)
        raw = nib.load(new_file)
        nnew = np.count_nonzero(np.array(raw.dataobj)==1)
    else:
        print("'%s' New file not found"%tag)
    
    print("'%s' new/raw ratio: %.2f"%(tag,nnew/nraw))
    print(" > n_raw voxels = %i"%nraw)
    print(" > n_new voxels = %i"%nnew)
    print()
        
        

