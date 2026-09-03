# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:43:35 2024

@author: angus
"""

import nibabel as nib
import numpy as np
import os


def global_image_analysis(img_path,seg_path):

    # Load iamges
    seg_nii = nib.load(seg_path)
    img_nii = nib.load(img_path)
    
    # Determine raw volume
    voxel_volume = np.abs(np.prod(np.diagonal(img_nii.affine)[0:3]))
    voxel_count = np.count_nonzero(np.array(seg_nii.dataobj)==1)
    segmentation_volume = voxel_count*voxel_volume*1e-3 # in mL
    
    # Load image
    image_voxels = np.array(img_nii.dataobj)[np.array(seg_nii.dataobj)==1]
    
    # Saturate image
    image_voxels[image_voxels>0] = 0
    image_voxels[image_voxels<-1000] = -1000
    
    # Transform into porosity
    image_voxels = image_voxels/-1000
    
    # Determine overall air and tissue volumes
    gas_fraction = np.sum(image_voxels)/voxel_count
    tissue_fraction = 1-gas_fraction
    
    gas_volume = gas_fraction*segmentation_volume
    tissue_volume = segmentation_volume-gas_volume
    
    return segmentation_volume, gas_volume, tissue_volume, gas_fraction, tissue_fraction

# %%

path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/"

subjects = ["PIG%i"%i for i in [2,3,4,5,6]]
treatments = ["ARDSnet"]#, "APRV"]
states = ["Exp"]#,"Insp"]


fields_of_interest = ["SegmentationVolume","GasVolume","TissueVolume","GasFraction","TissueFraction"]

data_dictionary = {subject:{treatment:{state:{} for state in states} for treatment in treatments} for subject in subjects}

for subject in subjects:
    
    print("\nSubject: %s"%subject)
    
    for treatment in treatments:
        
        print(" > Treatment: %s"%treatment)
        path_to_images = path+"%s/%s/NIFTI/"%(subject,treatment)

        for state in states:
            
            problem = False
            print("   - State: %s"%state)
            
            # Determining images and segmentation paths:
            attempts = 0
            for root in ["SMOOTH_%s.nii.gz","%s.nii.gz"]:
                
                img_path = path_to_images+root%state
                
                if os.path.isfile(img_path):
                    break
                
                attempts += 1
                
                if attempts == 2:
                    print("No image was found for %s-%s at state %s"%(subject,treatment,state))
                    problem = True
                
            attempts = 0
            for root in ["NEW_Mask_%s.nii.gz","RAW_Mask_%s.nii.gz"]:
                
                seg_path = path_to_images+root%state
                
                if os.path.isfile(seg_path):
                    break
                
                attempts += 1
                
                if attempts == 2:
                    print("No segmentation was found for %s-%s at state %s"%(subject,treatment,state))
                    problem = True
                    
            
            if not problem:
                res = global_image_analysis(img_path, seg_path)
                data_dictionary[subject][treatment][state].update({field:res[i] for i,field in enumerate(fields_of_interest)})
            
# %%

formatter = {"SegmentationVolume":"%4.0f  ",
             "GasVolume":"%4.0f  ",
             "TissueVolume":"%4.0f  ",
             "GasFraction":"%4.2f  ",
             "TissueFraction":"%4.2f  "}

treatment = treatments[0]
subject = subjects[1]

print("Subject: %s | Treatment: %s\n"%(subject,treatment))
print(" "*22+"    E     I ")
for field in fields_of_interest:
    data = []
    for state in states:
        data += [data_dictionary[subject][treatment][state][field]]
    data = tuple(data)
    print("%20s"%field+" | "+(formatter[field]*2)%data)
    
# %%

formatter = {"SegmentationVolume":"%4.0f  ",
             "GasVolume":"%4.0f  ",
             "TissueVolume":"%4.0f  ",
             "GasFraction":"%4.2f  ",
             "TissueFraction":"%4.2f  "}

subject = subjects[4]

print("Subject: %s "%(subject))
print(" "*18+"       ARDSnet APRV ")
for state,field in [("Exp","SegmentationVolume"),
                    ("Exp","TissueVolume"),
                    ("Exp","GasVolume"),
                    ("Insp","TissueVolume"),
                    ("Insp","GasVolume"),
                    ]:
    data = []
    for treatment in treatments:
        data += [data_dictionary[subject][treatment][state][field]]
    data = tuple(data)
    print("%20s (%1s)"%(field,state[0])+" | "+(formatter[field]*2)%data)