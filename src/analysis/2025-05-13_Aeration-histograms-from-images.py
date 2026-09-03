# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:02:53 2025

@author: angus
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import meshio as io

# %% Code associated to ARAOS previous work, modified

def compute_volumes(img_path, mask_path):

    img_ = nib.load(img_path) 
    mask_ = nib.load(mask_path)
    
    img_data = np.array(img_.dataobj)
    mask_data = np.array(mask_.dataobj)
    
    aff = img_.affine
    voxsize = aff[0,0]*aff[1,1]*aff[2,2]
    
    simple_mask_count = np.count_nonzero(mask_data==1)*voxsize/1000
    
    nonnat_mask = np.logical_and(img_data < -100, mask_data==1)
    
    nonnat_mask_count = np.count_nonzero(nonnat_mask)*voxsize/1000
    
    img_data[img_data>0.0] = 0.0
    img_data[img_data<-1000.0] = -1000.0
    gf = np.sum(img_data[mask_data==1]/-1000.0)/np.count_nonzero(mask_data==1)
    
    return simple_mask_count, nonnat_mask_count, gf


def retrieve_segmented_image(path,phase):
    '''Extract segmented voxel data'''
    
    
    image = path+"%s.nii.gz"%phase
    mask = path+"NEW_Mask_%s.nii.gz"%phase
    
    if not os.path.isfile(mask):
        mask = path+"RAW_Mask_%s.nii.gz"%phase
        if not os.path.isfile(mask):
            print("Mask not available")
            print("Path: %s"%path)
            return None
    
    img = nib.load(image)
    dimg = np.array(img.dataobj)
    
    seg = nib.load(mask)
    dseg = np.array(seg.dataobj)
    
    return dimg[dseg==1]

def compute_histogram(path,phase):
    '''Compute the histogram for a given image path at a phase'''
    img = retrieve_segmented_image(path,phase) 
    bins = [-1000,-900,-500,-100,100]
    index = np.digitize(img, bins)
    N = len(index)


    dic = {i:np.count_nonzero(index==i) for i in range(6)}
    hist = {"HIT":(dic[0]+dic[1])/N, # HyperInsuflufflated tissue ]-infty,-900] 
            "AT":dic[2]/N, # Normally-aerated tissue ]-900,-500]
            "PAT":dic[3]/N, # Poorly-aerated tissue ]-500,-100]
            "NAT":(dic[4]+dic[5])/N, # Non-aerated tissue ]-100, infty[
            }
    return hist

def analyze_subject(root, subject, states):
    print("Function: 'analyze_subject'")
    print(" > Subject: PIG%i"%subject)
    handle = {state:{} for state in states}
    
    for state in states:
        print(" > Processing state '%s'"%state)
        nifti_path = root+"PIG%i"%subject+"/%s/NIFTI/"%state
        for phase in ["Exp","Insp"]:
            handle[state].update({phase:compute_histogram(nifti_path,phase)})
    
    return handle

# %% Code from other ARAOS project

def safety_check(test_subj, time, requested_image, p2s, verbose=False):

    # create a main path
    path = os.path.join(p2s,test_subj,time,"NIFTI")
    # list the available images
    available_images = os.listdir(path)
    
    if verbose: print("\nChecking for the CT image ... ")
    if "%s"%requested_image in available_images:
        if verbose: print("CT image '%s' found"%requested_image)
        None
    else:
        if verbose: 
            print("CT image '%s' not found"%requested_image)
            print("SHUTDOWN")
        return False, None
    
    if verbose: print("\nChecking for the corresponding segmentation ... ")
    if "NEW_Mask_%s"%requested_image in available_images:
        seg = "NEW_Mask_%s"%requested_image
        if verbose: print("Using %s as a segmentation"%seg)
    elif "RAW_Mask_%s"%requested_image in available_images:
        seg = "RAW_Mask_%s"%requested_image
        if verbose: 
            print("Using '%s' as a segmentation"%seg)
            print("This histogram is only provisional\nThe segmentation should be processed")
    else:
        if verbose: 
            print("Segmentation unavailable")
            print("SHUTDOWN")
        return False, None
    
    return True, seg

def scout_segmentation_bounds(seg_img):
    ''' determine the bounds for a given segmentation '''
    
    print("scout_segmentation_bounds")
    sx,sy,sz = seg_img.shape
    
    for l in range(sz):
        vox_count = np.count_nonzero(seg_img.dataobj[:,:,l])
        if vox_count>0:
            #print("Lower axial (basal) bound found: slice=%i"%l)
            break
    basal = l
        
    for l in range(sz):
        vox_count = np.count_nonzero(seg_img.dataobj[:,:,sz-l-1])
        if vox_count>0:
            #print("Higher axial (apical) bound found: slice=%i"%(sz-l-1))
            break
    apical = sz-l-1
        
    for l in range(sy):
        vox_count = np.count_nonzero(seg_img.dataobj[:,l,:])
        if vox_count>0:
            #print("Lower coronal (dorsal) bound found: slice=%i"%l)
            break
    dorsal = l
    
    for l in range(sy):
        vox_count = np.count_nonzero(seg_img.dataobj[:,sy-l-1,:])
        if vox_count>0:
            #print("Higher coronal (ventral) bound found: slice=%i"%(sy-l-1))
            break
    ventral = sy-l-1
    
    return (basal,apical), (dorsal,ventral)


def histogram_from_voxel_list(roi_voxels, k0=0.0, k1=1.0):
    '''receive a list with multiple voxels in HU and classify them according
    into bins depending on their intensity0'''
    bins = [-1000,-900,-500,-100,100]
    
    if ((not (k0 is None)) and (not (k1 is None))):
        roi_voxels = np.array(roi_voxels)*k1+k0
    
    sorter = np.digitize(roi_voxels,bins)
    HIT = np.count_nonzero(sorter==1) # hyperinflated tissue
    AT = np.count_nonzero(sorter==2) # normally-aerated tissue
    PAT = np.count_nonzero(sorter==3) # poorly-aerated tissue
    NAT = np.count_nonzero(sorter==4) # non-aerated tissue
    OOB = np.count_nonzero(np.logical_or(sorter==0,sorter==5)) # out of bounds
    return HIT, AT, PAT,NAT,OOB

def determine_aeration_compartments_roiwise(ct_img, seg_img, direction,
                                            target_voxnum_per_roi):
    
    print("determine_aeration_compartments_roiwise")
    # retrieve bounds
    ba, vd = scout_segmentation_bounds(seg_img)
    if direction == "BA":
        bot, top = ba
    elif direction == "VD":
        bot, top = vd

    # this holds the current amount of voxels within a roi
    vox_count_hist = 0
    # this holds the voxels for the histogram locally
    voxel_holder = []
    # this keeps track of the total amount of voxels per roi
    count_history = []
    # holds the different histograms
    histogram_holder = []
    
    roi_number = 1
    print(" > Processing ROI#%i"%roi_number)
    
    for i in range(bot, top+1):
        # retrieve voxels
        
        if direction == "BA":
            ct_slice = ct_img.dataobj[:,:,i]
            seg_slice = seg_img.dataobj[:,:,i]
            roi_voxels = ct_slice[seg_slice==1]
    
        elif direction == "VD":
            ct_slice = ct_img.dataobj[:,i,:]
            seg_slice = seg_img.dataobj[:,i,:]
            roi_voxels = ct_slice[seg_slice==1]
        else:
            print("Error at direction '%s'!"%direction)
        
        # count the number of voxels
        vox_count = np.count_nonzero(seg_slice)
        
        # if we have not reached the amount of target voxels
        if vox_count_hist+vox_count < target_voxnum_per_roi:
            # append the voxels to the voxel holder
            voxel_holder += list(roi_voxels)
            vox_count_hist += vox_count
        else:
            roi_number +=1
            print(" > Processing ROI#%i"%roi_number)
            histogram_holder += [histogram_from_voxel_list(voxel_holder)]
            count_history += [vox_count_hist]
            # restart local ROI voxel holder
            voxel_holder = []
            # restart voxel count history
            vox_count_hist = 0
            
    if roi_number > 1:
        histogram_holder += [histogram_from_voxel_list(voxel_holder)]
        count_history += [vox_count_hist]
    
    # compute the histograms and append the data into the corresponding bins
    # placeholders
    clean_histograms = []
    NAT = []
    PAT = []
    AT = []
    HIT = []
    
    # evaluate each roi
    for roi in histogram_holder:
        roi_ = np.array(roi)
        roi_ = roi_/(roi[0]+roi[1]+roi[2]+roi[3])
        clean_histograms += [roi_]
        HIT += [roi_[0]]
        AT  += [roi_[1]]
        PAT += [roi_[2]]
        NAT += [roi_[3]]
    
    return {"HIT":HIT, "AT":AT, "PAT":PAT, "NAT":NAT} 


def process_subject_roiwise(test_subj, time, direction,
                    p2s="D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/", 
                    nrois=10):
    
    # hold the histograms for both Exp and Insp
    data_holder = {}
    # folder containing the images
    path = os.path.join(p2s,test_subj,time,"NIFTI")
    
    # retrieve histogram for both states
    for requested_image in ["Exp.nii.gz", "Insp.nii.gz"]:
    
        check, seg = safety_check(test_subj, time, requested_image, p2s)
        
        if not check:
            print("Safety check failed for:")
            print(" > Subject: %s"%test_subj)
            print(" > Time: %s"%time.split("_")[0])
            print(" > Requested image: %s"%requested_image)
            return None

        # load the ct image and the corresponding segmentation
        ct_img = nib.load(os.path.join(path,requested_image))
        seg_img = nib.load(os.path.join(path,seg))
        
        # count the number of voxels and assign the target number of roi voxels
        seg_voxels = np.count_nonzero(seg_img.dataobj)
        target_voxnum_per_roi = np.round(seg_voxels/nrois).astype(int)
        
        page  = determine_aeration_compartments_roiwise(ct_img, 
                                                        seg_img, 
                                                        direction,
                                                        target_voxnum_per_roi)
        
        key = "%s_%s"%(test_subj,time.split("_")[0])
        
        data_holder.update({requested_image[0]:page})
    
    return {key:data_holder}

# %% Assessing image histogram, best comparisons

# Declare paths

# We'll interpolate intensities towards a different mesh
subject = 3
protocol = 'ARDSnet'
key = (subject,protocol)
r_root = "D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/"
r_mesh_path = r_root+"PIG%i/%s/MESH/"%key
r_nifti_path = r_root+"PIG%i/%s/NIFTI/"%key
r_registration_path = r_root+"PIG%i/%s/REGISTRATION/"%key

# Images to be used
exp_ct = r_nifti_path+"Exp.nii.gz"
insp_ct = r_nifti_path+"Insp.nii.gz"
exp_seg = r_nifti_path+"RAW_Mask_Exp.nii.gz"
insp_seg = r_nifti_path+"RAW_Mask_Insp.nii.gz"


# %% Determine volumes

# Load images and determine voxel volume
eimg = nib.load(exp_ct); iimg = nib.load(insp_ct)
evox = np.abs(np.product([eimg.affine[i,i] for i in range(3)]))
ivox = np.abs(np.product([iimg.affine[i,i] for i in range(3)]))

# Analyze images and segmentations
e_simple_mask_count, _, egf =compute_volumes(exp_ct,exp_seg)
i_simple_mask_count, _, igf =compute_volumes(insp_ct,insp_seg)

# Determine raw delta volume
evol = e_simple_mask_count
ivol = i_simple_mask_count
dvol = ivol-evol

# Determine delta air
eair = e_simple_mask_count*egf
iair = i_simple_mask_count*igf
dair = iair-eair

# Present results
print("Subject (Protocol): PIG%i (%s)"%(subject,protocol))
print(" + The air difference between images is: %.1f (mL)"%dair)
print(" + The vol difference between images is: %.1f (mL)"%dvol)
print(" + Global volumetric strain: %.1f"%(dvol/evol*100)+" (%)")

# %%



# %% zero-dimensional IMAGE analysis

if False:
    # Analyze
    handle = analyze_subject(r_root,subject,[protocol])

    # Generate one-dimensional plot
    states = ["Exp","Insp"]
    figsize = (5,5)
    legend = ["Hyperinflated tissue", "Normally aerated tissue",\
               "Poorly aerated tissue", "Non aerated tissue"]

    # Recover information from the master dictionary.
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=figsize,dpi=150)
        
    for i,phase in enumerate(["Exp","Insp"]):
                            
        hst = handle[protocol][phase]
                
        # Tuning parameters for caption location
        ax.bar(i, hst['NAT'], color = "w",edgecolor='k') 
        ax.text(i,hst['NAT']/2,"%.2f"%hst['NAT'],color="k",ha='center',va='center')
        ax.bar(i, hst['PAT'],bottom=hst['NAT'], color = "k", alpha = 0.25, edgecolor="k")
        ax.text(i,hst['NAT']+hst['PAT']/2,"%.2f"%hst['PAT'],color="w",ha='center',va='center')
        ax.bar(i, hst['AT'],bottom=hst['NAT']+hst['PAT'], color="k", alpha = 0.5, edgecolor="k")
        ax.text(i,hst['NAT']+hst['PAT']+hst['AT']/2,"%.2f"%hst['AT'],color="w",ha='center',va='center')
        ax.bar(i, hst['HIT'],bottom=hst['NAT']+hst['PAT']+hst['AT'], color = "k", edgecolor="k", alpha = 0.999)
        ax.text(i,hst['NAT']+hst['PAT']+hst['AT']+hst['HIT']/2,"%.2f"%hst['HIT'],color="w",ha='center',va='center')

    ax.set_xlabel("States",size=18)
    ax.set_ylabel("Fraction [-]",size=18)
    ax.set_yticks(np.linspace(0,1,6))
    ax.set_yticklabels(["%.1f"%lab for lab in np.linspace(0,1,6)],size=16)
    ax.set_xticks([0,1])
    ax.set_xticklabels(states,size=18)
    
    plt.tight_layout()
    
# %% One dimensional IMAGE analysis

if False:
    
    directions = ['VD','BA']

    mngr = {direction:process_subject_roiwise("PIG%i"%subject,protocol,direction, p2s=r_root,nrois=10) for direction in directions}
    
    figsize = (10,8)

    legend = ["Hyperinflated tissue", "Normally aerated tissue",\
               "Poorly aerated tissue", "Non aerated tissue"]

    # Recover information from the master dictionary.
    dirs = ['VD','BA']
    
    for d in dirs:
                
        holder = mngr[d]["PIG%i_%s"%key]
        
        fig,axes = plt.subplots(nrows=2,ncols=1,figsize=figsize)
        
        for ax,phase in zip(axes,["E","I"]):
            
            for i in range(10):
                
                binner = [holder[phase][comp][i] for comp in ['NAT','PAT','AT','HIT']]
                
                # Tuning parameters for caption location
                ax.bar(i, binner[0], color = "w",edgecolor='k') 
                ax.bar(i, binner[1],bottom=binner[0], color = "k", alpha = 0.25, edgecolor="k")
                ax.bar(i, binner[2],bottom=binner[0]+binner[1], color="k", alpha = 0.5, edgecolor="k")
                ax.bar(i, binner[3],bottom=binner[0]+binner[1]+binner[2], color = "k", edgecolor="k", alpha = 0.999)
    
                ax.set_xlabel("ROIS")
                ax.set_ylabel("Fraction [-]")
                ax.text(-1.0,-0.12,d[0])
                ax.text(10.0,-0.12,d[1],size=8)
    
            ax.set_xticks(np.arange(10))
            ax.set_xticklabels(np.arange(1,11,1))
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.90, hspace=0.40, wspace=1.0)
    
# %%

