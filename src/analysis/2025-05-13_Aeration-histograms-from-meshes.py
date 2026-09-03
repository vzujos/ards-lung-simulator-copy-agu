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
from collections import defaultdict
from scipy.sparse import coo_matrix
import legacy.ROIAnalysis as ROI

# %% From analysis ARAOS

def retrieve_global_histogram(mesh_path, 
                              bins=[0.0,0.1,0.5,0.9,1.0],
                              verbose=True,
                              porosity_field='Eulerian Porosity'):
    
    '''
    Generate a histogram for the whole lung mesh.
    '''
    
    # Read the mesh
    mesh = io.read(mesh_path)
    # Extract the Eulerian porosity
    porosity = mesh.point_data[porosity_field] 
    N = len(porosity)
    # Clasify the points according to their intensity (porosity)
    digit = np.digitize(porosity,bins)
    # Generate a counter for each possible classification
    counter = {i:None for i in range(6)}
    # Organize data
    for bin_ in range(6):
        counter[bin_] = np.count_nonzero(digit==bin_)
    # Keep track of the names
    namer = {0:"OOB-NAT", # Out-of-bounds non-aerated tissue
             1:"NAT", # Non-aerated tissue
             2:"PAT", # Poorly aerated tissue
             3:" AT", # Normally aerated tissue
             4:"HIT", # Hyperinflated tissue
             5:"OOB-HIT"} # Out-of-bounds hypperinflated tissue
    
    # Keep the final bins
    binner = {i:None for i in range(4)}

    # Generate the final sorting on each bin
    for i in [1,2,3,4]:
        if i == 1:
            binner[i-1] = (counter[0]+counter[1])/N
            if verbose: print("NAT: %.1f"%((counter[0]+counter[1])/N*100)+"%")
        elif i == 4:
            binner[i-1] = (counter[4]+counter[5])/N
            if verbose: print("HIT: %.1f"%((counter[4]+counter[5])/N*100)+"%")
        else:
            binner[i-1] = counter[i]/N
            if verbose: print("%s: %.1f"%(namer[i],counter[i]*100/N)+"%")
    
    return binner, counter, digit

def retrieve_regional_histogram(mesh_path,
                                direction,
                                weights=None,
                                nrois=10,
                                verbose=False,
                                deformed_state=True,
                                bins=[0.0,0.1,0.5,0.9,1.0],
                                porosity_field='Eulerian Porosity'):
   
    mesh = io.read(mesh_path)
    
    if deformed_state: # Add deformation field to the mesh points
        xyz = mesh.points
        u = mesh.point_data['u']
        xyz += u
    else: # Use raw point data
        xyz = mesh.points
        
    # Retrieve porosity field
    porosity = mesh.point_data[porosity_field] 


    # Define the directions in use
    dirs = {"BA" : np.mat([0.,0.,1.]).T, # BA tested; Direction checks out 
            "VD" : np.mat([0.,1.,0.]).T, # VD tested; Direction checks out
            "RL" : np.mat([1.,0.,0.]).T}

    # Dummy; Node mass should be used. How do I compute it?
    # TODO: Use the actual mass
    if weights is None:
        w = np.ones(xyz.shape[0])
    else:
        w = weights

    _, id_roi = ROI.IsoVolumetricSegmentation(dirs[direction],w,xyz,nrois)

    counter = {roi:{i:None for i in range(6)} for roi in range(nrois)}
    gen_binner = {roi:{i:None for i in [1,2,3,4]} for roi in range(nrois)}


    namer = {0:"OOB-NAT", # Out-of-bounds non-aerated tissue
             1:"NAT", # Non-aerated tissue
             2:"PAT", # Poorly aerated tissue
             3:" AT", # Normally aerated tissue
             4:"HIT", # Hyperinflated tissue
             5:"OOB-HIT"} # Out-of-bounds hypperinflated tissue

    for roi in range(nrois):
        
        roimask = id_roi == roi
        roi_porosity = porosity[roimask]
        N = len(roi_porosity)
        
        digit = np.digitize(roi_porosity,bins)

        for bin_ in range(6):
            counter[roi][bin_] = np.count_nonzero(digit==bin_)

        for i in [1,2,3,4]:
            if i == 1:
                gen_binner[roi][i-1] = (counter[roi][0]+counter[roi][1])/N
                if verbose: print("NAT: %.1f"%((counter[roi][0]+counter[roi][1])/N*100)+"%")
            elif i == 4:
                gen_binner[roi][i-1] = (counter[roi][4]+counter[roi][5])/N
                if verbose: print("HIT: %.1f"%((counter[roi][4]+counter[roi][5])/N*100)+"%")
            else:
                gen_binner[roi][i-1] = counter[roi][i]/N
                if verbose: print("%s: %.1f"%(namer[i],counter[roi][i]*100/N)+"%")

    return gen_binner, counter


# %% From biomechanical analysis

def Tet4DefGradSmoothingOptiSPARSE(xyz, defmap, LM):
	"""
	This subroutine computes the assemble of mass vector and force vector (matrix)
	Input:
	  xyz		: array with node coordinates in reference configuration
	  defmap	 : array with nodal deformation mapping vector 
	  LM		 : connectivity matrix

	Output:
	  Mlumped   : Lumped mass matrix (in vector form)
	  P		 : Weighted deformation-gradient vector
	"""	
	me  = LM
	q   = xyz
	phi = defmap
	
	V = np.zeros((me.shape[0],4,4))
	
	qme   = q[me[:]]
	phime = phi[me[:]]
	
	ones = np.ones((me.shape[0],4))
	x = qme[:,:,0]
	y = qme[:,:,1]
	z = qme[:,:,2]
	
	T = np.column_stack((ones[:],x[:],y[:],z[:]))
	V[:,0,0]=T[:,0];V[:,0,1]=T[:,1];V[:,0,2]=T[:,2];V[:,0,3]=T[:,3];
	V[:,1,0]=T[:,4];V[:,1,1]=T[:,5];V[:,1,2]=T[:,6];V[:,1,3]=T[:,7];
	V[:,2,0]=T[:,8];V[:,2,1]=T[:,9];V[:,2,2]=T[:,10];V[:,2,3]=T[:,11]
	V[:,3,0]=T[:,12];V[:,3,1]=T[:,13];V[:,3,2]=T[:,14];V[:,3,3]=T[:,15]
	
	ve0=np.zeros(me.shape[0])
	ve0[:]=np.linalg.det(V[:])
	ve0=ve0/6.
	
	xdef=phime[:,:,0]
	ydef=phime[:,:,1]
	zdef=phime[:,:,2]
	
	T=np.column_stack((ones[:],xdef[:],ydef[:],zdef[:]))
	
	V[:,0,0]=T[:,0];V[:,0,1]=T[:,1];V[:,0,2]=T[:,2];V[:,0,3]=T[:,3];
	V[:,1,0]=T[:,4];V[:,1,1]=T[:,5];V[:,1,2]=T[:,6];V[:,1,3]=T[:,7];
	V[:,2,0]=T[:,8];V[:,2,1]=T[:,9];V[:,2,2]=T[:,10];V[:,2,3]=T[:,11]
	V[:,3,0]=T[:,12];V[:,3,1]=T[:,13];V[:,3,2]=T[:,14];V[:,3,3]=T[:,15]
	
	ve=np.zeros(me.shape[0])
	ve[:]=np.linalg.det(V[:])
	ve=ve/6.
	
	x=qme[:,:,0]
	y=qme[:,:,1]
	z=qme[:,:,2]
	
	a1=(y[:,3]-y[:,1])*(z[:,2]-z[:,1])-(y[:,2]-y[:,1])*(z[:,3]-z[:,1])	  #yd[3,1]*zd[2,1]-yd[2,1]*zd[3,1]
	a2=(y[:,2]-y[:,0])*(z[:,3]-z[:,2])-(y[:,2]-y[:,3])*(z[:,0]-z[:,2])	  #yd[2,0]*zd[3,2]-yd[2,3]*zd[0,2]
	a3=(y[:,1]-y[:,3])*(z[:,0]-z[:,3])-(y[:,0]-y[:,3])*(z[:,1]-z[:,3])	  #yd[1,3]*zd[0,3]-yd[0,3]*zd[1,3]
	a4=(y[:,0]-y[:,2])*(z[:,1]-z[:,0])-(y[:,0]-y[:,1])*(z[:,2]-z[:,0])	  #yd[0,2]*zd[1,0]-yd[0,1]*zd[2,0]
	
	b1=(x[:,2]-x[:,1])*(z[:,3]-z[:,1])-(x[:,3]-x[:,1])*(z[:,2]-z[:,1])	  #xd[2,1]*zd[3,1]-xd[3,1]*zd[2,1]
	b2=(x[:,3]-x[:,2])*(z[:,2]-z[:,0])-(x[:,0]-x[:,2])*(z[:,2]-z[:,3])	  #xd[3,2]*zd[2,0]-xd[0,2]*zd[2,3]
	b3=(x[:,0]-x[:,3])*(z[:,1]-z[:,3])-(x[:,1]-x[:,3])*(z[:,0]-z[:,3])	  #xd[0,3]*zd[1,3]-xd[1,3]*zd[0,3]
	b4=(x[:,1]-x[:,0])*(z[:,0]-z[:,2])-(x[:,2]-x[:,0])*(z[:,0]-z[:,1])	  #xd[1,0]*zd[0,2]-xd[2,0]*zd[0,1]
	
	c1=(x[:,3]-x[:,1])*(y[:,2]-y[:,1])-(x[:,2]-x[:,1])*(y[:,3]-y[:,1])	  #xd[3,1]*yd[2,1]-xd[2,1]*yd[3,1]
	c2=(x[:,2]-x[:,0])*(y[:,3]-y[:,2])-(x[:,2]-x[:,3])*(y[:,0]-y[:,2])	  #xd[2,0]*yd[3,2]-xd[2,3]*yd[0,2]
	c3=(x[:,1]-x[:,3])*(y[:,0]-y[:,3])-(x[:,0]-x[:,3])*(y[:,1]-y[:,3])	  #xd[1,3]*yd[0,3]-xd[0,3]*yd[1,3]
	c4=(x[:,0]-x[:,2])*(y[:,1]-y[:,0])-(x[:,0]-x[:,1])*(y[:,2]-y[:,0])	  #xd[0,2]*yd[1,0]-xd[0,1]*yd[2,0]
	
	
	DN=[[a1[:]/(6*ve0[:]),a2[:]/(6*ve0[:]),a3[:]/(6*ve0[:]),a4[:]/(6*ve0[:])],[b1[:]/(6*ve0[:]),b2[:]/(6*ve0[:]),b3[:]/(6*ve0[:]),b4[:]/(6*ve0[:])],[c1[:]/(6*ve0[:]),c2[:]/(6*ve0[:]),c3[:]/(6*ve0[:]),c4[:]/(6*ve0[:])]]		
	
	DN=np.array(DN).T
	
	F11=DN[:,0,0]*phime[:,0,0]+DN[:,1,0]*phime[:,1,0]+DN[:,2,0]*phime[:,2,0]+DN[:,3,0]*phime[:,3,0]
	F12=DN[:,0,1]*phime[:,0,0]+DN[:,1,1]*phime[:,1,0]+DN[:,2,1]*phime[:,2,0]+DN[:,3,1]*phime[:,3,0]
	F13=DN[:,0,2]*phime[:,0,0]+DN[:,1,2]*phime[:,1,0]+DN[:,2,2]*phime[:,2,0]+DN[:,3,2]*phime[:,3,0]
	
	F21=DN[:,0,0]*phime[:,0,1]+DN[:,1,0]*phime[:,1,1]+DN[:,2,0]*phime[:,2,1]+DN[:,3,0]*phime[:,3,1]
	F22=DN[:,0,1]*phime[:,0,1]+DN[:,1,1]*phime[:,1,1]+DN[:,2,1]*phime[:,2,1]+DN[:,3,1]*phime[:,3,1]
	F23=DN[:,0,2]*phime[:,0,1]+DN[:,1,2]*phime[:,1,1]+DN[:,2,2]*phime[:,2,1]+DN[:,3,2]*phime[:,3,1]
	
	F31=DN[:,0,0]*phime[:,0,2]+DN[:,1,0]*phime[:,1,2]+DN[:,2,0]*phime[:,2,2]+DN[:,3,0]*phime[:,3,2]
	F32=DN[:,0,1]*phime[:,0,2]+DN[:,1,1]*phime[:,1,2]+DN[:,2,1]*phime[:,2,2]+DN[:,3,1]*phime[:,3,2]
	F33=DN[:,0,2]*phime[:,0,2]+DN[:,1,2]*phime[:,1,2]+DN[:,2,2]*phime[:,2,2]+DN[:,3,2]*phime[:,3,2]
	
	
	F=np.zeros((me.shape[0],3,3))
	
	F[:,0,0]=F11[:]
	F[:,0,1]=F12[:]
	F[:,0,2]=F13[:]
	
	F[:,1,0]=F21[:]
	F[:,1,1]=F22[:]
	F[:,1,2]=F23[:]
	
	F[:,2,0]=F31[:]
	F[:,2,1]=F32[:]
	F[:,2,2]=F33[:]
	
	Ft=np.zeros((me.shape[0],3,3))
	
	Ft[:,0,0]=F11[:]
	Ft[:,0,1]=F21[:]
	Ft[:,0,2]=F31[:]
	
	Ft[:,1,0]=F12[:]
	Ft[:,1,1]=F22[:]
	Ft[:,1,2]=F32[:]
	
	Ft[:,2,0]=F13[:]
	Ft[:,2,1]=F23[:]
	Ft[:,2,2]=F33[:]
	
	
	Ig=me.reshape((me.shape[0]*4,))
	Mg=np.array([ve0[:],ve0[:],ve0[:],ve0[:]])*1./4.
	Mg=Mg.T
	Mg=Mg.reshape((me.shape[0]*4,))
	M=coo_matrix((Mg,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	Mlumped=np.array(M.diagonal())
	
	
	e11=np.array([ve0[:]*F11[:],ve0[:]*F11[:],ve0[:]*F11[:],ve0[:]*F11[:]])*1./4.
	e11=e11.T
	e11=e11.reshape((me.shape[0]*4,))
	E11=coo_matrix((e11,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E11=np.array(E11.diagonal())
	
	e12=np.array([ve0[:]*F12[:],ve0[:]*F12[:],ve0[:]*F12[:],ve0[:]*F12[:]])*1./4.
	e12=e12.T
	e12=e12.reshape((me.shape[0]*4,))
	E12=coo_matrix((e12,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E12=np.array(E12.diagonal())
	
	
	e13=np.array([ve0[:]*F13[:],ve0[:]*F13[:],ve0[:]*F13[:],ve0[:]*F13[:]])*1./4.
	e13=e13.T
	e13=e13.reshape((me.shape[0]*4,))
	E13=coo_matrix((e13,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E13=np.array(E13.diagonal())
	
	e21=np.array([ve0[:]*F21[:],ve0[:]*F21[:],ve0[:]*F21[:],ve0[:]*F21[:]])*1./4.
	e21=e21.T
	e21=e21.reshape((me.shape[0]*4,))
	E21=coo_matrix((e21,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E21=np.array(E21.diagonal())
	
	
	e22=np.array([ve0[:]*F22[:],ve0[:]*F22[:],ve0[:]*F22[:],ve0[:]*F22[:]])*1./4.
	e22=e22.T
	e22=e22.reshape((me.shape[0]*4,))
	E22=coo_matrix((e22,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E22=np.array(E22.diagonal())
	
	e23=np.array([ve0[:]*F23[:],ve0[:]*F23[:],ve0[:]*F23[:],ve0[:]*F23[:]])*1./4.
	e23=e23.T
	e23=e23.reshape((me.shape[0]*4,))
	E23=coo_matrix((e23,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E23=np.array(E23.diagonal())
	
	e31=np.array([ve0[:]*F31[:],ve0[:]*F31[:],ve0[:]*F31[:],ve0[:]*F31[:]])*1./4.
	e31=e31.T
	e31=e31.reshape((me.shape[0]*4,))
	E31=coo_matrix((e31,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E31=np.array(E31.diagonal())
	
	e32=np.array([ve0[:]*F32[:],ve0[:]*F32[:],ve0[:]*F32[:],ve0[:]*F32[:]])*1./4.
	e32=e32.T
	e32=e32.reshape((me.shape[0]*4,))
	E32=coo_matrix((e32,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E32=np.array(E32.diagonal())
	
	e33=np.array([ve0[:]*F33[:],ve0[:]*F33[:],ve0[:]*F33[:],ve0[:]*F33[:]])*1./4.
	e33=e33.T
	e33=e33.reshape((me.shape[0]*4,))
	E33=coo_matrix((e33,(Ig,Ig)), shape=(q.shape[0],q.shape[0]))
	E33=np.array(E33.diagonal())
	
	P=np.array([E11,E12,E13,E21,E22,E23,E31,E32,E33]).T
	
	return Mlumped, P, ve, ve0


# %% Correction schemes for intensities and porosities

def find_surface_points(ien):
    face_dict = defaultdict(int)
    
    # Extract faces from the tetrahedrons
    for elem in ien:
        faces = [
            tuple(sorted([elem[0], elem[1], elem[2]])),
            tuple(sorted([elem[0], elem[1], elem[3]])),
            tuple(sorted([elem[0], elem[2], elem[3]])),
            tuple(sorted([elem[1], elem[2], elem[3]]))
        ]
        
        # Count faces
        for face in faces:
            face_dict[face] += 1
    
    # Surface faces are those that appear only once
    surface_faces = [face for face, count in face_dict.items() if count == 1]
    
    # Extract surface points from the surface faces
    surface_points = set()
    for face in surface_faces:
        surface_points.update(face)
    
    return list(surface_points)

def count_surface_points_per_cell(ien, surface_points):
    # Convert surface_points to a set for faster lookup
    surface_points_set = set(surface_points)
    
    # Initialize an array to hold the count of surface points per cell
    counts = np.zeros(len(ien), dtype=int)
    
    # Iterate over each element in the connectivity matrix
    for i, elem in enumerate(ien):
        # Count how many points in the element are in the surface_points_set
        counts[i] = sum(1 for point in elem if point in surface_points_set)
    
    return counts

def compute_normal(sample_xyz, point_on_surface, verbose=False):
    """
    Computes a unit normal vector based on the number of True booleans in `point_on_surface`.

    Parameters:
    - sample_xyz: numpy array of shape (4,3) with the coordinates of the tetrahedron vertices
    - point_on_surface: list of boolean values indicating which points are on the surface
    - verbose: boolean flag for printing additional information (default is False)

    Returns:
    - normal_unit: unit normal vector
    """
    # Convert to numpy arrays if not already
    sample_xyz = np.array(sample_xyz)
    point_on_surface = np.array(point_on_surface)

    # Count number of True values
    num_true = np.sum(point_on_surface)
    
    if num_true == 3:
        # Case 1: Three True points and one False point
        # Extract the three points on the surface
        surface_points = sample_xyz[point_on_surface]
        
        # Select the fourth point not on the surface
        fourth_point = sample_xyz[~point_on_surface][0]
        
        # Compute vectors from surface points to use in cross product
        v1 = surface_points[1] - surface_points[0]
        v2 = surface_points[2] - surface_points[0]
        
        # Compute the normal using cross product
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        
        # Create a vector from one of the surface points to the fourth point
        to_fourth = fourth_point - surface_points[0]
        
        # Compute the dot product between the normal and the vector to the fourth point
        dot_product = np.dot(normal, to_fourth)
        
        # Determine the direction of the normal
        if dot_product > 0:
            if verbose: print("Normal points towards the fourth point")
            return normal
        else:
            if verbose: print("Normal points away from the fourth point")
            return -normal

    elif num_true == 2:
        # Case 2: Two True points and two False points
        # Extract True and False points
        true_points = sample_xyz[point_on_surface]
        false_points = sample_xyz[~point_on_surface]

        # Ensure there are exactly two True and two False points
        assert len(true_points) == 2 and len(false_points) == 2, "There should be exactly two True and two False points"

        # Form vectors from the True points
        v1 = true_points[1] - true_points[0]  # Vector between the two True points
        v2 = false_points[1] - false_points[0]  # Vector between the two False points

        # Compute the cross product
        normal = np.cross(v1, v2)

        # Normalize the normal vector
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("The cross product resulted in a zero vector. The points might be collinear.")
        normal_unit = normal / norm

        # Check the direction of the normal vector
        # Calculate a vector from the True points to the False points
        vector_true_to_false = np.mean(false_points, axis=0) - np.mean(true_points, axis=0)

        # Ensure the normal vector is facing the False points
        if np.dot(normal_unit, vector_true_to_false) < 0:
            normal_unit = -normal_unit

        return normal_unit

    elif num_true == 1:
        # Case 3: One True point and three False points
        # Extract the single True point and the three False points
        true_point = sample_xyz[point_on_surface]
        false_points = sample_xyz[~point_on_surface]

        # Ensure there is exactly one True and three False points
        assert len(false_points) == 3 and len(true_point) == 1, "There should be exactly one True and three False points"

        # Compute the centroid of the triangle formed by the three False points
        centroid = np.mean(false_points, axis=0)

        # Form the vector from the True point to the centroid
        vector_true_to_centroid = centroid - true_point[0]

        # Normalize the vector
        norm = np.linalg.norm(vector_true_to_centroid)
        if norm == 0:
            raise ValueError("The True point coincides with the centroid of the face.")
        normal_unit = vector_true_to_centroid / norm

        return normal_unit

    else:
        raise ValueError("The number of True booleans should be 1, 2, or 3.")

def determine_shape_function(X):

    X_ = np.vstack([np.ones((1,4)),X.T])
    A = np.linalg.inv(X_)
    
    return A



def correct_porosity(xyz, ien, Phi,
                     saturate_porosities=True,
                     max_loops = 4):
    
    ''' 
    Solve a no-flow problem on the surface of the finite element mesh. Intensities or
    porosities are computed by using that of points on the neighbourhood of every 
    point of interest. The result is a field with less anomalies associated to poor
    interpolation.
    '''
    
    # Determine surface points
    surface_points = find_surface_points(ien)
    
    # Count how many surface points are in each cell
    counts = count_surface_points_per_cell(ien,surface_points)
    
    # Sort the element ids according to their surface point count
    cell_ids = np.arange(len(ien))
    surface_classifier = {count:cell_ids[counts==count] for count in [0,1,2,3,4]}
    
    # Case: Dual scenario 1, 2 and 3 nodes on surface and one inside
    
    # Determine surface node ids
    surface_cells = np.append(surface_classifier[2],surface_classifier[3])
    surface_cells = np.append(surface_cells, surface_classifier[1],)
    point_ids = np.unique(ien[surface_cells])
    
    # Points that are going to be acted on
    active_surface_points = list(set(point_ids).intersection(set(surface_points)))
    
    # Change point ID to DoF ()
    translator = {"point_to_dof":{point:e for e,point in enumerate(active_surface_points)},
                  "dof_to_point":{e:point for e,point in enumerate(active_surface_points)}, }
    
    #  Generate a list for each point that carries the cells associated with each surface point
    point_in_cell_indexer = {point:[] for point in active_surface_points}
    
    for point in active_surface_points:
        point_in_cell_indexer[point] = list(np.argwhere(ien==point)[:,0])
    
    # For each type-3 element (surface)
    ndof = len(active_surface_points)
    K = np.zeros((ndof,ndof),dtype=float)
    f = np.zeros((ndof,1),dtype=float)
    
    # Now operate on every cell of interest
    for surface_id in surface_cells:
        
        # Retrieve the point ids associated with the surface
        points = ien[surface_id]
        # Sort the type of points
        Bs = []; As = []
        
        point_on_surface = []
        for point in points:
            if point in active_surface_points:
                Bs += [point]
                point_on_surface += [True]
            else:
                As += [point]
                point_on_surface += [False]
                
        # Transform mask into a numpy array
        point_on_surface = np.array(point_on_surface)
                
        # Determine shape functions
        N = determine_shape_function(xyz[points])
        
        # Determine gradN_x = [bi, ci, di]
        gradN_x = N[:,1:4]
        
        # Determine normal
        normal = compute_normal(xyz[points],point_on_surface)
        
        # The normal derivative of N
        dNdn = np.dot(gradN_x,normal)
        
        # Roll for C
        for ci,(point_id_c,in_surface_c) in enumerate(zip(points,point_on_surface)):
            
            # Skip out of surface nodes
            if not in_surface_c:
                continue
            
            J =  translator["point_to_dof"][point_id_c]
            
            # Roll for B
            for bj,(point_id_b,in_surface_b) in enumerate(zip(points,point_on_surface)):
                
                # Skip out of surface nodes
                if not in_surface_b:
                    
                    aj = bj # the local index associated to the out of 
                    point_id_a = point_id_b
                    
                    # Contribution associated to the right-hand term (Phi_A and such)
                    f[J,0] += -dNdn[ci]*dNdn[aj]*Phi[point_id_a]
                    
                    continue
                
                # Retrieve DoF numbering associated to index B
                I = translator["point_to_dof"][point_id_b]
                
                # Add contribution of this surface element
                K[I,J] += dNdn[ci]*dNdn[bj]
            
    # Solve the linear system
    Phi_sol = np.linalg.solve(K,f)
    
    # Generate the new porosity field
    new_phi = Phi.copy()
    
    for i in range(len(Phi_sol)):
        
        I = translator["dof_to_point"][i]
        new_phi[I] = Phi_sol[i][0]
    
    if saturate_porosities:
        new_phi[new_phi<0.0] = 0.0
        new_phi[new_phi>1.0] = 1.0
    
    
    # Type 4 refers to elements whose all points exist at the surface
    # There are the points associated to that condition
    type4_points = np.unique(ien[surface_classifier[4]])
    ntype4 = len(type4_points)
    
    # Store the id of all the points in this category
    actives = 0
    isolated_points = []
    for point in type4_points:
        
        # Some of these points may have received some treatment 
        # because of other nodes that have the point within their tetrad.
        if point in active_surface_points:
            actives +=1
        else:
            # If it didn't receive any treatment before, call them
            # 'isolated_points'.
            isolated_points += [point]
        
    # Present info through screen
    print("Number of active points: %i/%i"%(actives,ntype4))
    
    loops = 0
    # Check for  adjacent points to each isolated point
    point_appareances = {point:list(np.argwhere(ien==point)[:,0]) for point in isolated_points}
    
    while (len(isolated_points)>0) and loops<max_loops:
    
        # 'point_connections' will be adjacent points to the isolated points where
        # we have some porosity info we can use to determine the local intensity.
        point_connections = {}
        # Disconnected points will have no adjacent known points and have to be treated
        # differently.
        disconnected_points = []
        
        # Each isolated point will undergo this treatment
        for point in isolated_points:
            # The list of nearby points
            connections = list(np.unique(ien[point_appareances[point]]))
            # Remove the point of interest
            connections.remove(point)
            # For every adjacent point, see if they belong in the same category (type4)
            for testpoint in connections.copy(): # Use a dummy array and simple procedure to 
            # determine which points have an adjacent node with known information.
                if testpoint in isolated_points:
                    connections.remove(testpoint)
        
            # If there are no known points with regional info nearby, a different treatment is 
            # required. We'll store them as disconnected_points
            if len(connections) == 0:
                disconnected_points += [point]
                
            point_connections.update({point:connections})
        
        for point in point_connections.keys():
            
            neighbours = point_connections[point]
            if len(neighbours) == 0:
                continue
            
            new_phi[point] = np.average(new_phi[neighbours])
            
            isolated_points.remove(point)
            
        print(" > Pending points: %i"%(len(isolated_points)))
        loops += 1
        
        if loops==max_loops:
            print("Maximum amount of loops reached. Exiting while cycle. Careful with the results.")
            
        if len(isolated_points)==0:
            print("All nodes have been covered. Successfully exiting while cycle.")
        
    return new_phi

# %% Imported from the biomechanical analysis library

def InterpolateBSplines(filenameMesh, filenameCPP, filenameRef, 
                        adjustment = np.zeros((3))):
	'''
	Interpolate data using cubic B-splines.
	filenameResults : .npz file containing nodal information
	filenameCPP	 : file containing control point position from registration process
	filenameRef	 : Image file
	
	Output: Phi (nodal deformation mapping)
	'''	
	
	#Filename Results
	meshfile = io.read(filenameMesh)
	X = meshfile.points+adjustment
	
	#Reference Affine
	img = nib.load(filenameRef)
	affine=img.affine
		
	#Affine 0 (origin in (0,0,0) and positive spacing) 
	affine0=np.zeros((4,4))
	affine0[0:3,0:3]=np.abs(affine[0:3,0:3])
	affine0[3,3]=1.0
	affine0_inv=np.array(np.matrix(affine0)**-1)

	#Real World Coordinates (Affine0) 2 Voxel Coordinates  
	X_VoxCoord_Affine0=np.array(np.matrix(affine0_inv[0:3,0:3])*np.matrix(X).T)
	ones=np.ones((1,X_VoxCoord_Affine0.shape[1]))
	X_VoxCoord_Affine0=np.vstack((X_VoxCoord_Affine0,ones))
	
	# Voxel Coordinates 2 Real World Coordinates (Reference Image)
	X_RWC=np.matrix(affine)*np.matrix(X_VoxCoord_Affine0)
	X_RWC=np.array(X_RWC[0:3,:].T)

	#CCP and Affine of the CPP Image
	imgCPP = nib.load(filenameCPP)
	affineCPP=imgCPP.affine
	dataCPP=np.squeeze(np.array(imgCPP.dataobj))
	
	#CPP Affine 
	A=np.matrix(affineCPP[0:3,0:3])**-1
	b=affineCPP[0:3,3]
	
	#COORDENADAS INDICIALES CPP DE X, FLOOR Y LOCAL
	X_VoxelCoord_CPP=np.array((np.matrix(A)*np.matrix(X_RWC-b).T).T)
	X_VoxelCoord_CPP_Floor=np.floor(X_VoxelCoord_CPP)
	X_VoxelCoord_CPP_Local=X_VoxelCoord_CPP-X_VoxelCoord_CPP_Floor

	
	#FUNCIONES BSPLINE
	def B0(u):
		return (1.-u)**3./6.
	def B1(u):
		return (3.*u**3.-6.*u**2.+4.)/6.
	def B2(u):
		return (-3.*u**3.+3.*u**2.+3.*u+1.)/6.
	def B3(u):
		return u**3/6
	
	defmap=[]
	
	#GRID A CONSIDERAR EN LA INTERPOLACION EN CADA REGION (64 PUNTOS)
	i0=np.arange(0,4)   
	j0=np.arange(0,4)
	k0=np.arange(0,4)
	i0,j0,k0=np.meshgrid(i0,j0,k0)
	i0=i0.reshape((i0.shape[0]*i0.shape[1]*i0.shape[2],))
	j0=j0.reshape((j0.shape[0]*j0.shape[1]*j0.shape[2],))
	k0=k0.reshape((k0.shape[0]*k0.shape[1]*k0.shape[2],))   
		
	for p in np.arange(X_VoxelCoord_CPP.shape[0]):
	   iref=X_VoxelCoord_CPP_Floor[p,0].astype(int)-1
	   jref=X_VoxelCoord_CPP_Floor[p,1].astype(int)-1
	   kref=X_VoxelCoord_CPP_Floor[p,2].astype(int)-1
	   i=i0+iref
	   j=j0+jref
	   k=k0+kref   
	   u=X_VoxelCoord_CPP_Local[p,0]
	   v=X_VoxelCoord_CPP_Local[p,1]
	   w=X_VoxelCoord_CPP_Local[p,2]
	   bu=np.array([B0(u),B1(u),B2(u),B3(u)])
	   bv=np.array([B0(v),B1(v),B2(v),B3(v)])
	   bw=np.array([B0(w),B1(w),B2(w),B3(w)])
	
	   defmap.append([sum(bu[i0]*bv[j0]*bw[k0]*dataCPP[i,j,k,0]),sum(bu[i0]*bv[j0]*bw[k0]*dataCPP[i,j,k,1]),sum(bu[i0]*bv[j0]*bw[k0]*dataCPP[i,j,k,2])])
	   
	PHI_RWC=np.array(defmap)

	#Affine 0 (origin in (0,0,0) and positive spacing) 
	A=np.matrix(affine[0:3,0:3])**-1
	b=affine[0:3,3]
	PHI_VoxelCoord_RefImage=np.array((A*np.matrix(PHI_RWC-b).T))
	PHI=np.matrix(affine0[0:3,0:3])*np.matrix(PHI_VoxelCoord_RefImage)
	PHI=np.array(PHI.T)
	
	return PHI	

# %%
def interpolateIntensities(X, image):
	
	'''
	X: Coordinates associated to the geometry being used,
        it can be the raw 'xyz' (expiratory state) or the 
        'phi' coordinates (interpolated inspiratory state).
    
    image: The path to the image in use, that is the expiratory or
    inspiratory image
	'''
    
	im = nib.load(image)
    
	affine = im.affine
	
	# Check for a non-diagonal affine matrix
	non_null_subdiagonal = False
	for (i,j) in [(0,1),(0,2),(1,2)]:
		if affine[i,j] != 0 or affine[j,i] != 0:
			non_null_subdiagonal = True
	
	if non_null_subdiagonal:
		if affine[0,0] != affine[1,1]:
			af_max = np.max(np.abs([affine[0,0],affine[1,1]]))
			if affine[0,0]<0:
				af_max *= -1
		else:
			af_max = affine[0,0]
		
		for (i,j) in [(0,1),(0,2),(1,2)]:
			affine[i,j] = 0.0
			affine[j,i] = 0.0
		affine[0,0] = af_max
		affine[1,1] = af_max

	data = np.array(im.dataobj)
		
	affine0 = np.zeros((4,4))
	affine0[0:3,0:3] = np.abs(affine[0:3,0:3]) 
	affine0[3,3] = 1.0
	affine0_inv=np.array(np.matrix(affine0)**-1)
	
	X_VoxCoord = np.array(np.matrix(affine0_inv[0:3,0:3])*np.matrix(X).T).T
	X_VoxCoord_Floor = np.floor(X_VoxCoord)
	X_VoxCoord_Local = X_VoxCoord - X_VoxCoord_Floor
		
	i0 = np.arange(0,4)
	j0 = np.arange(0,4)
	k0 = np.arange(0,4)
	i0,j0,k0 = np.meshgrid(i0,j0,k0)
	i0 = i0.reshape((i0.shape[0]*i0.shape[1]*i0.shape[2],))
	j0 = j0.reshape((j0.shape[0]*j0.shape[1]*j0.shape[2],))
	k0 = k0.reshape((k0.shape[0]*k0.shape[1]*k0.shape[2],))
	
	def B0(u):
		return (1.-u)**3./6.
	def B1(u):
		return (3.*u**3.-6.*u**2.+4.)/6.
	def B2(u):
		return (-3.*u**3.+3.*u**2.+3.*u+1.)/6.
	def B3(u):
		return u**3/6
	
	warpedIntensities = []
	
	for p in np.arange(X_VoxCoord.shape[0]):
	
		iref=X_VoxCoord_Floor[p,0].astype(int)-1
		jref=X_VoxCoord_Floor[p,1].astype(int)-1
		kref=X_VoxCoord_Floor[p,2].astype(int)-1
	
		i=i0+iref
		j=j0+jref
		k=k0+kref
		
		u=X_VoxCoord_Local[p,0]
		v=X_VoxCoord_Local[p,1]
		w=X_VoxCoord_Local[p,2]
	
		bu=np.array([B0(u),B1(u),B2(u),B3(u)])
		bv=np.array([B0(v),B1(v),B2(v),B3(v)])
		bw=np.array([B0(w),B1(w),B2(w),B3(w)])

		warpedIntensities.append(sum(bu[i0]*bv[j0]*bw[k0]*data[i,j,k]))

	return np.array(warpedIntensities)

# %% Previously generated functions in IoU_Study

def tetrahedron_volume(pts):
    """
    Compute the volume of a tetrahedron from 4 points in 3D.

    Parameters:
        pts : array-like, shape (4, 3)
            The four 3D vertices of the tetrahedron.

    Returns:
        float
            The volume of the tetrahedron.
    """
    pts = np.asarray(pts)
    if pts.shape != (4, 3):
        raise ValueError("Input must be an array of shape (4, 3)")
    
    a, b, c, d = pts
    volume = np.abs(np.dot(d - a, np.cross(b - a, c - a))) / 6.0
    return volume

def geometric_center(rvolumes, rcellcenters):
    """
    Compute the volume-weighted geometric center of a set of cells.

    Parameters:
        rvolumes      : array-like, shape (Ncells, 1) or (Ncells,)
        rcellcenters  : array-like, shape (Ncells, 3)
        total_rvolume : float or None (optional). If None, it's computed as sum of rvolumes.

    Returns:
        ndarray of shape (3,)
            The volume-weighted geometric center.
    """
    rvolumes = np.asarray(rvolumes).reshape(-1, 1)
    rcellcenters = np.asarray(rcellcenters)

    total_rvolume = np.sum(rvolumes)
    
    weighted_sum = np.sum(rvolumes * rcellcenters, axis=0)
    center = weighted_sum / total_rvolume
    return center

def determine_geometric_center(points,cells,verbose=False):
   
    # Empty list
    volumes = []
    cellcenters = []
    # Presenter
    ncells = len(rcells)
    shower = np.ceil(ncells/10).astype(int)

    # Every cell
    for e,cell in enumerate(cells):
        
        # Determine every cell volume
        volumes += [tetrahedron_volume(points[cell])]
        # Cell center
        cellcenters += [np.mean(points[cell],axis=0)]
        
        if e%shower==0 and verbose:
            print(" > %.0f"%((e/ncells)*100)+"%")
    
    if verbose: print(" > 100%\nDone")

    # Change format
    volumes = np.array(volumes)
    cellcenters = np.array(cellcenters)
    
    # Determination of the geometric center of this mesh
    return geometric_center(volumes,cellcenters)

# %% Assessing image histogram, best comparisons

# Declare paths

# We'll interpolate intensities towards a different mesh
subject = 5
protocol = 'ARDSnet'
key = (subject,protocol)
r_root = "D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/"
r_mesh_path = r_root+"PIG%i/%s/MESH/"%key
r_nifti_path = r_root+"PIG%i/%s/NIFTI/"%key
r_registration_path = r_root+"PIG%i/%s/REGISTRATION/"%key

# Images to be used
exp_ct = r_nifti_path+"Exp.nii.gz"
insp_ct = r_nifti_path+"Insp.nii.gz"
exp_seg = r_nifti_path+"NEW_Mask_Exp.nii.gz"
insp_seg = r_nifti_path+"NEW_Mask_Insp.nii.gz"

# Simulation-associated paths
s_mesh_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/%s/medium/"%key
sim_tetra_mesh = s_mesh_path+"FEniCS/mesh000000.vtu"
in_simulated_ee_surf = s_mesh_path+"FEniCS/boundary_markers000000.vtu"
temp_out_mesh = s_mesh_path+"temp.vtu"

# Source files
in_registered_ee_surf = r_mesh_path+"Surface_Exp.vtk"
reg_npz = np.load(r_mesh_path+"Exp_NEW.npz")

# Nifti files
filenameCPP = r_registration_path+"cpp_9000-000666.nii.gz"
filenameRef = r_nifti_path+"Exp.nii.gz"

# Point towards the deformed simulated mesh result
sim_root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/"
#simulated_mesh = sim_root+"/codes/DeleteMe/MFSIMS/PIG%i/output/post/"%subject
simulated_mesh = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
simulated_mesh += "gravity-tests/birzle-grav/post/"

if subject == 5:
    simulated_mesh += "full_0.750000000000.vtu"
elif subject == 4:
    simulated_mesh += "full_0.760000000000.vtu"
elif subject == 3:
    simulated_mesh += "full_0.890000000000.vtu"
else:
    simulated_mesh = None


# %% Determine the displacement between geometries and determine bounding box

# Retrieve points
rcells = reg_npz['elem']
rpoints = reg_npz['xyz']

# Geometric center of the registrated mesh
reg_center = determine_geometric_center(rpoints,rcells)

# Load FEniCS mesh and determine geometric center
# Retrieve tetra points
smesh = io.read(sim_tetra_mesh)
spoints = smesh.points
scells = smesh.cells_dict['tetra']

# Determine geometric center
sim_center = determine_geometric_center(spoints,scells)

# Displace points
# Corrector; sum to the simulated points
delta = reg_center - sim_center

# Testing unit to visualize point position
if True:
    
    d0=0;d1=2
    rmsk = np.random.rand((rpoints.shape[0]))<0.20
    smsk = np.random.rand((spoints.shape[0]))<0.85
    plt.scatter(spoints[:,d0][smsk]+delta[d0],
                spoints[:,d1][smsk]+delta[d1],alpha=0.2)
    
    plt.scatter(rpoints[:,d0][rmsk],rpoints[:,d1][rmsk],alpha=0.2)  
    
# %% 

# Interpolate intensities into the FE mesh
# Use the simulation mesh
'''

We will interpolate the intensities from the raw images into the simulation mesh
    
OK  1) Access the simulation mesh
OK  2) Displace the simulation mesh into the biomech analysis mesh
OK  3) Interpolate EE intensities 
OK  4) Interpolate EI intensities maybe?
    5) Correct the intensities using the porosity algorithm
    6) Generate the histograms

'''

# The simulation mesh has already been accessed
cs_points = spoints+delta # Displaced points
# scells: The simulated mesh connectivity list

# Determine the displaced coordinates associated to the FE mesh
phi = InterpolateBSplines(sim_tetra_mesh,filenameCPP,filenameRef,
                          adjustment=delta)

# Extract intensities
ee_intensities = interpolateIntensities(cs_points,exp_ct)
ei_intensities = interpolateIntensities(phi,insp_ct)
# %%
# Determine and saturate porosity
ee_porosity = ee_intensities/-1000
ei_porosity = ei_intensities/-1000
ee_porosity[ee_porosity<0.0] = 0.0
ee_porosity[ee_porosity>1.0] = 1.0
ei_porosity[ei_porosity<0.0] = 0.0
ei_porosity[ei_porosity>1.0] = 1.0


new_ee_porosity = correct_porosity(cs_points, scells, ee_porosity)
new_ei_porosity = correct_porosity(cs_points, scells, ei_porosity)

# Generate a visualization object for preliminary evaluation
temp_out= io.Mesh(points=cs_points,cells={'tetra':scells},
         point_data={"ee_intensities":ee_intensities,
                     "ei_intensities":ei_intensities,
                     "ee_porosity":ee_porosity,
                     "ei_porosity":ei_porosity,
                     "new_ee_porosity":new_ee_porosity,
                     "new_ei_porosity":new_ei_porosity,
                     "u":phi-cs_points})

temp_out.write(temp_out_mesh)

# %% Determine mass

M,_,_,_ = Tet4DefGradSmoothingOptiSPARSE(cs_points, phi, scells)

# %% Generate global histogram

if True:
    
    # Analyze
    gh_ee = retrieve_global_histogram(temp_out_mesh,porosity_field='new_ee_porosity')
    gh_ei = retrieve_global_histogram(temp_out_mesh,porosity_field='new_ei_porosity')
    gh_sim = retrieve_global_histogram(simulated_mesh)
    
    handle = {"Exp":{"NAT":gh_ee[0][0],
                     "PAT":gh_ee[0][1],
                     "AT":gh_ee[0][2],
                     "HIT":gh_ee[0][3]},
              "Insp":{"NAT":gh_ei[0][0],
                      "PAT":gh_ei[0][1],
                      "AT":gh_ei[0][2],
                      "HIT":gh_ei[0][3]},
              "Sim":{"NAT":gh_sim[0][0],
                      "PAT":gh_sim[0][1],
                      "AT":gh_sim[0][2],
                      "HIT":gh_sim[0][3]},
              }
    
# %% 
    
    # Generate one-dimensional plot
    states = ["Exp","Insp", "Sim"]
    names = ["Expiration","Inspiration \n(Experimental)","Inspiration \n(Simulation)"]
    figsize = (6,6)
    legend = ["Hyperinflated tissue", "Normally aerated tissue",\
               "Poorly aerated tissue", "Non aerated tissue"]
        
    textsize=14

    # Recover information from the master dictionary.
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=figsize,dpi=300)
        
    for i,phase in enumerate(states):
                            
        hst = handle[phase]
                
        # Tuning parameters for caption location
        ax.bar(i, hst['NAT'], color = "w",edgecolor='k') 
        if hst['NAT']>0.01: ax.text(i,hst['NAT']/2,"%.2f"%hst['NAT'],color="k",ha='center',va='center',size=textsize)
        ax.bar(i, hst['PAT'],bottom=hst['NAT'], color = "k", alpha = 0.25, edgecolor="k")
        ax.text(i,hst['NAT']+hst['PAT']/2,"%.2f"%hst['PAT'],size=textsize,color="w",ha='center',va='center')
        ax.bar(i, hst['AT'],bottom=hst['NAT']+hst['PAT'], color="k", alpha = 0.5, edgecolor="k")
        ax.text(i,hst['NAT']+hst['PAT']+hst['AT']/2,"%.2f"%hst['AT'],size=textsize,color="w",ha='center',va='center')
        ax.bar(i, hst['HIT'],bottom=hst['NAT']+hst['PAT']+hst['AT'], color = "k", edgecolor="k", alpha = 0.999)
        if hst['HIT']>0.05: ax.text(i,hst['NAT']+hst['PAT']+hst['AT']+hst['HIT']/2,"%.2f"%hst['HIT'],size=textsize,color="w",ha='center',va='center')

   # ax.set_xlabel("States",size=18)
    ax.set_ylabel("Fraction [-]",size=18)
    ax.set_yticks(np.linspace(0,1,6))
    ax.set_yticklabels(["%.1f"%lab for lab in np.linspace(0,1,6)],size=16)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(names,size=14)
    
    plt.tight_layout()
        
# %% 
    
    # Generate one-dimensional plot
    states = ["Exp","Sim"]
    names = ["Expiration","Inspiration \n(Simulation)"]
    figsize = (4,6)
    legend = ["Hyperinflated tissue", "Normally aerated tissue",\
               "Poorly aerated tissue", "Non aerated tissue"]

    # Recover information from the master dictionary.
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=figsize,dpi=200)
        
    for i,phase in enumerate(states):
                            
        hst = handle[phase]
                
        # Tuning parameters for caption location
        ax.bar(i, hst['NAT'], color = "w",edgecolor='k') 
        if hst['NAT']>0.01: ax.text(i,hst['NAT']/2,"%.2f"%hst['NAT'],color="k",ha='center',va='center')
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
    ax.set_xticklabels(names,size=14)
    
    plt.tight_layout()
    
# %% Determine regional histograms

directions = ['BA','VD']
states = ["Exp","Insp","Sim"]
package = {"Exp":(temp_out_mesh,'new_ee_porosity',False), # mesh_path, field, deformed_state
           "Insp":(temp_out_mesh,'new_ei_porosity',True),
           "Sim":(simulated_mesh,'Eulerian Porosity',True)}

manager = {}
nrois = 10
for direction in directions:
    
    for state in states:
        
        path,porosity_field,deformed_state =  package[state]
        
        page = retrieve_regional_histogram(path,direction,
                                            weights=None, verbose=False,
                                            deformed_state=deformed_state,
                                            porosity_field=porosity_field,
                                            nrois=nrois)
        
        manager.update({(state,direction):page})
        
# %%

directions = ["VD"]
states = ["Exp","Insp","Sim"]
textsize = 12
threshold = 0.05

for state in states:
    for direction in directions:

        fig, ax = plt.subplots(figsize=(10,6),dpi=200)
        width = 0.8
        
        gen_binner = manager[(state,direction)][0]
        e=0
        dx=0
        
        for roi in range(nrois):
            
            binner = gen_binner[roi]
            
            ax.bar(roi+(e-0.5)*dx, binner[0],width=width, color = "w",edgecolor='k') 
            ax.bar(roi+(e-0.5)*dx, binner[1],width=width,bottom=binner[0], color = "silver", alpha = 1.0, edgecolor="k")
            ax.bar(roi+(e-0.5)*dx, binner[2],width=width,bottom=binner[0]+binner[1], color="gray", alpha = 1.0, edgecolor="k")
            ax.bar(roi+(e-0.5)*dx, binner[3],width=width,bottom=binner[0]+binner[1]+binner[2], color = "k", edgecolor="k", alpha = 0.999)
            # Include text
            if binner[0]>threshold and False:
                ax.text(roi+(e-0.5)*dx, binner[0]*0.5, "%.2f"%(binner[0]),
                    ha='center',va='center',size=textsize,color='k')
            if binner[1]>threshold and False:
                ax.text(roi+(e-0.5)*dx, binner[1]*0.5+binner[0], "%.2f"%(binner[1]),
                    ha='center',va='center',size=textsize,color='k')
            
            if binner[2]>threshold and False:
                ax.text(roi+(e-0.5)*dx, binner[2]*0.5+binner[1]+binner[0], "%.2f"%(binner[2]),
                    ha='center',va='center',size=textsize,color='w')
            
            if binner[3]>threshold and False:
                ax.text(roi+(e-0.5)*dx, binner[3]*0.5+binner[2]+binner[1]+binner[0], "%.2f"%(binner[3]),
                    ha='center',va='center',color="w",size=textsize)
        
        ax.set_title("Regional histogram - %s - %s"%(state, direction))
        ax.set_ylabel("Fraction (-)", size=textsize+4)
        ax.set_xticks(range(nrois))
        ax.set_xticklabels(np.arange(1,nrois+1), size=textsize+4)
        yticks =ax.get_yticklabels()
        ax.set_yticklabels(yticks,size=textsize+4)
        ax.text(-1,-0.10,direction[0], weight='bold',size=textsize+8)
        ax.text(10,-0.10,direction[1], weight='bold',size=textsize+8)
        ax.set_xlabel("ROI#", size=textsize+4)
        
        
# %%

temp_out_mesh =  'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/medium/temp.vtu'%subject
exp_mesh = io.read(temp_out_mesh)
sim_mesh = io.read(simulated_mesh)

sim_ei_porosity = sim_mesh.point_data['Eulerian Porosity']
sim_dgf = sim_mesh.point_data['Delta Porosity']

exp_ee_porosity = exp_mesh.point_data['new_ee_porosity']
exp_ei_porosity = exp_mesh.point_data['new_ei_porosity']

# %%
lims = (0,1)
fig,axes = plt.subplots(ncols=2,figsize=(8,4),dpi=200)
ax=axes[0]
ax.scatter(exp_ee_porosity,exp_ei_porosity, alpha=0.25, s=12)
ax.set_title("Experimental data")
ax.set_ylabel("End-inspiratory porosity")

ax=axes[1]
ax.scatter(exp_ee_porosity,sim_ei_porosity, alpha=0.25, s=12)
ax.set_title("Simulated data")
ax.set_yticks([])

for ax in axes:
    ax.plot(np.linspace(0,1),np.linspace(0,1),ls='--',color="k",alpha=0.85,label="Identity line", lw=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("End-expiratory porosity")
    
plt.tight_layout()

# %%

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(exp_ei_porosity, sim_ei_porosity)


fig, ax = plt.subplots(figsize=(4,4),dpi=200)

ax.scatter(exp_ei_porosity, sim_ei_porosity,alpha=0.25,s=12)
ax.set_ylim(lims)
ax.plot(np.linspace(0,1),np.linspace(0,1),ls='--',color="k",alpha=0.85,label="Identity line", lw=1.5)

x_vals = np.linspace(min(exp_ei_porosity), max(exp_ei_porosity), 100)
y_vals = intercept + slope * x_vals
ax.plot(x_vals, y_vals, color='red', label=f"Fit: y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}", linewidth=1.5,alpha=0.85)


ax.set_ylabel("[EI] Simulated porosity")
ax.set_xlabel("[EI] Experimental porosity")

ax.legend()

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define groups, intervals, and colors
labels = [
    r"NAT | Non-aerated tissue $\phi \in [0.0, 0.1)$",
    r"PAT | Poorly-aerated tissue $\phi \in [0.1, 0.5)$",
    r"AT  | Normally-aerated tissue $\phi \in [0.5, 0.9)$",
    r"HIT | Hyperinflated tissue $\phi \in [0.9, 1.0]$"
]
colors = ["w", "silver", "gray", "k"]

# Reverse so NAT is last, HIT is first (top to bottom in legend)
colors.reverse()
labels.reverse()

# Create legend patches
patches = []
for i, (c, l) in enumerate(zip(colors, labels)):
    if l.startswith("NAT"):  # ensure NAT box (white) gets black edge
        patches.append(mpatches.Patch(facecolor=c, edgecolor="black", label=l))
    else:
        patches.append(mpatches.Patch(facecolor=c, edgecolor="black", label=l))  # <- you can drop edgecolor here if you only want NAT outlined

# Make a blank figure just for the legend
fig, ax = plt.subplots(dpi=300)  # <-- set dpi here
ax.axis("off")

legend = ax.legend(
    handles=patches,
    loc="center",
    frameon=True,
    framealpha=1,
    edgecolor="black",
    fontsize=14,
    title="Porosity-based aeration histogram",
    title_fontsize=18
)

# Show on screen
plt.show()

# Save high-resolution file (uncomment if needed)
# fig.savefig("aeration_legend.png", dpi=300, bbox_inches="tight")

