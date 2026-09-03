# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:56:48 2025

@author: angus
"""

# %% import libraries from biomechanical analysis

import numpy as np
from scipy.sparse import coo_matrix
from numpy.core.umath_tests import matrix_multiply 
import meshio as io
import matplotlib.pyplot as plt
from collections import defaultdict
import nibabel as nib

# %% Biomechanical analysis

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


def ComputeNodalDefGradTensor(Mlumped,P):
	
	"""
	This subroutine computes the nodal values of deformation gradient
	by solving the Mlumped*u=P system, where
	Mlumped	: Diagonal mass matrix (in vector form)
	P		  : weighted deformation-gradient vector

	Output:
	Fnodal	 : [..., F_nodoi, ...]
	"""
	Fnodal = []
	append=Fnodal.append
	nodenumber = 0
	for m, fvec in zip(Mlumped, P[:]) :
		Fnode = np.matrix(np.zeros((3,3)))
		assert (m > 0), "nodal mass is zero in node " + str(nodenumber)
		fvec/=m
		for i in np.arange(3) :
			for J in np.arange(3) :
				Fnode[i,J] = fvec[3*i+J]
		append(Fnode)
		nodenumber += 1
	return Fnodal
	
	
def FemAnalysisSPARSE(LM,xyz,phi):
	'''
	Estimate deformation measurements and save information in a VTU file.
	
	Inputs:
		filenameResults  : .npz file containing nodal information
	Outputs:
		filenamefields   : string with file name (including .vtu extension)
	'''	

	Mlumped, P,ve,ve0 = Tet4DefGradSmoothingOptiSPARSE(xyz, phi, LM)
	Fnodal = ComputeNodalDefGradTensor(Mlumped,P) 


	Fnodal_T=np.transpose(Fnodal,(0,2,1))
	C = matrix_multiply(Fnodal_T,Fnodal)

	lam2,N = np.linalg.eig(C[:])	
	for n in np.arange(lam2.shape[0]):
		idx = np.argsort(lam2[n])[::-1]
		lam2[n,:] = lam2[n,idx]
		N[n] = N[n,:,idx].T
		
	B = matrix_multiply(Fnodal,Fnodal_T)
	
	aux,nvec = np.linalg.eig(B[:])
	for n in np.arange(aux.shape[0]):
		idx = np.argsort(aux[n])[::-1]
		aux[n,:] = aux[n,idx]
		nvec[n] = nvec[n,:,idx].T
	
	J = np.linalg.det(Fnodal[:])

	I1 = np.sum(lam2,axis=(1))
	I2 = lam2[:,0]*lam2[:,1]+lam2[:,1]*lam2[:,2]+lam2[:,2]*lam2[:,0]
	I3 = J**2
   
	# Setup scalar and vector fields for export
	sf = {"VolStrain":((J-1)*100),"I1":I1,"I2":I2,"I3":I3}
	vf = {"DefMap":phi,"DispField":(phi-xyz),}
		
	return sf, vf

# Retrieve the registration associated results; apply biomechanical analysis into
# these results, and then save it somewhere useful.

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
    ncells = len(cells)
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

#  Correction schemes for intensities and porosities

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

def reorient_if_needed(points, tets):

    # Compute signed volumes
    qme = points[tets]  # (n_elem, 4, 3)
    ones = np.ones((tets.shape[0], 4))
    T = np.stack([ones, qme[:,:,0], qme[:,:,1], qme[:,:,2]], axis=-1)  # (n_elem, 4, 4)
    volumes = np.linalg.det(T) / 6.0
    
    cells = tets.copy()
    
    counter = 0
    for i in range(len(tets)):
        if volumes[i] < 0:
            counter += 1
            cells[i, [0, 1]] = cells[i, [1, 0]]  # Swap nodes 0 and 1
            
    print("%i out of %i cells were flipped"%(counter,len(cells)))
    return cells
    

def IsoVolumetricSegmentation(direction, Mvec, xyz, nROI):

	V = np.sum(Mvec)
	dv = V/nROI
	d = np.ravel(xyz*direction)
	sum_mass = np.cumsum(Mvec[np.argsort(d)])
	lim_index = np.array([np.abs(sum_mass-i).argmin() for i in
						  np.linspace(dv, V, nROI)])
	id_roi = np.ones(len(Mvec))*(nROI-1)
	for i, j in enumerate(lim_index):
		if i == 0:
			id_roi[:j] = np.ones(j)*i
		else:
			id_roi[lim_index[i-1]:j] = np.ones(j-lim_index[i-1])*i

	id_roi_output = [None]*len(id_roi)
	for i, j in zip(id_roi, np.argsort(d)):
		id_roi_output[j] = int(i)

	return [sum_mass, np.sort(d)], np.array(id_roi_output)


# %%

# We'll interpolate intensities towards a different mesh
subject = 3
protocol = 'ARDSnet'
key = (subject,protocol)
r_root = "D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/"
mesh_quality = 'medium-fine'


r_mesh_path = r_root+"PIG%i/%s/MESH/"%key
r_nifti_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/%s/NIFTI/"%key
r_registration_path = r_root+"PIG%i/%s/REGISTRATION/"%key

# Images to be used
exp_ct = r_nifti_path+"Exp_median.nii.gz"
insp_ct = r_nifti_path+"Insp_median.nii.gz"
exp_seg = r_nifti_path+"NEW_Mask_Exp.nii.gz"
insp_seg = r_nifti_path+"NEW_Mask_Insp.nii.gz"

# Simulation-associated paths
s_mesh_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/%s/"%key+"%s/"%mesh_quality
sim_tetra_mesh = s_mesh_path+"FEniCS/mesh000000.vtu"
in_simulated_ee_surf = s_mesh_path+"FEniCS/boundary_markers000000.vtu"

# Source files
in_registered_ee_surf = r_mesh_path+"Surface_Exp.vtk"
reg_npz = np.load(r_mesh_path+"Exp_NEW.npz")

# Nifti files
filenameCPP = r_registration_path+"cpp_9000-000666.nii.gz"
filenameRef = r_nifti_path+"Exp.nii.gz"

# Point towards the deformed simulated mesh result
sim_root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/"
sim_root += 'ards-lung-simulator/PIG%i-mf-per-3/'%subject
simulated_mesh = sim_root+"/post/"

if subject == 5:
    simulated_mesh += "full_0.750000000000.vtu"
elif subject == 4:
    simulated_mesh += "full_0.760000000000.vtu"
elif subject == 3:
    simulated_mesh += "full_0.890000000000.vtu"
elif subject == 6:
    simulated_mesh += "full_0.805000000000.vtu"
elif subject == 2:
    simulated_mesh += "full_0.995000000000.vtu"

else:
    simulated_mesh = None
    
# Force end inspiratory to be bounded between 1 and 0
saturate_simulation_ei = False
    
# Out meshes. To be used in figures and animations
out_video_registration = s_mesh_path+"reg_anim_ready.vtu"
out_video_simulation = s_mesh_path+"sim_anim_ready.vtu"

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

#observer = []
#for F in np.linspace(6,10,17):
if True:
    if subject == 2:
        F = 27.5
        fix = (0.0, 0.0, F) # 28
        ei_intensities = interpolateIntensities(phi+fix,insp_ct)
    elif subject == 3:
        
        F = 8.25
        fix = (0.0, 0.0, F) # 28
        ei_intensities = interpolateIntensities(phi+fix,insp_ct)
      #  observer +=[F,np.linalg.norm(ee_intensities-ei_intensities)]

    else:    
        ei_intensities = interpolateIntensities(phi,insp_ct)

# 
# Determine and saturate porosity
ee_porosity = ee_intensities/-1000
ei_porosity = ei_intensities/-1000
ee_porosity[ee_porosity<0.0] = 0.0
ee_porosity[ee_porosity>1.0] = 1.0

if saturate_simulation_ei:
    ei_porosity[ei_porosity<0.0] = 0.0
    ei_porosity[ei_porosity>1.0] = 1.0

# %%
# Correct the interpolated porosity
new_ee_porosity = correct_porosity(cs_points, scells, ee_porosity)
new_ei_porosity = correct_porosity(cs_points, scells, ei_porosity, saturate_porosities=saturate_simulation_ei)

# Determine the porosity change between EE and EI
delta_porosity = new_ei_porosity-new_ee_porosity

# Manage finite element mesh
xyz = cs_points
phi = phi
LM = scells

# Flip the finite element nodes
LM_flipped = reorient_if_needed(xyz, LM)

# Determine the biomechanical analysis
M,_,_,_ = Tet4DefGradSmoothingOptiSPARSE(xyz,phi, LM_flipped)
sf, vf = FemAnalysisSPARSE(LM_flipped,xyz,phi)


new_jacobian = correct_porosity(cs_points, scells, sf['VolStrain']/100+1,saturate_porosities=False)


# Generate a visualization object of the registration mesh
registration_out = io.Mesh(points=cs_points,cells={'tetra':scells},
         point_data={"ee_intensities":ee_intensities,
                     "ei_intensities":ei_intensities,
                     "original_ee_porosity":ee_porosity,
                     "original_ei_porosity":ei_porosity,
                     "End-Expiratory Porosity":new_ee_porosity,
                     "End-Inspiratory Porosity":new_ei_porosity,
                     "Delta Porosity":delta_porosity,
                     "u":phi-cs_points,
                     'VolStrain':sf['VolStrain'],
                     'Raw Jacobian':sf['VolStrain']/100+1,
                     'Jacobian':new_jacobian,
                     'DispField':vf['DispField'],
                     'DefMap':vf['DefMap'],
                     'Mass':M})

registration_out.write(out_video_registration)
reg_center = np.average(registration_out.points,axis=0)

# %%

# Load the simulated mesh
sim_mesh = io.read(simulated_mesh)
# Read the points assicuated
sim_points = sim_mesh.points
# Determine some representative point
sim_center = np.average(sim_points,axis=0)
# "Normalize"
sim_points -= sim_center
sim_points += reg_center

pd = sim_mesh.point_data

point_data = {'DispField':pd['u'],
              'DefMap':pd['u']+sim_points,
              'Jacobian':pd['Jacobian'],
              'Jacobian Forward':pd['Jacobian Forward'],
              'VolStrain':(pd['Jacobian Forward']-1)*100,
              'End-Inspiratory Porosity':pd['Eulerian Porosity'],
              'End-Expiratory Porosity':(pd['Eulerian Porosity']-pd['Delta Porosity']),
              'Delta Porosity':pd['Delta Porosity'],
              'Mass':M,
              'Pressure (cmH2O)':pd['Pressure (cmH2O)'],
              'Q':pd['QQint'],
              
              }

sim_out = io.Mesh(points=sim_points, cells=sim_mesh.cells_dict, point_data=point_data)
sim_out.write(out_video_simulation)

# %% Determine unidimensional analysis for the fields of interes

import statsmodels.stats.weightstats as sw

directions = {'BA':np.mat([0.,0.,1.]).T,
              'VD':np.mat([0.,1.,0.]).T,}

direction = 'VD'
nrois = 10

_, ids = IsoVolumetricSegmentation(directions[direction], M, sim_points, nrois)


data_manager = {kind:{field:{'mean':[],'std':[]} for field in ['VS','DGF']} for kind in ['sim','reg']}

for mesh,kind in zip([sim_out, registration_out],['sim','reg']):

    vs = mesh.point_data['VolStrain']
    dgf = mesh.point_data['Delta Porosity']
    mass = mesh.point_data['Mass']
    
    for i in range(nrois):
        
        mask = ids==i
        local_vs = vs[mask]
        local_dgf = dgf[mask]
        local_mass = mass[mask]
        
        vs_stat = sw.DescrStatsW(data=local_vs,weights=local_mass)
        dgf_stat = sw.DescrStatsW(data=local_dgf,weights=local_mass)
        
        data_manager[kind]['VS']['mean'] += [vs_stat.mean]
        data_manager[kind]['VS']['std'] += [vs_stat.std]
        data_manager[kind]['DGF']['mean'] += [dgf_stat.mean]
        data_manager[kind]['DGF']['std'] += [dgf_stat.std]
        
# % visualize scatter plot

if False:
    
    fig, ax = plt.subplots(dpi=150, figsize=(5,5))
    
    if direction == 'BA':
        d1 = 0; d2 = 2
    elif direction == 'VD':
        d1 = 1; d2 = 2    
    
    # pick a colormap
    cmap = plt.cm.viridis  # you can try "plasma", "turbo", "tab10", etc.
    
    for idx in range(nrois):
        mask = ids == idx
        color = cmap(idx / (nrois - 1))  # normalized between 0 and 1
        ax.scatter(sim_points[:, d1][mask],
                   sim_points[:, d2][mask],
                   alpha=0.5,
                   s=5,
                   color=color,
                   label=f"ROI {idx}")
    
    ax.legend(title="ROI", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

# %% Generate a visualization using unidimensional plot

colors = {'reg':'tab:orange','sim':'tab:cyan'}
renamer = {'sim':'Simulation',
           'reg':'Experiment'}

fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(5,5),dpi=300)

xdata = np.arange(nrois)+1
for ax, field in zip(axes,['VS','DGF']):
    
    for kind in ['sim','reg']:
        mean = np.array(data_manager[kind][field]['mean'])
        std = np.array(data_manager[kind][field]['std'])
        ax.plot(xdata, mean,color=colors[kind], label=renamer[kind])
        ax.fill_between(xdata,mean-std,mean+std,color=colors[kind],alpha=0.2)
    
axes[0].set_xticks([])
axes[1].set_xticks(xdata)
axes[1].set_xlabel('ROI')
axes[0].set_ylabel('Volumetric strain (%)', size=12)
axes[1].set_ylabel('Delta Gas Fraction (-)', size=12)

    
_,y1 = axes[0].get_ylim()
axes[0].set_ylim((-10.0,y1))
   
    

if subject == 5:
    yticks = [-0.10, 0.0, 0.10, 0.20]
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(["%.1f"%yt for yt in yticks],size=12)
    
    ylabs = axes[0].get_yticklabels()
    axes[0].set_yticklabels(ylabs,size=12)

    xlabs = axes[1].get_xticklabels()
    axes[1].set_xticklabels(xlabs,size=12)

    axes[1].legend(loc='upper left')
    axes[0].legend(loc='lower right')


x0,x1 = axes[1].get_xlim(); dx = x1-x0
y0,y1 = axes[1].get_ylim(); dy = y1-y0

axes[1].text(x0-0.025*dx,y0-0.12*dy,direction[0],weight='bold')
axes[1].text(x1+0.025*dx,y0-0.12*dy,direction[1],weight='bold')

fig.tight_layout()

# %% Volumetric Strain

colors = {'reg':'tab:orange','sim':'tab:cyan'}
renamer = {'sim':'Simulation',
           'reg':'Experimental'}

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,3),dpi=300)

xdata = np.arange(nrois)+1
field = "VS"
for kind in ['sim','reg']:
    mean = np.array(data_manager[kind][field]['mean'])
    std = np.array(data_manager[kind][field]['std'])
    ax.plot(xdata, mean,color=colors[kind], label=renamer[kind])
    ax.fill_between(xdata,mean-std,mean+std,color=colors[kind],alpha=0.2)
    
ax.set_xticks(xdata)
ax.set_xlabel('ROI')
ax.set_ylabel('Volumetric strain (%)')

    
_,y1 = ax.get_ylim()
ax.set_ylim((-10.0,y1))


ax.legend()    
    

x0,x1 = ax.get_xlim(); dx = x1-x0
y0,y1 = ax.get_ylim(); dy = y1-y0

ax.text(x0-0.025*dx,y0-0.12*dy,direction[0],weight='bold')
ax.text(x1+0.025*dx,y0-0.12*dy,direction[1],weight='bold')

fig.tight_layout()


# %% Delta porosity

colors = {'reg':'tab:orange','sim':'tab:cyan'}
renamer = {'sim':'Simulation',
           'reg':'Experimental'}

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,3),dpi=300)

xdata = np.arange(nrois)+1
field = "DGF"
for kind in ['sim','reg']:
    mean = np.array(data_manager[kind][field]['mean'])
    std = np.array(data_manager[kind][field]['std'])
    ax.plot(xdata, mean,color=colors[kind], label=renamer[kind])
    ax.fill_between(xdata,mean-std,mean+std,color=colors[kind],alpha=0.2)
    
ax.set_xticks(xdata)
ax.set_xlabel('ROI')
ax.set_ylabel('Porosity change (-)')
    

x0,x1 = ax.get_xlim(); dx = x1-x0
y0,y1 = ax.get_ylim(); 

if subject == 5:
    y1 = 0.25
elif subject == 4:
    y1 = 0.15; 
    y0 = -0.05
    ax.set_yticks([-0.05, 0.0, 0.05, 0.10, 0.150])

dy = y1-y0
ax.set_ylim((y0,y1))

ax.text(x0-0.025*dx,y0-0.12*dy,direction[0],weight='bold')
ax.text(x1+0.025*dx,y0-0.12*dy,direction[1],weight='bold')
ax.legend(loc='upper left')    

fig.tight_layout()


# %% Registration-associated delta volume and global volumetric strian

# Determine volumes associated to the meshes
ee_vols = []
ei_vols = []

# For each cell determine the tetrahedron volume
for cell in scells:
    ee_vols += [tetrahedron_volume(cs_points[cell])]
    ei_vols += [tetrahedron_volume(phi[cell])]

# Determine the volume associated to each state
ee_vol = np.sum(ee_vols)*1e-3
ei_vol = np.sum(ei_vols)*1e-3

# Determine delta volume and global volumetric strain
reg_dv = (ei_vol - ee_vol)
reg_gvs = reg_dv/ee_vol*100

# %% Image-based delta volume and global volumetric strain

# Load images and determine voxel size
ee_img = nib.load(r_nifti_path+"NEW_Mask_Exp.nii.gz")
ei_img = nib.load(r_nifti_path+"NEW_Mask_Insp.nii.gz")
ee_voxel = np.prod([np.abs(ee_img.affine[i,i]) for i in range(3)])
ei_voxel = np.prod([np.abs(ei_img.affine[i,i]) for i in range(3)])

# Determine segmentation volume
nvox_ee = np.count_nonzero(ee_img.dataobj)
nvox_ei = np.count_nonzero(ei_img.dataobj)
ee_imgvol = nvox_ee*ee_voxel*1e-3
ei_imgvol = nvox_ei*ei_voxel*1e-3

# Determine delta volume and global volumetric strain
img_dv = (ei_imgvol-ee_imgvol)
img_gvs = img_dv/ee_imgvol*100

# %% Determine the simulated mesh global volumetric strain and delta volume

# Load the simulated mesh
sim_mesh = io.read(simulated_mesh)

# Determine the deformed-simulated mesh
sim_phi = sim_mesh.point_data['u']+sim_mesh.points
sim_vols = []

# Cell by cell, determine the deformed tetrahedron volume
for cell in scells:
    sim_vols += [tetrahedron_volume(sim_phi[cell])]

# Determine the volume for the end-inspiratory mesh
sim_vol = np.sum(sim_vols)*1e-3

# Determine delta volume and global volumetric strain
sim_dv = (sim_vol - ee_vol)
sim_gvs = sim_dv/ee_vol*100