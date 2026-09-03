# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 15:52:01 2025

@author: angus
"""

from scipy.stats import qmc
from scipy.optimize import curve_fit
from skopt.space import Real
from skopt import Optimizer
from multiprocessing import Pool, cpu_count, Manager, Lock
import time
import numpy as np
import pyvista as pv
import meshio as io
import nibabel as nib
import trimesh
import embreex
from collections import Counter
from trimesh.ray.ray_pyembree import RayMeshIntersector
import os
import matplotlib.pyplot as plt

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

def determine_geometric_center(tetmesh_path,verbose=False):
    
    fmt = tetmesh_path.split(".")[-1]
    if fmt in ["vtu","vtk","vti"]:
        # Load mesh
        mesh = io.read(tetmesh_path)
        cells = mesh.cells_dict['tetra']
        points = mesh.points
    elif fmt in ["npy","npz"]:
        npz = np.load(tetmesh_path)
        points = npz['xyz']
        cells = npz['elem']
    else:
        print("Unknown file format %s")
   
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


def hybrid_voting_raycast(surf_ply, points, directions=None, vote_threshold=2):
    """
    Perform raycasting along multiple directions and combine results by voting.

    Parameters:
        surf_ply       : str
            Path to the surface mesh (.ply file).
        points         : ndarray, shape (N, 3)
            Points cloud to test.
        directions     : list of ndarray
            List of direction vectors to cast rays.
            Defaults to [(1,0,0), (0,1,0), (0,0,1)] if None.
        vote_threshold : int
            Minimum number of directions voting "inside" to keep a point.

    Returns:
        mask_combined  : ndarray of bools, shape (N,)
            Mask of points classified as inside by voting.
    """
    if directions is None:
        directions = [np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])]
    
    mesh = trimesh.load(surf_ply)
    intersector = RayMeshIntersector(mesh)

    votes = np.zeros(len(points), dtype=int)

    for direction in directions:
        locs = intersector.intersects_location(ray_origins=points,
                                               ray_directions=np.repeat([direction], len(points), axis=0),
                                               multiple_hits=True)
        hits = Counter(locs[1])
        mask_dir = np.zeros(len(points), dtype=bool)
        for i in range(len(points)):
            if hits[i] % 2 == 1:
                mask_dir[i] = True
        votes += mask_dir.astype(int)

    mask_combined = votes >= vote_threshold
    return mask_combined


def generate_registration_ply(calibration_config, delta = np.array([0.,0.,0.])):
   
    """
    Convert a .vtu file containing displacement data to a .ply surface mesh.

    Parameters
    ----------
    original_tetra_geometry : str
        Path to the input VTU file.
    out_ply : str
        Path where the resulting PLY file should be saved.

    Returns
    -------
    bool
        True if the PLY file was successfully written, False otherwise.
    """
    
    
    registration_lq_tetra_geometry = calibration_config['paths']['path_to_mesh']+"mesh000000.vtu"
    registration_tri_geometry = calibration_config['paths']['registration_tri']
    nifti_cpp = calibration_config['paths']['nifti_cpp']
    nifti_exp = calibration_config['paths']['nifti_exp']
    
    # Remove old file if it exists
    if os.path.isfile(registration_tri_geometry):
        os.remove(registration_tri_geometry)

    # Proceed only if the target VTU file exists
    if not os.path.isfile(registration_lq_tetra_geometry):
        print("Tetra geometry not found")
        return False
    
    # Generate a temporary vtu geometry if an npz is received
    if registration_lq_tetra_geometry[-3:] == "npz":
        vtu_tetrahedral_name = registration_lq_tetra_geometry[:-3]+"vtu"
        npz = np.load(registration_lq_tetra_geometry)
        points = npz['xyz']
        ien = npz['elem']
        u = npz['DefMap']
        mesh = io.Mesh(points, cells={'tetra':ien},point_data={'u':u})
        mesh.write(vtu_tetrahedral_name)
        
    elif not registration_lq_tetra_geometry.split(".")[-1] in ["vtu","vtk"]:
        print(" > Problem at 'generate_ply' function")
        print("'%s'"%registration_lq_tetra_geometry.split(".")[-1])
        print("'%s'"%registration_lq_tetra_geometry[:-3])
        return False 
    
    else:
        vtu_tetrahedral_name = registration_lq_tetra_geometry
    
    # Read the VTU mesh
    print("Working on the vtu tetrahedral mesh: \n > %s"%vtu_tetrahedral_name)
    mesh = pv.read(vtu_tetrahedral_name)

    # Extract surface mesh (converts volume mesh to triangle surface mesh)
    surf = mesh.extract_surface()

    # Get deformed surface: points + displacement field
    points = surf.points
    
    phi = InterpolateBSplines(points, nifti_cpp, nifti_exp,
                        adjustment=delta)
    
    # Extract triangle connectivity (IEN)
    ien = [cell.point_ids for cell in surf.cell]

    # Write PLY using meshio
    
    
   # ee_ply = io.Mesh(points, cells={"triangle": ien})
   # ee_ply.write(registration_tri_geometry[:-3]+"_exp.ply")
    
    ei_ply = io.Mesh(phi, cells={"triangle": ien})
    ei_ply.write(registration_tri_geometry)

    return os.path.isfile(registration_tri_geometry)


def generate_simulation_ply(simulation_vtu_geometry, simulation_ply):
   
    """
    Convert a .vtu file containing displacement data to a .ply surface mesh.

    Parameters
    ----------
    original_tetra_geometry : str
        Path to the input VTU file.
    out_ply : str
        Path where the resulting PLY file should be saved.

    Returns
    -------
    bool
        True if the PLY file was successfully written, False otherwise.
    """
    

    
    # Remove old file if it exists
    if os.path.isfile(simulation_ply):
        os.remove(simulation_ply)
        print("Simulation ply file found and removed.")
    else:
        print("Simulation ply file not found.")
    print("Proceeding with the simulation ply generation")

    # Proceed only if the target VTU file exists
    if not os.path.isfile(simulation_vtu_geometry):
        print("Tetra geometry not found")
        return False
        
    # Read the VTU mesh
    mesh = pv.read(simulation_vtu_geometry)

    # Extract surface mesh (converts volume mesh to triangle surface mesh)
    surf = mesh.extract_surface()

    # Get deformed surface: points + displacement field
    points = surf.points
    disp_field = surf.get_array("u")  # Assumes "u" contains displacement vectors
    phi = points + disp_field  # Final deformed positions    
    
    # Extract triangle connectivity (IEN)
    ien = [cell.point_ids for cell in surf.cell]

    # Write PLY using meshio
    ei_ply = io.Mesh(phi, cells={"triangle": ien})
    ei_ply.write(simulation_ply)
    
    if os.path.isfile(simulation_ply):
        print("Simulation ply generated successfully at %s"%simulation_ply)
        return True
    else:
        print("Failed to generate simulation ply")
        print("%s not found"%simulation_ply)
        return False

def determine_bbox(mesh_path1, mesh_path2, f=0.025):
    """Compute bounding box enclosing both meshes with expansion factor f."""
    m0, m1 = io.read(mesh_path1).points, io.read(mesh_path2).points
    min_corner = np.minimum(m0.min(axis=0), m1.min(axis=0))
    max_corner = np.maximum(m0.max(axis=0), m1.max(axis=0))
    delta = max_corner - min_corner
    return [(min_corner[i] - f * delta[i], max_corner[i] + f * delta[i]) for i in range(3)]

def compute_iou(mask_a, mask_b):
    """Compute intersection-over-union (IoU) and component counts."""
    union = np.logical_or(mask_a, mask_b)
    inter = np.logical_and(mask_a, mask_b)
    return {
        'sim_dots': np.count_nonzero(mask_a),
        'reg_dots': np.count_nonzero(mask_b),
        'inter_dots': np.count_nonzero(inter),
        'union_dots': np.count_nonzero(union),
        'only_sim': np.logical_and(mask_a, np.logical_not(inter)),
        'only_reg': np.logical_and(mask_b, np.logical_not(inter)),
        'iou': np.count_nonzero(inter) / np.count_nonzero(union)
    }

def generate_uniform_points(bbox, rho):
    """Generate 3D point cloud within bbox based on density rho."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    dx = (1 / rho)**(1/3)
    nx, ny, nz = max(1, int((xmax - xmin)/dx)), max(1, int((ymax - ymin)/dx)), max(1, int((zmax - zmin)/dx))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel())), (nx, ny, nz)

def determine_shape_misfit(it_path, wave_config, calibration_config, delta = np.array([0.,0.,0.])):
    
    # Retrieve time-related configuration
    Tsyr = wave_config['Tsyr']
    Tpausa = wave_config['Tpausa']
    Tinsp = Tsyr+Tpausa
    
    # Retrieve comparison geometry
    ref_ply_path = calibration_config['paths']['registration_tri']
    
    # Compose path
    sim_vtu_path = it_path+"/VTK/Displacement_%.12f.vtu"%Tinsp
    sim_ply_path = it_path+"/Simulation_Insp.ply"
    success_ply = generate_simulation_ply(sim_vtu_path, sim_ply_path)
    
    print("Determining misfit")
    print(" > Reference ply path: %s"%ref_ply_path)
    print(" > Simulation vtu path: %s"%sim_vtu_path)
    print(" > Simulation ply path: %s"%sim_ply_path)
    
    # Evaluate generated geometry
    if not success_ply: 
        print(" > Failed to generate 'ply' geometry\n")
        return None
    else: 
        print(" > Success generating 'ply' geometry\n")
    
    # Determine bbox
    bbox = determine_bbox(ref_ply_path, sim_ply_path)
    
    # Generate cloud
    cloud, shape = generate_uniform_points(bbox, calibration_config['iou_rho'])
    
    # Generate masks
    sim_mask = hybrid_voting_raycast(sim_ply_path, cloud)
    ref_mask = hybrid_voting_raycast(ref_ply_path, cloud)
    
    results = compute_iou(sim_mask, ref_mask)
    print(" +++ Computed IOU value: %.4f +++\n"%results['iou'])
    
    return 1 - results['iou']

def InterpolateBSplines(X, filenameCPP, filenameRef, 
                        adjustment = np.zeros((3)),
                        correctAffine_ref = False, 
                        correctAffine_cpp = False):
	'''
	Interpolate data using cubic B-splines.
	filenameResults : .npz file containing nodal information
	filenameCPP	 : file containing control point position from registration process
	filenameRef	 : Image file
	
	Output: Phi (nodal deformation mapping)
	'''	
	
	#Filename Results
	#meshfile = io.read(filenameMesh)
	#X = meshfile.points+adjustment
	X += adjustment
    
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
	
	if correctAffine_cpp:
		non_null_subdiagonal = False
		for (i,j) in [(0,1),(0,2),(1,2)]:
			if affineCPP[i,j] != 0 or affineCPP[j,i] != 0:
				non_null_subdiagonal = True
		
		if non_null_subdiagonal:
			if affineCPP[0,0] != affineCPP[1,1]:
				af_max = np.max(np.abs([affineCPP[0,0], affineCPP[1,1]]))
				if affineCPP[0,0]<0:
					af_max *= -1
			else:
				af_max = affineCPP[0,0]
			
			for (i,j) in [(0,1),(0,2),(1,2)]:
				affineCPP[i,j] = 0.0
				affineCPP[j,i] = 0.0
			affineCPP[0,0] = af_max
			affineCPP[1,1] = af_max
	
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
	
	return PHI-adjustment

def preprocess_shape_calibration(paths):
    
    
    # Note that registration tetra is only used to obtain a reference CoM for the registration
    # data but another mesh is going to be used for the triangular registration mesh.
    #   > This is due to registration mesh being usually too high quality and located in a 
    #    different (displaced) coordinate system.
    
    # Unpack registration-associated files
    registration_hq_tetra_geometry = paths['registration_tetra']
    
    # If the tetra geometry is not available, return False
    if not os.path.isfile(registration_hq_tetra_geometry):
        print("Tetrahedral geometry associated to registration not found")
        return False
    
    # Determine center of mass for the registration geometry
    reg_com = determine_geometric_center(registration_hq_tetra_geometry)
    print("reg_com: ", reg_com)
    
    # Extract path for the simulation mesh
    simulation_tetra_geometry = paths['path_to_mesh']+"mesh000000.vtu"
    
    # Determine center of mass for the simulation geometry
    sim_com = determine_geometric_center(simulation_tetra_geometry)
    print("sim_com", sim_com)
    
    # If the tri geometry is not available, generate ply file
    success_ply = generate_registration_ply(calibration_config, delta=reg_com-sim_com)
    
    if not success_ply:
        print(" > Registration ply geometry failed to be generated")
        return False
    else:
        print("  > Registration ply geometry generated successfully")
        return True

if __name__ == '__main__':  

    # Declaration of variables
    case = "FEniCS"
    path_to_mesh = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/MESH/"
    path_to_airway = path_to_mesh+"skel.vtu"    

    # Path to experimental data, load matlab file and prepare data
    path_to_signal = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/PIG5-ARDSnet.npz'
    
    # Finish the mesh path
    path_to_mesh += "%s/"%case

    output_path = 'C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/bayesian-cal-000/'
    folders = sorted(os.listdir(output_path))[:-3]    
      
    
    # Registration-associated geometries
    registration_tetra_geom = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/temp-dev/Exp_NEW.npz"
    registration_tri_geom = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/temp-dev/Surface_Insp.ply"
    # Nifti images
    nifti_cpp = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/temp-dev/cpp_9000-000666.nii.gz"
    nifti_exp = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/temp-dev/NEW_Mask_Exp.nii.gz"
    
    paths = {"output_path":output_path,
             "path_to_mesh":path_to_mesh,
             "path_to_airways":path_to_airway,
             "path_to_signal":path_to_signal,
             "registration_tetra":registration_tetra_geom,
             "registration_tri":registration_tri_geom,
             "nifti_cpp":nifti_cpp,
             "nifti_exp":nifti_exp,}
        

    # Declaration of variables
    case = "FEniCS"
    path_to_mesh = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/MESH/"
    path_to_airway = path_to_mesh+"skel.vtu"    

    # Path to experimental data, load matlab file and prepare data
    signal_path = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/PIG5-ARDSnet.npz'



    # Evaluate if indicated files exist
    print("Checking for the existence of some indicated files:")
    
    for file,name in zip([signal_path, path_to_airway],
                         ["Signal","Airways"]):
        if not os.path.isfile(file): 
            print(" The file '%s' was not found at %s"%(name,file))
            print(" Stop the simulation")
        else:
            print(" %s found"%name)
    

   
    calibration_config = {"iou_rho":5e-2,
                          "continue_calibration":False,
                          "save_vtk":True,
                          "ninternaldivs":[5,10,10,50,20,45],
                          "ncheckpoints":[2,10,2,2,4,4],
                          "paths":paths
                          }
    
    wave_config = {"Tsyr":0.375,
                   "Tpausa":0.375}

    parameters = {"paths":paths,
                  "calibration_config":calibration_config,
                  }
    
    preprocess_shape_calibration(paths)
    
    
    sim = folders[1]
    sim_path = output_path+sim+"/"
    sim_vtu = sim_path+"/VTK/Displacement_%.12f.vtu"%(0.75)
    # Check for file
    isfile = os.path.isfile(sim_vtu)
    
    costs = []
    for sim in folders[:5]:
        value = determine_shape_misfit(sim_path,wave_config,calibration_config)
        costs += [value]

