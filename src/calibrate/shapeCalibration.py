# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:57:14 2024

@author: angus
"""

import os
import sys
sys.path.append("/home1/agustin.perez/model2/src/")
import numpy as np
from vcv_lung_diaphragm import execute_vcv_simulation
import pyvista as pv
import meshio as io
import trimesh
import embreex
from collections import Counter
from trimesh.ray.ray_pyembree import RayMeshIntersector
import nibabel as nib


def generate_vcv_dictionary(wave_config,
                            parenchyma_config,
                            peripheral_config,
                            calibration_config,
                            **kwargs):
    
    # Flow regimes
    # A: Flow moves from zero to prescribed value. Lasts 0.001 s
    # B: Steady inflation. Lasts 0.999 s
    # C: Transition. Changes steady flow to zero flow. Lasts 0.001 s
    # D: Zero flow. Achieve plateau pressure. 0.25 s
    # E: Expiration begins. Rapid changes. 0.25 s
    # F: Pseudo-steady expiration. Lasts long but is kind of regular. 1.75 s.
        
   # Simulation Codename
   # codename = "CORNELL-PIG6-ARDSnet" # This is a name for the folder where the output is directed
   # mesh_name = "" #  This should change for different states/subjects; While in development, just keep 'stable'
   # case ="FEniCS" # This is the specific name for the mesh in use

    # Checkpoint parameters
    restart_from_last_checkpoint = False
    save_checkpoints = False
    save_vtk = calibration_config['save_vtk']
    
    # Path-related
    paths = calibration_config['paths']
    path_to_mesh = paths['path_to_mesh'] 
    output_to = paths['output_to']
    path_to_airway = paths['path_to_airways']

    # Processing the paths
    #path_to_mesh =  sf.manage_mesh_directory(path_to_mesh,mesh_name,case)
    #output_to = sf.manage_output_directory(output_to,codename,restart_from_last_checkpoint)
    
    permeability_dict = {"variable_permeability":parenchyma_config['variable_permeability'],
                         "permeability_file":parenchyma_config['permeability_file'],
                         "KK_exp":parenchyma_config['permeability_exp'],
                         "KK_factor":parenchyma_config['permeability_factor'],
                         }

    porosity_dict = {"activate":parenchyma_config['variable_porosity'],
                     "mean":parenchyma_config['porosity_mean'],
                     "file":parenchyma_config['porosity_file']}

    # Temporal config
    ncheckpoints = calibration_config['ncheckpoints']
    ninternaldivs = calibration_config['ninternaldivs']

    # Tolerances in the iterative cycles
    max_nit = 30
    tol_p_it = 1e-3
    tol_v_it = 1e-2
    
    # Selecting solver
    solver_type = "dict"
    solver_dict = {"nonlinear_solver":"snes",
                   "snes_solver":{"linear_solver":"mumps",
                                  "relative_tolerance":1e-6,
                                  "absolute_tolerance":1e-8,
                                  "maximum_iterations":60,
                                  "line_search":"bt",
                                  "report":True,
                                  "error_on_nonconvergence":True,
                                  "preconditioner":"default"}
                                  }
    
    # Modifying the base resistance for the airway tree
    additional_resistances = {"upstream":None,
                              "downstream":None,
                              "pedley_config":{"activate":peripheral_config['pedley_activate'],
                                               "expiratory_gamma":peripheral_config['pedley_expiratory_gamma'],
                                               "inspiratory_gamma":peripheral_config['pedley_inspiratory_gamma'], # 0.327 original value
                                               "tolerance":peripheral_config['pedley_tolerance'],
                                               "nitmax":peripheral_config['pedley_nitmax']}
                              }
                              
    time_config = {"ncycles":wave_config['ncycles'],
                   "Tsyr":wave_config['Tsyr'],
                   "Tpausa":wave_config['Tpausa'],
                   "Texp":wave_config['Texp']}

    # Constitutive model; 'ber','ma','yoshi','bir2019','rausch'
    cm = parenchyma_config['constitutive_model']
    beta = 1.075

    vt = wave_config['vt']
    K_cw_stiffness = peripheral_config['K_cw_stiffness']
    K_d_stiffness = peripheral_config['K_d_stiffness']
    c_tissue = parenchyma_config['c_tissue']
    
    # Isotropic prestrain F_prestrain = alpha*I
    isotropic_prestrain = parenchyma_config['isotropic_prestrain']
    
    # Generate flow and pressure functions dictionary
    # These functions will be manually deactivated in this implementation
    if wave_config['bc_dict'] is None:
        bc_dict = {'activate':False,
                   'flow_func':None,
                   'pressure_func':None}
    else:
        bc_dict = wave_config['bc_dict']

    args = {"restart_from_last_checkpoint":restart_from_last_checkpoint,
            "save_checkpoints":save_checkpoints,
            "mesh_dir":path_to_mesh,
            "output_to":output_to,
            "ncheckpoints":ncheckpoints,
            "ninternaldivs":ninternaldivs,
            "K_cw_stiffness":K_cw_stiffness,
            "K_d_stiffness":K_d_stiffness,
            "permeability_dict":permeability_dict,
            "porosity_config":porosity_dict,
            "isotropic_prestrain":isotropic_prestrain,
            "solver_type":solver_type,
            "solver_dict":solver_dict,
            "tidal_volume":vt*(10**6), # conversion from L to mm3
            "time_config":time_config,
            "constitutive_model":cm,
            "constitutive_parameters":(c_tissue,beta),
            "path_to_airway":path_to_airway,
            "tol_p_it":tol_p_it,
            "tol_v_it":tol_v_it,
            "max_nit":max_nit,
            "save_vtk":save_vtk,
            "bc_dict":bc_dict,
            "additional_resistances":additional_resistances,
            }
        
    return args


def execute_single_simulation(path,args,params,calibration_type):
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    # Manage parameters
    
    # Update dictionary
    args_a = args.copy()  
   
    # Write parameters
    np.savetxt(path+"iteration_params.txt", params)
    
    print("Requested calibration type: %s"%calibration_type)
    
    if calibration_type == "shape-calibration":
        # The stiffness of different springs are being calibrated
        K_cw_stiffness, K_d_stiffness = params
        
        if K_cw_stiffness < 0.0:
            print("[STOPPING SIMULATION] Negative K_cw_stiffness: %.5f"%K_cw_stiffness)
            return False
        else:
            print(" > K_cw: %.5f"%K_cw_stiffness)
        
        if K_d_stiffness < 0.0:
            print("[STOPPING SIMULATION] Negative K_d_stiffness: %.5f"%K_d_stiffness)
            return False       
        else:
            print(" > K_d: %.5f"%K_d_stiffness)
        
        args_a["K_cw_stiffness"] = K_cw_stiffness
        args_a["K_d_stiffness"] = K_d_stiffness
        args_a["output_to"] = path
        print("Run A: output_to: %s"%args_a["output_to"])
        
    else:
        print("Calibration type '%s' not valid"%calibration_type)
        return False

    # 'complete lung' simulation
    try:
        execute_vcv_simulation(args_a)
        flag = True
    except:
        flag = False  

    # confirm whether the simulations succeded or crashed
    if not flag: 
        print("Convergence error when simulating the 'complete' lung")
    
    return flag

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

    return os.path.isfile(simulation_ply)

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

def preprocess_shape_calibration(calibration_config, wave_config):
    
    
    # Note that registration tetra is only used to obtain a reference CoM for the registration
    # data but another mesh is going to be used for the triangular registration mesh.
    #   > This is due to registration mesh being usually too high quality and located in a 
    #    different (displaced) coordinate system.
    
    # Unpack registration-associated files
    registration_hq_tetra_geometry = calibration_config['paths']['registration_hq_tetra']
    
    # If the tetra geometry is not available, return False
    if not os.path.isfile(registration_hq_tetra_geometry):
        print("Tetrahedral geometry associated to registration not found")
        return False
    
    # Determine center of mass for the registration geometry
    reg_com = determine_geometric_center(registration_hq_tetra_geometry)
    print("reg_com: ", reg_com)
    
    # Extract path for the simulation mesh
    simulation_tetra_geometry = calibration_config['paths']['path_to_mesh']+"mesh000000.vtu"
    
    # Determine center of mass for the simulation geometry
    sim_com = determine_geometric_center(simulation_tetra_geometry)
    print("sim_com", sim_com)
    
    # If the tri geometry is not available, generate ply file
    success_ply = generate_registration_ply(calibration_config, delta=reg_com-sim_com)
    
    if not success_ply:
        print(" > Registration ply geometry failed to be generated")
        return False
    
    return True
    

def step(path, it, args, x, calibration_config, wave_config,
         parallel=True,geometry_factor=1.0):
    '''
    All the steps joined in a simple function
    '''   
    # Type of calibration 'regression' or 'partial-signal'
    calibration_type = calibration_config['calibration_type']

    # Build path
    it_path = path+"%3.3i/"%it;
    
    print("Generating the step at path:\n > '%s'"%it_path)
    if not os.path.isdir(it_path): os.mkdir(it_path)
    
        
    if calibration_type == "shape-calibration":
        
        flag = execute_single_simulation(it_path,args,x,calibration_config['calibration_type'])
        
        if not flag: return None

        values = determine_shape_misfit(it_path, wave_config, calibration_config)
        
        return values
    
    else:
        print("Error in the calibration_type: '%s'"%calibration_type)
        return None
    
def reflect(vs,wid):
    # Compute the reflected vertex of the worst point
    nv = np.zeros_like(vs[wid]) # new vertex
    for e,v in enumerate(vs):
        if e == wid:
            nv -= v
        else:
            nv += v
    return nv

def extend(wv,rv,alpha=1.0):
    # Define the extension 'ext' as half the vector between the worst and the reflected points
    ext = (rv-wv)/2
    # Note that alpha allows to modify the length of the extension
    return rv + alpha*ext

def inner_contraction(wv, rv, beta=0.5):
    # Define the extension 'ext' as half the vector between the worst and the reflected points
    ext = (rv-wv)/2
    # Note that beta sends the new point the original simplex
    return rv - beta*ext

def outer_contraction(wv, rv, gamma=0.5):
    # Define the extension 'ext' as half the vector between the worst and the reflected points
    ext = (rv-wv)/2
    # Note that beta sends the new point the original simplex
    return rv + gamma*ext

def shrink(vs, sorter, delta=0.5):
    
    mid_ext = vs[sorter[0]] - vs[sorter[1]]
    new_mid = vs[sorter[1]] + delta*mid_ext
    
    wst_ext = vs[sorter[0]] - vs[sorter[2]]
    new_wst = vs[sorter[2]] + delta*wst_ext
    
    return new_mid, new_wst

def iterate(path,it,vs,zs,args,calibration_config,wave_config):
    
    # Sort the points depending on they performance
    sorter = np.argsort(zs)
    
    # Retrieve the sorted ids for each vector
    bid = sorter[0] # best
    mid = sorter[1] # mid
    wid = sorter[2] # worst
    
    # Determine reflected vector  
    rv = reflect(vs,wid)
    
    # Compute the z-value of the new vertex
    text_manager(path+"it_indexer.txt", "%3.3i - Reflect\n"%(it+1), "a")
    it+=1; rz = step(path,it,args,rv,calibration_config,wave_config)
    
    if rz < zs[bid]: # Extend
    
        # if the reflected point is better than the best point, extend
        # Determine extended point
        ev = extend(vs[wid],rv)
        # Compute the z-value for the extended point
        text_manager(path+"it_indexer.txt", "%3.3i - Extend\n"%(it+1), "a")
        it+=1; ez = step(path,it,args,ev,calibration_config,wave_config)
        
        if ez < rz:
            return (ev, ez), "Extend",it
        else:
            return (rv, rz), "Reflect",it
        
    elif rz < zs[mid]: 
        return (rv,rz), "Reflect",it
        
    elif rz < zs[wid]:
        # if the reflected point is better than the worst point, but not better than the other
        # two, then contract
        icv = inner_contraction(vs[wid], rv)
        ocv = outer_contraction(vs[wid], rv)
        text_manager(path+"it_indexer.txt", "%3.3i - Inner Contraction\n"%(it+1), "a")
        it+=1; icz = step(path,it,args,icv,calibration_config,wave_config)
        text_manager(path+"it_indexer.txt", "%3.3i - Outer Contraction\n"%(it+1), "a")
        it+=1; ocz = step(path,it,args,ocv,calibration_config,wave_config)
        
        if icz < ocz: 
            return (icv,icz), "Inner-contraction",it
        else:
            return (ocv,ocz), "Outer-contraction",it
        
    else:
        # Shrinkage
        nmv, nwv = shrink(vs,sorter)
        text_manager(path+"it_indexer.txt", "%3.3i - Shrink-1\n"%(it+1), "a")
        it+=1; nmz = step(path,it,args,nmv,calibration_config,wave_config)
        text_manager(path+"it_indexer.txt", "%3.3i - Shrink-2\n"%(it+1), "a")
        it+=1; nwz = step(path,it,args,nwv,calibration_config,wave_config)
        return (nmv,nmz,nwv,nwz), "Shrink", it


def process(vs,zs,out):
    
    if len(out) == 4:
        # implies shrinkage
        # Sort the points depending on they performance
        sorter = np.argsort(zs)
        bid = sorter[0] # best
        
        nvs = np.array([vs[bid], out[0],out[2]])
        nzs = [zs[bid],out[1],out[3]]
    
    elif len(out) == 2:
        
        sorter = np.argsort(zs)
        bid = sorter[0] # best
        mid = sorter[1]
        
        nvs = np.array([vs[bid], vs[mid],out[0]])
        nzs = [zs[bid],zs[mid],out[1]]
    else:
        raise Exception("Error at process routine")
        return None

    # sort the new points
    bid, mid, wid = np.argsort(nzs)
   
    dim = len(vs[0])
    
    if dim == 3:
        area_err = np.abs(np.linalg.det(np.hstack([nvs])))
    elif dim==2: 
        area_err = np.abs(np.linalg.det(np.hstack([nvs,np.ones((3,1))])))
    else:
        area_err = -100000.
    funct_err = np.abs((nzs[bid]-nzs[wid])/(np.abs(nzs[bid])+1e-9))
    
    err = [area_err, funct_err]
    return nvs, nzs, err


def text_manager(path,text,mode):
    
    file = open(path,mode=mode)
    file.write(text)
    file.close()
    

def manage_paths(calibration_config):
    
    ''' 
    Determines the current path were the simulation data will be
    dumped, also creates the 'history' folder if it does not exist, 
    which will be used to resume the analysis and to help postprocessing
    '''
    # Extract paths
    paths = calibration_config['paths']
    
    # In case that we restart calibration from another point
    continue_calibration = calibration_config['continue_calibration']
    restart_from = calibration_config['restart_from']
    
    # Calibration config
    maximum_evaluations = calibration_config['maximum_evaluations']
    
    # Paths asociated data
    task_name = paths['task_name']
    output_to = paths['output_to']
    
    if not continue_calibration: 
        # We enter this branch if we are starting a new simulation
        for run_no in range(maximum_evaluations):
            # check if a folder already exists
            if not os.path.isdir(output_to+task_name%run_no):
                # if it doesn't, create and exit loop
                os.mkdir(output_to+task_name%run_no)
                break
        path = output_to+task_name%run_no
    else:
        path = output_to+task_name%restart_from
        
    if not os.path.isdir(path+"history/"): 
        print(" > Creating 'history' directory")
        os.mkdir(path+"history/")
    else:
        print(" > 'history' directory already exists")
    
    return path
    
def initialize_calibration(path,
                           vcv_dict,
                           calibration_config,
                           wave_config):
     
    # Retrieve initializing points
    x0 = calibration_config['x0']
    x1 = calibration_config['x1']
    x2 = calibration_config['x2']
    # Shape them as a numpy array
    xs = np.array([x0,x1,x2])
    
    # Continuing calibration boolean
    continue_calibration = calibration_config['continue_calibration']
    
    # Misfit holder
    ms = []
    
    # Parallel simulation flags
    parallel = calibration_config['parallel_simulations']
    
    cont = preprocess_shape_calibration(calibration_config,wave_config)
    if not cont:
        print("Preprocessing of the calibration has failed")
        return None
    
    # If the calibration starts from scratch
    if not continue_calibration:
        
        # Execute every simulation
        for it,x in enumerate(xs):
            mode = "w" if it==0 else "a" 
            text_manager(path+"it_indexer.txt", "%3.3i - Initial\n"%it, mode)

            # Complete step to compute a misfit
            ms += [step(path, it, vcv_dict, x, calibration_config, wave_config,
                        parallel=parallel,)]

        history = {"simplex":[xs.copy()],
                   "misfits":[np.array(ms)],
                   "actions":["Start"],
                   "area_err":[],
                   "func_err":[],
                   }
        np.savez(path+"history/calibration_results.npz",**history)

    else:
                
        old = np.load(path+"history/calibration_results.npz")
            
        history = {"simplex":old["simplex"],
                   "misfits":old["misfits"],
                   "actions":old["actions"],
                   "area_err":old["area_err"],
                   "func_err":old["func_err"],
                }

        simulations = os.listdir(path)
        for folder in ["history","images","it_indexer.txt"]:
            if folder in simulations: simulations.remove(folder)
        
        it = int(simulations[-1])
        xs = history["simplex"][0]
        ms = history["misfits"][0]
    
    nm_state = (it, xs, ms, history)
    return nm_state
        
def neldermead_iterations(path, nm_state, args, 
                          calibration_config, wave_config):
        
    err = [1,1]
    cycles = 0
    
    tol = calibration_config["neldermead_tolerance"]
    nitmax = calibration_config["neldermead_nitmax"]
    
    #Unpack Nelder-Mead state
    it, xs, ms, history = nm_state

    # Build targets

    # =============================================================================
    # Nelder-Mead loop
    # =============================================================================
    
    # Stopping criteria is variation within successive points being below tolerance
    # and reaching a maximum number of iterations within this loop.
    
    while((err[0] > tol and err[1] > tol) or (cycles < nitmax) ):
        
        # Declare the current iteration path    
        out, action, it = iterate(path,it,xs,ms,args,calibration_config,wave_config)
        xs, ms, err = process(xs,ms,out)
        
        history["simplex"] += [xs.copy()]
        history["misfits"] += [np.array(ms)]
        history["actions"] += [action]    
        history["area_err"] += [err[0]]    
        history["func_err"] += [err[1]]    
        
        np.savez(path+"history/calibration_results.npz",**history)
        
        cycles += 1


def execute_vcv_calibration(wave_config,
                            parenchyma_config,
                            peripheral_config,
                            calibration_config):
                        
    ''' 
    Calls the different steps than comprehend a VCV calibration 
    '''
    print("-------------------------------")
    print("|| SHAPE CALIBRATION PROGRAM ||")
    print("-------------------------------")
    print()
    path = manage_paths(calibration_config)
    print("+++ Generating results into path: '%s' +++\n"%path)
    
    vcv_dict = generate_vcv_dictionary(wave_config,
                                       parenchyma_config,
                                       peripheral_config,
                                       calibration_config) 
    
    nm_state = initialize_calibration(path,vcv_dict,calibration_config,wave_config)
    
    neldermead_iterations(path, nm_state, vcv_dict,
                          calibration_config, wave_config)
        
# %%


if __name__ == '__main__':
        
    print("Debugging...")
    
    # Calibration type "partial-signal" or "regression"
    calibration_type = "shape-calibration"
    
    # Target values
    target_Ccw = 367.8 # placeholder (mL/cmH2O) Chest wall compliance
    target_Cl = 22.1  # placeholder (mL/cmH2O) Lung compliance
    target_peep = 10.5 # placeholder (cmH2O) PEEP value
    target_pplat = 20.6# (cmH2O)
    target_ppeak = 34.1 # (cmH2O)
    target_peep = 11.1 # (cmH2O)
    
    targets = np.array([target_Ccw, target_Cl, target_peep])

    # Physical parameters    
    K_cw_stiffness = 0.0168333
    K_d_stiffness = K_cw_stiffness/10.
    
    expiratory_gamma = 0.200
    inspiratory_gamma = 0.200

    constitutive_model = "bir2019"
    c_tissue = 2.5
    KK_exp = 5
    KK_factor = 1.0
    isotropic_prestrain = 1.0
    rho = 5e-2
    
    # Declaration of variables
    case = "FEniCS"
    path_to_mesh = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    path_to_airway = path_to_mesh+"skel.vtu"
    
    path_to_mesh += "%s/"%case
    
    output_to = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    task_name = "calibration-vcv-%2.2i/"
    
    # Organization of variables
    paths = {"path_to_mesh":path_to_mesh,
             "path_to_airways":path_to_airway,
             "output_to":output_to,
             "task_name":task_name,
             }
    
    # Wave config
    vt = 0.221
    Tsyr = 0.375;
    Tpausa = 0.375;
    Texp = 1.25;
    ncycles=1
    
    wave_config = {"vt":vt,
                   "Tsyr":Tsyr,
                   "Tpausa":Tpausa,
                   "Texp":Texp,
                   "ncycles":ncycles,
                   "pplat":target_pplat,
                   "ppeak":target_ppeak,
                   "peep":target_peep}
    
    
    x0 = (0.030, 0.00100)
    x1 = (0.020, 0.00050)
    x2 = (0.025, 0.00075)
    
    parenchyma_config = {"variable_porosity":True,
                         "porosity_mean":0.5,
                         "porosity_file":"phi0.xml.gz",
                         "variable_permeability":True,
                         "permeability_file":"k0.xml",
                         "permeability_exp":KK_exp,
                         "permeability_factor":KK_factor,
                         "c_tissue":c_tissue,
                         "constitutive_model":constitutive_model,
                         "isotropic_prestrain":isotropic_prestrain
                         }
    
    peripheral_config = {"K_cw_stiffness":K_cw_stiffness,
                         "K_d_stiffness":K_d_stiffness,
                         "pedley_activate":True,
                         "pedley_expiratory_gamma":expiratory_gamma,
                         "pedley_inspiratory_gamma":inspiratory_gamma,
                         "pedley_tolerance":1e-8,
                         "pedley_nitmax":100}
    
    # continue_calibration: Boolean, to restart calibration from some intermediate point
    # restart_from: Integer, number of a calibration process
    # maximum_evaluations: Integer, maximum number of calibration processes to be checked
    # target_Ccw: Float, Chest-wall compliance to be targeted
    # target_Cl: Float, Lung compliance to be targeted
    # x0, x1, x2: Float tuple, initial guesses for C_birzle and K_stiffness
    # parallel_simulations: Boolean, execute both required simulations per step in parallel
    # neldermead_tolerance: Float, tolerance required to stop NM loop (misfit-related)
    # neldermead_nitmax: Integer, maximum number of NM iterative cycles.
    
    calibration_config = {"calibration_type":calibration_type,
                          "iou_rho":rho,
                          "continue_calibration":False,
                          "restart_from":None,
                          "maximum_evaluations":100,
                          "targets":targets,
                          "x0":x0,"x1":x1,"x2":x2,
                          "parallel_simulations":True,
                          "neldermead_tolerance":1e-6,
                          "neldermead_nitmax":30,
                          "ninternaldivs":[5,10,5,10,20,45],
                          "ncheckpoints":[2,10,2,2,4,4],
                          "paths":paths,
                          }
           

    execute_vcv_calibration(wave_config,
                            parenchyma_config,
                            peripheral_config,
                            calibration_config,
                            )
