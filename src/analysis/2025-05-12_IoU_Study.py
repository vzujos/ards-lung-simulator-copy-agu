# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:03:51 2025

@author: angus
"""

import numpy as np
import os
import nibabel as nib
import meshio as io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from collections import Counter


# %% Volume shape

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

def generate_uniform_points(bbox, rho):
    """
    Generate isotropically spaced 3D points inside a bounding box with given point density.

    Parameters:
        bbox : list of tuples [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        rho  : float
            Desired point density (points per unit volume)

    Returns:
        points : ndarray of shape (N, 3)
            The generated 3D points
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin

    d = (1 / rho) ** (1/3)           # Ideal spacing in all directions

    nx = max(1, int(np.round(Lx / d)))
    ny = max(1, int(np.round(Ly / d)))
    nz = max(1, int(np.round(Lz / d)))

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    return points, (nx,ny,nz)


# %%

def plot_bounding_box(bbox, alpha=0.2, color='cyan',
                      points=None, point_color='red', point_size=5,
                      title=None,
                      dpi=100, figsize=(6,6),
                      domain_bounds=None):
    """
    Plot a 3D bounding box with transparent planes and optionally a cloud of 3D points.

    Parameters:
        bbox          : list of tuples [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        alpha         : float, transparency of the planes (0 = fully transparent, 1 = opaque)
        color         : str, color of the box planes
        points        : array-like, shape (Npoints, 3), optional
                        Cloud of 3D points to display inside the bounding box
        point_color   : str, color of the points
        point_size    : float, size of the points
        title         : str, optional title
        dpi           : int, resolution of the figure in dots per inch
        figsize       : tuple, size of the figure (width, height) in inches
        domain_bounds : list of tuples [(xmin, xmax), (ymin, ymax), (zmin, zmax)], optional
                        If provided, sets the axes limits to these values
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox

    # Define the 8 corners of the box
    corners = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax]
    ])

    # Define the 6 planes (each as a list of 4 corners)
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[1], corners[2], corners[6], corners[5]],  # right
        [corners[3], corners[0], corners[4], corners[7]]   # left
    ]

    fig = plt.figure(dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Draw bounding box
    box = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha)
    ax.add_collection3d(box)

    # Optionally draw points
    if points is not None:
        points = np.asarray(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   color=point_color, s=point_size)

    # Set axis limits (use domain_bounds if provided)
    if domain_bounds is None:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
    else:
        (dxmin, dxmax), (dymin, dymax), (dzmin, dzmax) = domain_bounds
        ax.set_xlim(dxmin, dxmax)
        ax.set_ylim(dymin, dymax)
        ax.set_zlim(dzmin, dzmax)

    # Labels
    if title is not None:
        ax.set_title(title)
   # ax.set_xlabel("X")
   # ax.set_ylabel("Y")
   # ax.set_zlabel("Z")

    # Remove tick labels (but keep ticks/grid if needed)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.tight_layout()
    plt.show()


# %%

def rewrite_surface_mesh(vtu, ply, delta=np.zeros((3)), surface_tag=2):
    ''' open a .vtu surface mesh generated in fenics and rewrite as .ply to
    allow being used in trimesh for raycasting
    
    surface_tag = 2 is a default value. In the current meshing process, the
    surface of the lung is tagged with label = 2. Change it if necessary.
    '''
    # Load triangular mesh
    triangle_mesh = io.read(vtu)
    # Those triangles labeled as 2 are the surface. This should be updated if that
    # tag is modified in the meshing process
    surface_mask = np.abs(triangle_mesh.cell_data['f'][0] - surface_tag)<1e-9
    surface_cells = triangle_mesh.cells_dict['triangle'][surface_mask]
    # Export as ply
    writer  = io.Mesh(points=(triangle_mesh.points+delta), 
                  cells={'triangle':surface_cells})
    writer.write(ply)
    
def naive_raycast_cloud(surf_ply,points,direction,
                        vtuname=None, npzname=None, return_cloud=False):
    '''
    for a given surface mesh 'surf_ply', a point_structure that receives 
    indices (i,j,k), and an array that indicates bounds ns = (nx,ny,nz) such 
    that i<nx, j<ny, and k<nz, perform a ray casting algorithm onto the mesh
    such that every point in the cloud will be tested against it, and according
    to the result, a new cloud point will be generated where all the points are
    contained within the surface equidistantly and according to a predefined
    point density.
    
    '''
    # load surface mesh
    surf_lung = trimesh.load(surf_ply)
    
    # array to store the kept points
    stored_points = []
    
    for point in points:

        # retrieve point                
        l0 = surf_lung.ray.intersects_location([point],[direction])[2]
        l1 = surf_lung.ray.intersects_location([point],[-direction])[2]
                
        # only odd combinations imply an inside point
        if len(l0)%2 == 1 and len(l1)%2==1:
            stored_points+=[1]
        else:
            stored_points+=[0]

    return np.array(stored_points)     

def InterpolateBSplines(filenameMesh, filenameCPP, filenameRef, 
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
	meshfile = io.read(filenameMesh)
	X = meshfile.points+adjustment
	
	#Reference Affine
	img = nib.load(filenameRef)
	affine=img.affine
	
	# Check for a non-diagonal affine matrix
	if correctAffine_ref:
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
	
	return PHI	

def extract_surface_mesh(points, cells):
    """
    Extract surface triangle mesh from a tetrahedral mesh.

    Parameters:
        points : ndarray (Npoints, 3)
            3D coordinates of the mesh nodes
        cells : ndarray (Nelems, 4)
            Tetrahedral elements (indices into points)

    Returns:
        tripoints : ndarray (M, 3)
            Points used in the triangle surface mesh
        tricells : ndarray (N, 3)
            Triangle surface elements (indices into tripoints)
    """
    # Generate all faces from tetrahedra
    faces = np.concatenate([
        cells[:, [0, 1, 2]],
        cells[:, [0, 1, 3]],
        cells[:, [0, 2, 3]],
        cells[:, [1, 2, 3]],
    ]).reshape(-1, 3)

    # Sort vertex indices in each face so identical faces match
    sorted_faces = np.sort(faces, axis=1)

    # Count face occurrences
    face_count = Counter(map(tuple, sorted_faces))

    # Extract faces that occur only once => surface
    surface_faces = np.array([face for face, count in face_count.items() if count == 1])

    # Find unique points used in the surface
    unique_indices, inverse_indices = np.unique(surface_faces, return_inverse=True)

    tripoints = points[unique_indices]
    tricells = inverse_indices.reshape(-1, 3)

    return tripoints, tricells

def determine_bbox(mesh0,mesh1,f=0.025):
    '''
    mesh0: Corrected simulated points
    mesh1: Registrated points
    
    '''
    # Load meshes
    mesh0 = io.read(mesh0)
    mesh1 = io.read(mesh1)
    
    points0 = mesh0.points; 
    points1 = mesh1.points;
    
    ps0_x0 = points0[:,0].min(); ps0_x1 = points0[:,0].max()
    ps0_y0 = points0[:,1].min(); ps0_y1 = points0[:,1].max()
    ps0_z0 = points0[:,2].min(); ps0_z1 = points0[:,2].max()
    
    #  Registrated points bounds
    ps1_x0 = points1[:,0].min(); ps1_x1 = points1[:,0].max()
    ps1_y0 = points1[:,1].min(); ps1_y1 = points1[:,1].max()
    ps1_z0 = points1[:,2].min(); ps1_z1 = points1[:,2].max()
    
    # Actual bounds
    x0 = min([ps0_x0,ps1_x0]); x1 = max([ps0_x1, ps1_x1])
    y0 = min([ps0_y0,ps1_y0]); y1 = max([ps0_y1, ps1_y1])
    z0 = min([ps0_z0,ps1_z0]); z1 = max([ps0_z1, ps1_z1])

    # Deltas
    dx = x1-x0; dy = y1-y0; dz = z1-z0
    
    # Bounding box
    return [(x0-dx*f,x1+dx*f),(y0-dy*f,y1+dy*f),(z0-dz*f,z1+dz*f)]

# %% Paths associated to this problem

# Subject

subject = 5

# We'll interpolate intensities towards a different mesh
r_mesh_path = "D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/PIG%i/ARDSnet/MESH/"%subject
r_nifti_path = "D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/PIG%i/ARDSnet/NIFTI/"%subject
r_registration_path = "D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/PIG%i/ARDSnet/REGISTRATION/"%subject

# Simulation-associated paths
s_mesh_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/ARDSnet/medium-fine/"%subject
sim_tetra_mesh = s_mesh_path+"FEniCS/mesh000000.vtu"
in_simulated_ee_surf = s_mesh_path+"FEniCS/boundary_markers000000.vtu"

# Source files
in_registered_ee_surf = r_mesh_path+"Surface_Exp.vtk"
reg_npz = np.load(r_mesh_path+"Exp_NEW.npz")

# Surfaces to be used in the ray casting algorithm
out_registered_ee_surf = s_mesh_path + "reg_ee_surf.ply"
out_simulated_ee_surf = s_mesh_path + "sim_ee_surf.ply"
out_registered_ei_surf = s_mesh_path + "reg_ei_surf.ply"
out_registered_alt_ei_surf = s_mesh_path + "reg_ei_surf_alt.ply"

out_simulated_ei_surf = s_mesh_path + "sim_ei_surf.ply"

# Nifti files
filenameCPP = r_registration_path+"cpp_9000-000666.nii.gz"
filenameRef = r_nifti_path+"Exp.nii.gz"

# Point towards the deformed simulated mesh result
sim_root = "C:/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/"
simulated_mesh = sim_root+"/codes/DeleteMe/MFSIMS/PIG%i/output/post/"%subject
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
  

# %% Rewriting the surface meshes into a ply format

# Rewrite simulated surface mesh
if not os.path.isfile(out_simulated_ee_surf):
    rewrite_surface_mesh(in_simulated_ee_surf, 
                         out_simulated_ee_surf,delta=delta)

# Rewrite registered surface mesh
if not os.path.isfile(out_registered_ee_surf):
    reg_surf = io.read(in_registered_ee_surf)
    reg_ply = io.Mesh(points=reg_surf.points.astype(float), cells=reg_surf.cells_dict)
    reg_ply.write(out_registered_ee_surf, file_format='ply')
    
    
# %% Generate the end-inspiration 'registered' mesh

# Determine the displaced surface points
phi = InterpolateBSplines(in_registered_ee_surf,filenameCPP,filenameRef)

temp_mesh = io.read(in_registered_ee_surf)
out_mesh = io.Mesh(points=phi,
                   cells=temp_mesh.cells_dict)
out_mesh.write(out_registered_ei_surf)

# %% Generate a surface mesh out of the deformed simulated lung at the end of the inspiratory pause

# Load the simulated mesh
sim_ei_mesh = io.read(simulated_mesh)
# Extract relevant field, compose deformed mesh
points = sim_ei_mesh.points
u = sim_ei_mesh.point_data['u']
phi = points+u
# Extract surface mesh and points
tripoints, tricells = extract_surface_mesh(phi, sim_ei_mesh.cells_dict['tetra'])
# Generate the ply file 
sim_ei_mesh_out = io.Mesh(points=tripoints+delta,cells={'triangle':tricells})
sim_ei_mesh_out.write(out_simulated_ei_surf) 

# %% Alternative end-inspiratorion 'registered' mesh
# Using the simulated mesh, displacing it, and using it to interpolate data

# Load the reference surface mesh
mesh = io.read(out_simulated_ee_surf)

phi = InterpolateBSplines(out_simulated_ee_surf,filenameCPP,filenameRef)
#phi = InterpolateBSplines(s_mesh_path+"temporal_disp_sim_mesh.vtu",filenameCPP,filenameRef)
reg_alt_ei_mesh_out = io.Mesh(phi, cells=mesh.cells_dict)
reg_alt_ei_mesh_out.write(out_registered_alt_ei_surf)

# %%
bbox = determine_bbox(out_registered_alt_ei_surf,out_simulated_ei_surf)

# %% Generate cloud

# Define a point density
rho = 5e-2
# Generate cloud
box, ns = generate_uniform_points(bbox,rho)
# Generate a plot
if True: plot_bounding_box(bbox, points=box)


# %% Execute raycasting algorithm generating masks for the used points

sim_mask = naive_raycast_cloud(out_simulated_ei_surf,box,np.array([1.,0.,0.]))
reg_mask = naive_raycast_cloud(out_registered_alt_ei_surf,box,np.array([1.,0.,0.]))

# %% Evaluate results

# Dots within the box
ndots = box.shape[0]

# Dots within each mask
sim_dots = np.count_nonzero(sim_mask)
reg_dots = np.count_nonzero(reg_mask)

# Complex masks
union_mask = np.logical_or(sim_mask,reg_mask)
inter_mask = np.logical_and(sim_mask,reg_mask)
inter_dots = np.count_nonzero(inter_mask)
union_dots = np.count_nonzero(union_mask)
only_sim = sim_mask==1 - inter_mask
only_reg = reg_mask==1 - inter_mask

# Present data
print("Simulated markers = %i/%i"%(sim_dots,ndots))
print("Registered markers = %i/%i"%(reg_dots,ndots))
print(" > Intersection markers = %i"%inter_dots)
print(" > Union markers = %i"%union_dots)
print(" *** IOU SCORE = %.2f ***"%(inter_dots/union_dots))

# %%
# Generate plots to visualize
# Simulation
plot_bounding_box(bbox, points=box[sim_mask==1], point_size=8,dpi=300, point_color='r',color='w')

#%%
# Experiment
plot_bounding_box(bbox, points=box[reg_mask==1], point_size=8,dpi=300, point_color='g',color='w')


#%%
    
plot_bounding_box(bbox, points=box[union_mask],title="Union")
plot_bounding_box(bbox, points=box[inter_mask],title="Intersection")

# %%
# Simulation minus intersection; Simulation exclusive points
plot_bounding_box(bbox, points=box[only_sim], point_color="r",color="w")

# %%
# Experiment minus intersection; Experiment exclusive points
plot_bounding_box(bbox, points=box[only_reg], point_color="g",color="w")