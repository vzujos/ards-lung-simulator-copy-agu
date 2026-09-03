# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:11:50 2025

@author: angus
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:27:37 2024

@author: angus
"""

import os
import numpy as np
import meshio as io
import networkx as nx
import nibabel as nib
from collections import defaultdict
import dolfin
import sys

def assign_stiffness(phi, k=80, c_tissue_max=10.0, c_tissue_min=2.0):
    # Sigmoid transition centered at 0.2, steepness controlled by k
    return c_tissue_max - (c_tissue_max - c_tissue_min) * (1 - 1 / (1 + np.exp(-k * (0.2 - phi))))

def tetvol(points):
    ps = np.hstack([points,np.ones((4,1))])
    return np.abs(np.linalg.det(ps))/6

def initialize_FEniCS_geometry(tetra_path,out_path, bc_info=None, tol=1e-8):

    M = io.read(tetra_path)
    xyz = M.points
    ien = M.cells_dict['tetra']

    # =============================================================================
    # CREATE MESH
    # =============================================================================
    
    # create mesh object and add nodes and elements
    """ mesh editor object
    (ref: https://fenicsproject.org/docs/dolfin/2017.2.0/python/programmers-reference/cpp/mesh/MeshEditor.html)
    """
    mesh = dolfin.cpp.mesh.Mesh()
    editor = dolfin.cpp.mesh.MeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3)
    editor.init_vertices(xyz.shape[0])
    editor.init_cells(ien.shape[0])
    [editor.add_vertex(i, n) for i, n in enumerate(xyz)]
    [editor.add_cell(i, n) for i, n in enumerate(ien.astype(np.uintp))]
    editor.close()

    coord = mesh.coordinates()
    coord[:, 0] -= (coord[:, 0].max() + coord[:, 0].min()) / 2.
    coord[:, 1] -= (coord[:, 1].max() + coord[:, 1].min()) / 2.
    coord[:, 2] -= (coord[:, 2].max() + coord[:, 2].min()) / 2.

    mesh.init(2, 3)

    boundary_markers = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)

    if bc_info is not None:
        print(" > Cargando malla superficial para marcar condiciones de contorno...")

        meshio_vtu = io.read(bc_info)

        # Extraer triángulos y su marcado 'Diaphragm'
        vtu_points = meshio_vtu.points
        vtu_diaphragm_mask = meshio_vtu.cell_data['Diaphragm'][0]
        vtu_dorsal_mask = meshio_vtu.cell_data['Dorsal'][0]
        vtu_mediastinum_mask = meshio_vtu.cell_data['Mediastinal'][0]

        vtu_cells = meshio_vtu.cells_dict['triangle']   
        
        print(" > Numero de celdas marcadas como 'Diafragm': %i"%np.count_nonzero(vtu_diaphragm_mask))
        print(" > Numero de celdas marcadas como 'Dorsal': %i"%np.count_nonzero(vtu_dorsal_mask))
        print(" > Numero de celdas marcadas como 'Mediastinal': %i"%np.count_nonzero(vtu_mediastinum_mask))

        # Generar lista con los centros de cada celda en el diafragma
        diaphragm_cell_centers = [np.average(vtu_points[cell],axis=0) for cell in vtu_cells[vtu_diaphragm_mask==1]]
        diaphragm_cell_centers = np.array(diaphragm_cell_centers)
        print(" > Número de celdas diafragma en uso: %i"%(len(diaphragm_cell_centers)))
        
        # Generar lista con los centros de cada celda en el dorso
        dorsal_cell_centers = [np.average(vtu_points[cell],axis=0) for cell in vtu_cells[vtu_dorsal_mask==0]]
        dorsal_cell_centers = np.array(dorsal_cell_centers)
        print(" > Número de celdas dorsales en uso: %i"%(len(dorsal_cell_centers)))        
        
        # Generar lista con los centros de cada celda en el mediastino
        mediastinum_cell_centers = [np.average(vtu_points[cell],axis=0) for cell in vtu_cells[vtu_mediastinum_mask==1]]
        mediastinum_cell_centers = np.array(mediastinum_cell_centers)
        print(" > Número de celdas mediastino en uso: %i"%(len(mediastinum_cell_centers)))
        
        marcados_diafragma = 0;         # Contador
        marcados_dorsales = 0;         # Contador
        marcados_mediastino = 0;         # Contador

        
        # Por cada cara de la malla
        for facet in dolfin.facets(mesh):
            
            # Si la cara es exterior
            if facet.exterior():
                
                # Trae los nodos y determina el centro
                node_indices = facet.entities(0)
                node_coords = mesh.coordinates()[node_indices]
                facet_center = np.average(node_coords,axis=0)
                
                if np.any(np.linalg.norm(diaphragm_cell_centers-facet_center,axis=1)<tol):
                    #print("Nodo diafragma encontrado, marcados: %i"%(marcados+1))
                    boundary_markers[facet] = 1
                    marcados_diafragma += 1
                elif np.any(np.linalg.norm(dorsal_cell_centers-facet_center,axis=1)<tol):
                    boundary_markers[facet] = 3 # Dorsal
                    marcados_dorsales += 1
                elif np.any(np.linalg.norm(mediastinum_cell_centers-facet_center,axis=1)<tol):
                    boundary_markers[facet] = 4 # This is mediastinum
                    marcados_mediastino += 1                
                else:
                    boundary_markers[facet] = 2 # Chest wall

        print(f" > Caras marcadas como diafragma: {marcados_diafragma}")
        print(f" > Caras marcadas como dorsales: {marcados_dorsales}")
        print(f" > Caras marcadas como mediastino: {marcados_mediastino}")
        print(" > Marcado de diafragma completado.")

    file_m = dolfin.File(out_path + 'mesh.pvd')
    file_m << mesh
    file_bc = dolfin.File(out_path + 'boundary_markers.pvd')
    file_bc << boundary_markers

    hdf = dolfin.HDF5File(mesh.mpi_comm(), out_path + 'tetrahedral_mesh.h5', "w")
    hdf.write(mesh, "/mesh")
    hdf.write(boundary_markers, "/boundary_markers")
    hdf.close()

    print("=======================================")
    print(" > ¡Archivos de malla creados con éxito!\n")
    print(f" > Nodos: {mesh.num_vertices()}")
    print(f" > Celdas: {mesh.num_cells()}")
    print(f" > Facetas: {mesh.num_facets()}\n")

    return mesh

def assign_subdomains(mesh_path, terminals, out_path):
    
    # Read the mesh
    reference_mesh = io.read(mesh_path)
    ien = reference_mesh.cells_dict['tetra']
    xyz = reference_mesh.points
    
    # Store here the distances
    distances = []
    
    ### CELLS
    # Move to cells centers (coordinate triplet x,y,z for all cell center)
    cell_center = []
    for cell in ien:
        cell_center += [np.average(xyz[cell],axis=0)]
    cell_center = np.array(cell_center)
    
    # Compute distances between terminals and cell centers
    distances = []
    for p in terminals:
        # Change into numpy array
        p=np.array(p)
        # Compute distances
        distances += [np.linalg.norm(cell_center-p,axis=1)]
    
    # Compute distances
    distances = np.matrix(distances).T
    
    # Assign subdomain
    cell_subdomain = np.ravel(np.argmin(distances,axis=1))
    
    #Load the file with mesh and the boundaries
    mesh = dolfin.Mesh()
    
    # Load tetraedral mesh
    hdf = dolfin.HDF5File(mesh.mpi_comm(),out_path+'tetrahedral_mesh.h5', "r")

    hdf.read(mesh, "/mesh", False)
    #boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    #hdf.read(boundary_markers, "/boundary_markers")
    
    # Define dolfin data structure to handle the data
    Omega_i = dolfin.MeshFunction("size_t",mesh,mesh.topology().dim())
    
    # Send info to mesh function
    for cell, k in zip(dolfin.cells(mesh),cell_subdomain):
        Omega_i[cell]=k
    
    # Create file to be saved
    Omega_i_file = dolfin.File(out_path+"Omega.xml.gz")
    Omega_i_file << Omega_i
    
    Omega_i_file = dolfin.File(out_path+"Omega.pvd")
    Omega_i_file << Omega_i

    hdf.close()
    # mesh here is returned to be used afterwards, it is not critical
    return cell_subdomain 

def terminals_from_skel_mesh(skel_path):
    '''
    Generate a list of points from a skeleton mesh
    '''
    skel = io.read(skel_path)
    points = skel.points
    distal_points = skel.point_data['distal']
    distal_point_ids = np.arange(points.shape[0])[distal_points==1]
    return points[distal_point_ids]
    

def interpolate_intensities(X, image):

    intensities = []
    
    # load image
    im = nib.load(image)
        
    # retrieve affine and data
    affine = im.affine
    data = np.array(im.dataobj)
        
    # interpolation routine
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
            
        intensities.append(sum(bu[i0]*bv[j0]*bw[k0]*data[i,j,k]))

    return np.array(intensities)

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
                     create_visualization_vtu=False,
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


def process_geometry(subject_number, mesh_type, case="ARDSnet",
                     root = "/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/%s/%s/",
                     export_visualization_for_porosity = True,
                     skel_name="skel.vtu", output_folder = "FEniCS/",
                     ctissue_low = 2.00):
    
    # Generate codenames and path
    subject_code = "PIG%i"%subject_number
    root = root%(subject_code, case)
            
    # Declare paths in a format apt for execution in WSL
    path_to_mesh = root+"%s/"%mesh_type
    path_to_images = root+"NIFTI/"
    
    # Folder where the FEniCS data is going to be placed
    if output_folder[-1] != "/":
        output_folder += "/"
    output_path = path_to_mesh+output_folder
    
    # Path to a surface mesh where the diaphragm has been isolated
    diaph_path = path_to_mesh+"bc_info.vtu"
    diaph_flag = os.path.isfile(diaph_path)
    if not diaph_flag:
        print("Diaphragm info not found")
        diaph_path = None
    
    # Path to the airway skeleton under use
    skel_path = path_to_mesh+skel_name
    skel_flag = os.path.isfile(skel_path)
    
    # Path to the basic tetrahedral geometry
    tetra_path = path_to_mesh+"tetrahedral_base.vtu"
    
    # Path to a reference image
    nii_path = path_to_images+"Exp.nii.gz"
    
    # Do these folders exist?
    print(" > Testing involved folders")
    for folder in [path_to_mesh, path_to_images]:
        if os.path.isdir(folder):
            print("    > [%s] Found."%(folder.split("/")[-2]))
        else:
            print("    > [%s] Not found"%(folder.split("/")[-2]))
            os.mkdir(folder)
    
    # Generate output path
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        
    # Standard procedure to generate the FEniCS mesh
    print(" > Generating FEniCS geometries:")
    mesh = initialize_FEniCS_geometry(tetra_path,output_path,diaph_path)
    
    # If the airway skeleton is available
    if not skel_flag:
        print(" > Skeleton geometry not found")
        return None
        
    # Retrieve the airway terminals
    print(" > Skeleton geometry found:")
    print(" > Extracting airway terminals")
    terminals = terminals_from_skel_mesh(skel_path)
        
    # Generate the subdomains upon the FEniCS mesh
    print(" > Generating subdomains in FE mesh")
    fenics_mesh_path = path_to_mesh+output_folder+"mesh000000.vtu"
    subdomains = assign_subdomains(fenics_mesh_path, terminals, output_path)

    # Analyze generated subdomains
    print(" > Analyzing the subdomains")
    counter = []; volumes = []
        
    # Read the mesh
    fenicsmesh = io.read(fenics_mesh_path)
    xyz = fenicsmesh.points
    cells = fenicsmesh.cells_dict['tetra']
        
    # Initialize a counter
    nterminals = len(terminals)
    
    # For each terminal
    for j in range(nterminals):
            
        # Isolate the cells belonging to each terminal
        sd_mask = subdomains==j
        sd_cells = cells[sd_mask]
            
        # Determine volume in subdomain
        vol = 0.0
        for scell in sd_cells:
            ps = xyz[scell]
            vol += tetvol(ps)    
                
        # Add the number of elements per subdomain and its volume
        counter += [np.count_nonzero(sd_mask)]
        volumes += [vol]
            
    # Transform into numpy arrays
    volumes = np.array(volumes)
    counter = np.array(counter)   
       
    # Present analysis
    print("\n > SUBDOMAIN ANALYSIS: ")
    print("     > Number of null domains: %i/%i"%(np.count_nonzero(counter==0),len(terminals)))
    print("     > Minimum volume: %.0f"%(np.min(volumes)))
    print("     > Maximum volume: %.0f"%(np.max(volumes)))
    print("     > Volume ratio: %.1f"%(np.max(volumes)/np.min(volumes)))
    
    print("> Interpolating intensities:")
    # Load raw matlab mesh
    M = io.read(tetra_path)
    xyz = M.points
    
    # Load the finished mesh
    fenicsmesh = io.read(fenics_mesh_path)
        
    # Interpolate the end-expiratory intensities to the fenics mesh
    intensities = interpolate_intensities(xyz, nii_path)
        
    print("Mean intensity: %.2f"%np.average(intensities))
        
    # Saturate the intensities outside the porosity range
    intensities[intensities>0.0] = 0.0
    intensities[intensities<-1000.0] = -1000.0
        
    # Determine porosity through naive means
    porosity_pd = intensities/-1000.0
        
    # Correct porosity
    porosity_pd = correct_porosity(xyz, fenicsmesh.cells_dict['tetra'], porosity_pd)
    
    # Change from point data to cell data
    porosity_cd = []
    for cell in fenicsmesh.cells_dict['tetra']:
        porosity_cd += [np.average(porosity_pd[cell])]
        
    # Export a visualization mesh for the porosity field
    if export_visualization_for_porosity:
        print(" > Exporting porosity visualization mesh")
        test_output = io.Mesh(points=fenicsmesh.points,
                              cells=fenicsmesh.cells_dict,
                              point_data={"HU_EE":intensities,
                                              "Porosity_EE":porosity_pd},
                              cell_data={"Porosity_EE":[porosity_cd]})
            
        test_output.write(path_to_mesh+output_folder+"Porosity_Visualization.vtu")
    
    # Create mesh functions for k0 (initial permeability)
    k0 = dolfin.MeshFunction("double", mesh, 3) # Function for permeability
    phi0 = dolfin.MeshFunction("double",mesh,3) # Function for porosity
    c_tissue = dolfin.MeshFunction("double",mesh,3) # Function for tissue stiffness
        
    # Tissue stiffness
    c_tissue_max = ctissue_low*5
    c_tissue_min = ctissue_low
    # default permeability
    knormal = 10**2
    kmin = 1e-1
    phi0_mean = 0.50 # mean porosity at PIG4 00h Exp
        
    print(" > Generating the FEniCS mesh")
        
    # Iterate over mesh and set values
    for cell, phi in zip(dolfin.cells(mesh), porosity_cd):
        if phi > 0.1:
            k0[cell]=knormal*((phi/phi0_mean)**(2/3)) + kmin #define permeability as a porosity function
        else:
            k0[cell]=kmin
        phi0[cell] = phi
        c_tissue[cell] = assign_stiffness(phi,c_tissue_max=c_tissue_max,c_tissue_min=c_tissue_min)

    # Store to file
    mesh_file = dolfin.File(path_to_mesh+output_folder+"mesh.xml.gz")
    k0_file = dolfin.File(path_to_mesh+output_folder+"k0.xml.gz")
    phi0_file = dolfin.File(path_to_mesh+output_folder+"phi0.xml.gz")
    c_tissue_file = dolfin.File(path_to_mesh+output_folder+"c_tissue.xml.gz")

    mesh_file << mesh
    k0_file << k0
    phi0_file << phi0
    c_tissue_file << c_tissue
     
    # Final visualization of what has been computed
    permeabilityparaview = dolfin.File(path_to_mesh+output_folder+"permeabilityparaview.pvd", "compressed")
    permeabilityparaview << (k0)
    
    print("\n >>>>> END OF SCRIPT <<<<<")

if __name__ == '__main__':
    
    args = sys.argv
    subject_number = int(args[1])
    mesh_type = args[2]
    
    skel_name = "skel.vtu"
    output_folder = "FEniCS"
    
    if len(sys.argv)==4:
        print("Using prescribed ctissue: %s"%args[3])
        ctissue_low = float(args[3])
    else:
        ctissue_low = 2.10
    
    if mesh_type in ["coarse","medium-coarse","medium","medium-fine","fine","medium-fine2"]:
        print("Processing geometries for subject 'PIG%i' and mesh-type '%s'"%(subject_number,mesh_type))
        process_geometry(subject_number, mesh_type, ctissue_low=ctissue_low, output_folder=output_folder, skel_name=skel_name)
    else:
        print("Mesh type '%s' is invalid")
        
