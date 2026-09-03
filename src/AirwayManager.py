# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:12:06 2023

@author: angus
"""

import os
os.environ["OMP_NUM_THREADS"] = '1'

import nibabel as nib
import meshio as io
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
import trimesh
import pickle


def assign_subdomains_to_cells(cms, points, cells):
    '''
    assign a subdomain to every cell using a distance rule
    the subdomains are defined by triplets stored in a list ('terminals')
    '''
    # Store here the distances
    distances = []
    
    # Move to cells centers (coordinate triplet x,y,z for all cell center)
    cell_center = []
    for cell in cells:
        cell_center += [np.average(points[cell],axis=0)]
    cell_center = np.array(cell_center)
    
    # Compute distances between terminals and cell centers
    distances = []
    for p in cms:
        # Change into numpy array
        p=np.array(p)
        # Compute distances
        distances += [np.linalg.norm(cell_center-p,axis=1)]
    
    # Compute distances
    distances = np.matrix(distances).T
    
    # Assign subdomain
    return np.ravel(np.argmin(distances,axis=1))

def assign_subdomains_to_points(cms, points):
    '''
    assign a subdomain to every cell using a distance rule
    the subdomains are defined by triplets stored in a list ('terminals')
    '''
    # Store here the distances
    distances = []
    
    # Compute distances between terminals and cell centers
    distances = []
    for p in cms:
        # Change into numpy array
        p=np.array(p)
        # Compute distances
        distances += [np.linalg.norm(points-p,axis=1)]
    
    # Compute distances
    distances = np.matrix(distances).T
    
    # Assign subdomain
    return np.ravel(np.argmin(distances,axis=1))


def tet_volume(ps):
    ''' determine the volume of a tetrahedron'''
    ps_ = np.pad(ps,[(0,0),(0,1)])
    ps_[:,3] = 1.0
    return abs(np.linalg.det(ps_)/6)

def rewrite_surface_mesh(vtu, ply, surface_tag=2):
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
    writer  = io.Mesh(points=triangle_mesh.points, 
                  cells={'triangle':surface_cells})
    writer.write(ply)


def naive_raycast_cloud(surf_ply,point_structure,ns,direction,
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
    
    # initialize counters
    i = 0; j= 0; k= 0;
    
    # nested while loops (better than "for"s for memory handling)
    while i< ns[0]:
        print(i,ns[0])
        j = 0 # reset j
        while j<ns[1]:
            k = 0 # reset k
            while k<ns[2]:
                
                # retrieve point
                test_point = point_structure(i,j,k)
                
                l0 = surf_lung.ray.intersects_location([test_point],[direction])[2]
                l1 = surf_lung.ray.intersects_location([test_point],[-direction])[2]
                
                # only odd combinations imply an inside point
                if len(l0)%2 == 1 and len(l1)%2==1:
                    stored_points+=[test_point]
                # update counts 
                k+=1
            j+=1
        i+=1
    
    stored_points = np.array(stored_points)        
    
    # testing output
    if not vtuname is None:
        # generate object
        out =io.Mesh(points=stored_points,
                      cells={"vertex":np.array([[i,] for i in range(len(stored_points))])})
        # export
        out.write(vtuname)
    
    if not npzname is None:
        # save cloud as *.npz
        np.savez(npzname, cloud=stored_points)

    if return_cloud:
        return stored_points

def query_interior_point(point,surf_lung,direction=np.array([1.,0.,0.])):
   
    '''
    This function tests whether a given point is inside a region that is 
    defined through a surface mesh 'surf_lung', which is a trimesh surface
    object loaded as:
        
    surf_lung = trimesh.load(surf_ply)
    
    In this case, the mesh was a surface mesh represented as a *.ply file
    which is also used for a similar algorithm that uniformly distributes 
    points along the lung domain.
    
    The function rewrite_surface_mesh reads a FEniCS-generated boundary
    markers mesh and writes the corresponding surf_ply file.
    
    Returns a Boolean value that is True if the point is inside and False
    if is outside.
    '''

    l0 = surf_lung.ray.intersects_location([point],[direction])[2]
    l1 = surf_lung.ray.intersects_location([point],[-direction])[2]

    inside = len(l0)%2 == 1 and len(l1)%2==1
    
    return inside


# divide a cloud using pca + kmeans
def check_coherence_in_clouds(cloud, parent, plot = False, threshold=10.0):
    
    '''
    this method is used to make sure that there is some coherence in the cloud
    of points. if we can split our data in two bodies that are separated, keep
    the body that is closest to the parent branch and dismiss the others.
    '''
    
    
    # determine the distance to the parent point
    distances = np.linalg.norm(cloud-parent,axis=1).reshape(-1,1)
    # generate a clusterization
    clusterizer = k_means(distances,n_clusters=2, n_init=1)
    # retrieve tags
    tags = clusterizer[1]
    
    if plot:
        plt.hist(distances[tags==0],color='r')
        plt.hist(distances[tags==1],color='g')
        plt.show()

    
    # divide the clouds
    cloud0 = cloud[tags==0]
    cloud1 = cloud[tags==1]
    ids = np.arange(len(cloud))
    id0 = ids[tags==0]
    id1 = ids[tags==1]
    
    # determine the center of mass of each cluster
    cm0 = np.mean(cloud0,axis=0)
    cm1 = np.mean(cloud1,axis=0)
    
    # determine representative points from each cloud that are close to the other
    dist01 = np.linalg.norm(cloud0-cm1,axis=1)
    dist10 = np.linalg.norm(cloud1-cm0,axis=1)
    
    # global id of the points from each_cloud
    id_pmin0 = id0[np.argmin(dist01)] 
    id_pmin1 = id1[np.argmin(dist10)]
    
    # global points
    pmin0 = cloud[id_pmin0]
    pmin1 = cloud[id_pmin1]

    if plot:
        plt.scatter(cloud[:,0][tags==0],cloud[:,2][tags==0])
        plt.scatter(cloud[:,0][tags==1],cloud[:,2][tags==1],color="r")
        plt.scatter(pmin0[0],pmin0[2],marker='x',color='k')
        plt.scatter(pmin1[0],pmin1[2],marker='x',color='k')
        plt.show()

    # representative distance between clouds
    cloud_distance = np.linalg.norm(pmin0-pmin1)
    
    # if the distance is less than the threshold, then is likely that the 
    # clusters are direct neighbors and they don't need to be divided
    if cloud_distance < threshold:
        return cloud
    else:
        # determine a representative distance to the parents from each cloud
        distance_to_parent0 = np.mean(np.linalg.norm(cloud0-parent,axis=1))
        distance_to_parent1 = np.mean(np.linalg.norm(cloud1-parent,axis=1))
        
        if distance_to_parent0 > distance_to_parent1:
            return cloud1
        else:
            return cloud0

# new algorithm, closer to the one in the papers, but:
# * still using the internal nodes instead of the uniformly distributed ones.
# * we won't use initial parent nodes from an skeletonization
# * we will use the center of mass as the initial terminal node

def split_cloud(cloud, parent, surf_lung,radius_of_interest=None, 
                kmeans_coherence=True, nattempts=3):
    
    # check if the cloud of points can be separated in two distincts regions,
    # one apart form the other, if so, return a reduced cloud that contains
    # the closest region.
    if kmeans_coherence:
        cloud = check_coherence_in_clouds(cloud, parent)
    
    # If a threshold length has been declared, we will reduce the cloud 
    # accordingly. This is particularly important for the first generations 
    # of the algorithms to avoid including points in distal regions of the 
    # lung.
    if not radius_of_interest is None:
        # compute the distance to the cloud
        distance = np.linalg.norm(cloud-parent,axis = 1)
        # remove unwanted points from the cloud
        cloud = cloud[distance < radius_of_interest]
    
    # Use the first principal component of the point cloud to define a normal 
    # for a plane fixed at the parent point
    pca = PCA()
    pca.fit(cloud)
    normal = pca.components_[0,:]
    
    # Determine the distances from the cloud to the plane
    distances = np.dot(cloud-parent,normal)
    # Use the plane to divide into cloud in two groups
    mask0 = distances>0.0
    masks = [mask0,np.logical_not(mask0)]
    
    children = []    
    lengths = []
    
    for mask in masks:
        
        # Determine the center of mass of each region
        cm = np.mean(cloud[mask],axis=0)
        # Determine a vector starting at the parent point and ending at cm0 
        vec = cm - parent
        # Determine its length and its direction
        dvec = np.linalg.norm(vec)
        nvec = vec/dvec
        
        for i in range(nattempts):
            
            # Determine a new length, 0.4 is the parameter used in literature,
            # see Tawhai et al (2004), Bordas et al (2015), Nousias et al (2020)
            branch_length = (0.4*(0.5)**i)*dvec
            
            # Determine a new branch (starts at parent_cm)
            branch = nvec*branch_length
            # Determine the position of the endpoint of the branch
            child = parent+branch
                            
            if query_interior_point(child, surf_lung):
                # if the child is within the mesh, exit this internal loop
                children += [child]
                lengths += [branch_length]
                break 
            else:
                print("Found a child that left the lung!")
            
    
    # Return branch end-point and length; also the masks
    return children, lengths

def corr_to_mesh(corr_txt):
    
    '''
    Read the txt file that holds the correspondance list generated in CGAL,
    it must be paired with the corresponding skel file.
    
    Remember that this values exist in a voxel-space, and they must be 
    multiplied by an image affine matrix to be expressed in the real world
    coordinates.

    '''
    
    str2float = lambda x: float(x)
    
    manager = []
    # Open file "skel_txt"
    f = open(corr_txt)
    for x in f:
        # Remove blank space
        y = x[:-1]
        # Divide coordinates
        z = y.split(" ")
        # Extract points
        p0 = np.array(list(map(str2float,z[1:4])))
        p1 = np.array(list(map(str2float,z[5:8])))
        # Append to list
        manager += [[p0,p1]]
    f.close()

    # Process the correspondance list
    inners = []; outers = []
    for p0, p1 in manager:
        inners += [p0]
        outers += [p1]
    
    inners = np.array(inners)
    outers = np.array(outers)
    
    return inners, outers


def skel_to_mesh(skel_txt,skel_mesh=None,tol = 1e-8):
    
    '''
    
    Read a txt file containing a valid skeletonization and return the point
    and element list. If skel_mesh is a path, the mesh will be saved there as
    well.
    
    skel_txt is the full path to a txt file that holds the geometrization
    such as 'skel-sm.polylines.txt'.
    
    skel_mesh should be a path to be used to save the skeleton mesh directly
    from the skeletonization, in voxel-coordinates
    
    '''
    
    str2float = lambda x: float(x)

    manager = []
    # Open file "skel_txt"
    f = open(skel_txt)
    for x in f:
        # Remove blank space
        y = x[:-1]
        # Divide coordinates
        z = y.split(" ")
        # Extract points
        p0 = np.array(list(map(str2float,z[1:4])))
        p1 = np.array(list(map(str2float,z[4:7])))
        # Append to list
        manager += [[p0,p1]]
    f.close()
    
    pointlist = []
    elem = []
    for e,(p0,p1) in enumerate(manager):
                
        if e == 0:
            pointlist += [p0,p1]
            elem += [[0,1]]
        else:
            np_pointlist = np.array(pointlist)     
            # check if the point exists in the point list
            
            p0_in_pointlist = np.linalg.norm(np_pointlist-p0,axis=1)<tol
            p1_in_pointlist = np.linalg.norm(np_pointlist-p1,axis=1)<tol

            if np.any(p0_in_pointlist): # if the point was already there
                # we do not need to add it again, but we will store its id
                loc = np.argwhere(p0_in_pointlist)
                if len(loc)>1: 
                    raise Exception("More than one match for the point")
                else:
                    id0 = loc[0][0]
            else: # if the point wasn't in the list
                pointlist += [p0]
                id0 = len(pointlist)-1
            
            # Repeat for p1
            if np.any(p1_in_pointlist): # if the point was already there
                # we do not need to add it again, but we will store its id
                loc = np.argwhere(p1_in_pointlist)
                if len(loc)>1: 
                    raise Exception("More than one match for the point")
                else:
                    id1 = loc[0][0]
            else: # if the point wasn't in the list
                pointlist += [p1]
                id1 = len(pointlist)-1
            
            # ADd the point to the element list
            elem += [[id0,id1]]
    
    if skel_mesh != None:
        mesh = io.Mesh(points=np.array(pointlist), 
                cells={"line":np.array(elem)})
        mesh.write(skel_mesh)
    
    return np.array(pointlist), np.array(elem)

def classify_skel_nodes(points,elem):
    
    ''' 
    read the nodes from a skeletonization line mesh and classify some
    notable points as intlet, endpoints, and bifurcations.
    '''
    
    
    freq_counter = np.zeros(len(points),dtype=int)
    
    for (i,j) in elem:
        freq_counter[i] += 1
        freq_counter[j] += 1
    
    freqs = np.unique(freq_counter)
    
    point_ids = np.arange(len(points)) # airway point ids
    
    terminals = point_ids[freq_counter==1]
    connections = point_ids[freq_counter>2]
    
    # determine the inlet terminal (top of the trachea)
    airway_inlet = terminals[np.argmax(points[:,2][terminals])]
    
    
    # exclude the top of the trachea from the terminals array
    terminals_purged = terminals[terminals != airway_inlet]
    
    # dictionary with skeleton data
    descr = {"inlet_id":airway_inlet,
             "endpoints_ids":terminals_purged,
             "bifurcations":connections,
             "nconnections":freqs,
             }
    
    return descr


def transform_skel(mat_lung,vtu_lung,nifti_lung,aw_points,aw_elem, 
                   outname=None, scale_airways_with_affine=True,
                   override_offset=False, positive_affine=True):
    '''
    correct the skeletonization using some reference geometries
    anything processed in matlab is by default without world coordinated, it 
    exists in a voxel-space. multiplication by the affine diagonal elements
    will assign metric units. 
    fenics meshes are centered in the geometric center, and thus, they extend
    towards every axis in symmetric lengths. 
    the complete correction transforms a wild voxel-space skeletonization into 
    a well-centered skeleton mesh compatible with the fenics lung mesh.
    '''
    img = nib.load(nifti_lung)
    affine = np.array(img.affine)
    
    # Create a vector to transform the points
    vec_affine = np.array([affine[0,0],affine[1,1],affine[2,2]]) # do not use abs!
    print(vec_affine)
    if positive_affine:
        vec_affine=np.abs(vec_affine)
       # vec_affine[2] *= -1
    print(vec_affine)
    
    if scale_airways_with_affine:
        aw_points = aw_points*vec_affine 
    
    # The FEniCS mesh (tetrahedral) is centered at the origin while the airway
    # is in another metric space. We will use the original tetrahedral mesh and
    # a FEniCS input mesh to determine the displacement
    points_ref = loadmat(mat_lung)['tetnode']*vec_affine
    
    # Load the FEniCS mesh now
    
    mesh_fem = io.read(vtu_lung)
    points_fem = mesh_fem.points
    
    # Compute offset between the meshes
    if not override_offset:
        offset = np.mean(points_ref,axis=0) - np.mean(points_fem,axis=0)
    else:
        offset = np.zeros((1,3))

    if not outname is None: 
        
        mesh = io.Mesh(points=aw_points-offset,cells={"line":aw_elem})
        mesh.write(outname)
    
    return aw_points-offset

def distribute_point_cloud(surf_ply, vtuname=None, dd=1.0,  alv_density=1e-3, 
                           npzname=None, return_cloud=False, 
                           raycast_direction=np.array([1,0,0])):
    
    '''
    distribute points in a surface mesh according to an alveolar (node) 
    density ('alv_density').
    this employs a ray casting algorithm as implemented in trimesh which
    can be improved (as it is too linear!)
    '''
    
    # read mesh and extract points
    mesh = io.read(surf_ply)
    points = mesh.points
    
    # Measure lung bounds
    xmin = np.min(points[:,0])-dd; xmax = np.max(points[:,0])+dd;
    ymin = np.min(points[:,1])-dd; ymax = np.max(points[:,1])+dd;
    zmin = np.min(points[:,2])-dd; zmax = np.max(points[:,2])+dd;
    ls = [xmax-xmin,ymax-ymin,zmax-zmin]
    
    # Define bounding box
    v_alv = 1./alv_density # volume for one alveoli
    l_alv = v_alv**(1./3.) # length for an box-like alveoli #
        
    # number of points per length
    ns = [round(li/l_alv) for li in ls]
    
    # representation of the points
    def point_from_cloud(i,j,k):
        if i>(ns[0]-1) or j>(ns[1]-1) or k>(ns[2]-1): 
            return None
        else:
            return (xmin+(i+0.5)*l_alv,ymin+(j+0.5)*l_alv,zmin+(k+0.5)*l_alv)

    return naive_raycast_cloud(surf_ply, point_from_cloud, ns, 
                               direction=raycast_direction,
                               vtuname=vtuname,
                               npzname=npzname, 
                               return_cloud=return_cloud)


class Branch:
    
    '''
    base structure to handle a 'Branch', a tuple of points connected through
    a straight line
    
    gets to conserve some useful skeletonization data and it should be expanded
    to generate a graph, and branching order.
    '''
    
    def __init__(self, start_id, start_point, branch_id, path):
        self.id0 = start_id # id in point list
        self.id1 = None # id in point list
        self.p0 = start_point # rwc point
        self.p1 = None # rwc point
        self.abs_length = 0.0 # length following complete chain of points
        self.internal_points = [start_id] # complete chain of points
        self.line_length = 0.0
        self.growing = True
        self.id = branch_id # branch-id
        self.path = path # list holding the chain of branches until here (+id)
        self.order = None # placeholder for the branch's strahler order
        self.radius = 1.0 # placeholder
        
        if len(path)>1: # exception for the root
          self.parent = path[-2] # points towards the parent branch id
        else:
          self.parent = None # this only happens to the root
          
        self.children = [] # store the children branch ids
        
        self.bdof = None
        self.vdof0 = None
        self.vdof1 = None
        
        self.re = None
        
    def close(self,end_id, end_point):
        # end line
        self.id1 = end_id # end point id
        self.p1 = end_point # end rwc point
        self.line_length = np.linalg.norm(self.p1-self.p0) # point-to-point length
        self.growing = False
    
    def extend(self,pid,delta):
        # extend line
        self.internal_points += [pid] 
        self.abs_length += delta
        self.growing = True
        
    def set_order(self, order):
        # set the strahler order as found in a tree method
        self.order = order
        
    def assign_vertex_dof(self, vdof0, vdof1):
        # Store the degree of freedom associated to the end points of a branch
        # in a processed mesh to form a system of equations associated to the 
        # pressure variables.
        # airway tree
        self.vdof0 = vdof0
        self.vdof1 = vdof1
        
    def assign_branch_dof(self, bdof):
        # Store the degree of freedom associated to a branch itself in a 
        # processed mesh to form a system of eqs. associated to flow variables
        self.bdof = bdof
    
    def set_children(self, children):
        # Assign the children list
        self.children = children
        
    def set_radius(self,radius):
        # set the radius
        self.radius = radius
        
    def set_re(self, re):
        # store a referential Reynolds number associated to the branch
        self.re = re
    
class Tree:
    
    def __init__(self,seed,points,elem, branch_length_threshold=2.00,
                 minimum_subdomain_size=1, rho = 1e-9, mu = 1.825e-8,
                 trimmed = False):
        
        self.seed = seed
        self.seen = []
        self.points = points
        self.elem = elem
        self.branches = {}
        self.skeleton_branches = []
        self.synthetic_branches = []
        self.seen_points = []
        self.seen_elems = []
        self.active_branch = 0
        self.growing_branches = []
        self.distal_points = []
        self.distal_branches = []
        self.branch_length_threshold = branch_length_threshold
        self.minimum_subdomain_size = minimum_subdomain_size
        self.mesh=None
        self.linear_system_A = None
        self.linear_system_f = None
        self.mu = mu
        self.rho = rho
        self.trimmed = trimmed
        
    def activate_seed(self):
        
        '''
        
        initiate processing of the skeletonization
        
        '''
        
        # Find which elements contain the seed
        self.active_node_id = self.seed
        # List of elements and positions
        where = np.argwhere(self.elem == self.active_node_id)
        
        # Confirm the seed is an external point
        if len(where) == 1:
            pass
        else:
            raise Exception("Not trachea!")
        
        # Find which elements contain the seed
        where = np.argwhere(self.elem == self.active_node_id)
        
        # Retrieve the element id and the location of the tracked node
        elem_id, pos = where[0]
        
        # Declate the other node as active node
        self.active_node_id = self.elem[elem_id,1-pos]
        self.active_node = self.points[self.active_node_id]

        # Add the current element id to the seen_elems list
        self.seen_points += [self.seed, self.active_node_id]
        self.seen_elems += [elem_id]
        
        
        # Default initiation of main branch
        self.branches.update({self.active_branch:Branch(self.seed, 
                                                        self.active_node,
                                                        self.active_branch,
                                                        [self.active_branch])})
        # List of skeleton-associated branches
        self.skeleton_branches += [self.active_branch]
        
        # List of growing branches when analyzing the skeleton mesh
        self.growing_branches += [self.active_branch]
        
    def extend_node(self, verbose=False):
        
        '''
        systematization of the branch processing, different paths depending on
        how many elements does the active_node resides in.
        
        if 1 element => Inlet or distal node
           2 elements => Within the branch, continue growing
           3 elements => Bifurcation, close this branch and create new others
        
        '''
        
        # Find where a point appears in the element list
        matches = np.argwhere(self.elem == self.active_node_id)    
        
        # Count how many times does the point appear in the element lists
        nmatches = len(matches)
        
        if nmatches==2: 
            # This is within a branch - regular advancing front
        
            # For each element in the where-list
            for match in matches:
                
                # Retrieve the element id and the local point position
                elem_id, pos = match
                
                # If we had already seen the element skip it
                if elem_id in self.seen_elems:
                    continue
                
                else: 
                    
                    # Previously unseen element
                    self.seen_elems += [elem_id]
                    
                    # Temporarily store the unseen node and its id
                    unseen_node_id = self.elem[elem_id,1-pos]
                    unseen_node = self.points[unseen_node_id]

                    # Determine the length of the new element
                    l = np.linalg.norm(unseen_node-self.active_node)
                    
                    # Update the active point and its id
                    self.active_node = unseen_node
                    self.active_node_id = unseen_node_id
                    self.seen_points += [unseen_node_id]

                    # Update the branch
                    self.branches[self.active_branch].extend(unseen_node_id,l)
                    
       
        elif nmatches == 1:
            
            if verbose :
                print("Found a distal node at branch #%i"%self.active_branch)
            # Only possible at the seed or at a distal/terminal node
            
            # Close this branch
            self.branches[self.active_branch].close(self.active_node_id,
                                                    self.active_node)
            
            # Store distal point and branch
            self.distal_points += [self.active_node]
            self.distal_branches += [self.active_branch]
            
            # Remove this branch from the active_branches list
            self.growing_branches.remove(self.active_branch)

        else:
            
            if verbose :
                print("Found a bifurcation at branch #%i"%self.active_branch)
            
            # This occurs when a branch is ending and a pair (or more) of
            # branches are going to be initiated.
            
            # Close the active branch
            self.branches[self.active_branch].close(self.active_node_id,
                                                    self.active_node)

            # Remove this branch from the active_branches list
            self.growing_branches.remove(self.active_branch)


            # For each of the elements matching out bifurcation node
            for match in matches:
                
                # Retrieve the element id and the local point position
                elem_id, pos = match  
                                
                # If we had already seen the element skip it
                if elem_id in self.seen_elems:
                    continue
                else:
                    
                    # Compute new branch data
                    branch_id = max(self.branches.keys()) + 1
                    path = self.branches[self.active_branch].path + [branch_id]
                    
                    if verbose :
                        print("Creating a new branch (#%i)"%branch_id)
                    
                    # Add new child branch info to the parent branch
                    self.branches[self.active_branch].children += [branch_id]
                    
                    # Create new branch
                    self.branches.update({branch_id:Branch(self.active_node_id,
                                                           self.active_node,
                                                           branch_id,
                                                           path)})
                    
                    # Add the branch to the skeleton-associated branches list
                    self.skeleton_branches += [branch_id]
                    
                    # Append this branch to the active_branches list
                    self.growing_branches += [branch_id]
                    
                    # Extend new branch
                    unseen_point_id = self.elem[elem_id, 1-pos]
                    unseen_point = self.points[unseen_point_id]
                    delta = np.linalg.norm(unseen_point-self.active_node)
                                
                    # Extend the new branch into the unseen point direction
                    self.branches[branch_id].extend(unseen_point_id,delta)
                    
                    # Declare this element and point as seen
                    self.seen_elems += [elem_id]
                    self.seen_points += [unseen_point_id]
                
        
    def grow_tree(self, verbose=False):
        
        ''' 
        cyclic application of the growing algorithm
        
        '''

        # While there are still some branches to be grown
        while len(self.growing_branches)>0:
            
            # Show which one is the branch growing
            if verbose: print(self.active_branch)
            
            # While that branch is still growing
            while self.branches[self.active_branch].growing:
                # Extend it!
                self.extend_node()
            
            # If we left that loop, we need to pick up a new active branch
            # and repeat.So if we still have some growing branches:
            if len(self.growing_branches)>0:
                self.active_branch = self.growing_branches[0]  
                self.active_node_id = self.branches[self.active_branch].internal_points[-1]
                self.active_node = self.points[self.active_node_id]
                
            # Now repeat the loop until the growing_branches list is empty!
            
    
    def process_mesh(self, outname=None, include_order=False):
        '''
        Transform the current mesh into a vtk or vtu file that can be opened
        in Paraview or another similar visor.
        The include_order option allows the exportation of the field order
        within each branch if it has been already computed. This will appear
        as a cell_data within the meshio "Mesh" data structure.
        '''

        # data holder for cell data
        cell_data = {}
        
        # if required, gather order data from every branch
        if include_order: order = np.zeros(len(self.branches.keys()))
        distal_nodes = []
        radii = np.zeros(len(self.branches.keys()))
        lengths = np.zeros(len(self.branches.keys()))
                
        for e,b in enumerate(self.branches):
            
            distal = b in self.distal_branches
                        
            # Pick a branch
            line = self.branches[b]
            line.assign_branch_dof(e)
            
            if include_order: order[e] = line.order    
            radii[e] = line.radius
            lengths[e] = line.line_length
            
            # Exception for the initial branch
            if b == 0:
                # Create the data structures
                # Initiate a pointlist (list and numpy array)
                pointlist = [line.p0, line.p1]
                np_points = np.array(pointlist)
                # Element table
                elems = [[0, 1]]
                line.assign_vertex_dof(0,1)

            else:
                # If we are not in the initial branch
                
                # Check if the proximal point was already stored using a 
                # distance norm compared with a tolerance
                pos_arr = np.linalg.norm(np_points-line.p0,axis=1)<1e-8
                
                # If there is a match, we already had the point
                if np.count_nonzero(pos_arr)==1:
                    # Retrieve its position
                    pos0 = np.argwhere(pos_arr)[0,0]
                    
                # If the point is repeated plenty of times... error!
                elif np.count_nonzero(pos_arr)>1:
                    raise Exception("Repeated point at pointlist when writing tree.")
                
                # If there was no match, then we assign the point to the pointlist
                else:
                    
                    pointlist += [line.p0] # add point to the pointlist
                    pos0 = len(pointlist)-1 # its id
                    np_points = np.array(pointlist) # update np_points

                # Check if the second point was already stored 
                pos_arr = np.linalg.norm(np_points-line.p1,axis=1)<1e-8
                
                # If there is only a matching point, that's OK
                if np.count_nonzero(pos_arr)==1:
                       pos1 = np.argwhere(pos_arr)[0,0]
                       
                # If the point is repeated plenty of times... error!
                elif np.count_nonzero(pos_arr)>1:
                    raise Exception("Repeated point at pointlist when writing tree.")
                    
                else:
                    pointlist += [line.p1] # add point to the pointlist
                    pos1 = len(pointlist)-1 # its id
                    np_points = np.array(pointlist)
                
                # Extend the element list
                elems += [[pos0,pos1]]
                # Store the global degree of freedom number in the branch
                line.assign_vertex_dof(pos0,pos1)
                # If the node is distal, keep it in a separate list
                if distal:
                    distal_nodes += [pos1]
                
        # Close both the element and the point list
        points = np.array(pointlist)
        elems = np.array(elems)
        
        distal_vertex = np.zeros(len(points), dtype=int)
        distal_vertex[distal_nodes] = 1
        
        # Determine the distal branches position in the cell array
        branches = np.array(list(self.branches.keys()))
        distal_ids = []
        for key in self.distal_branches:
            distal_ids += [np.argwhere(branches==key)[0][0]]
        distal_ids = np.array(distal_ids)
        
        distal_branch = np.zeros(len(points)-1, dtype=int)
        distal_branch[distal_ids] = 1
        
        
        self.mesh = {"xyz":points,"ien":elems,"distal":distal_vertex}    
        
        if include_order: 
            cell_data.update({"order":[order]})
            self.mesh.update({"order":order})
            
        cell_data.update({"radius":[radii], "length":[lengths],
                          "distal":[distal_branch]} )
        self.mesh.update({"radius":radii, "length":[lengths],})
        
        if not outname is None:

            obj = io.Mesh(points=points,cells={'line':elems},cell_data=cell_data,
                          point_data={"distal":distal_vertex})
            
            obj2 = io.Mesh(points[distal_nodes], 
                                cells={"vertex":np.array([[i,] for i in  
                                                range(len(distal_nodes))])})
        
            obj.write(outname)
            outname2 = outname.split(".vtu")[0]+"_distal_vertex.vtu"
            obj2.write(outname2)


    def radius_from_skeletonization(self, skel_txt, corr_txt, nifti_ref,
                                    quantile_criteria=0.75):
        
        '''
        This method reads an skeletonization, a correspondence txt, both generated
        in CGAL, and a nifti image used to generate real-world coordinates. 
        
        The radius of each branch is estimated pointwise and then averaged for the
        whole branch, using length-related weights for each subsegment.
        
        '''
                     
        # Read correspondance list
        inners, outers = corr_to_mesh(corr_txt)
        # Read the skeletonization
        points, elem = skel_to_mesh(skel_txt,skel_mesh=None)
        
        
        img = nib.load(nifti_ref)
        affine = np.array(img.affine)
        
        # Create a vector to transform the points
        vec_affine = np.array([affine[0,0],affine[1,1],affine[2,2]]) # do not use abs!
        
        # Scale the points to real-world coordinates
        points *= vec_affine
        inners *= vec_affine
        outers *= vec_affine
        
        # Take a branch and read its skeleton points
        for bid in self.skeleton_branches:
            
            # Extract the points associated with the branch
            internal_points = self.branches[bid].internal_points # these are ids
            radii = []; weights = []
            
            # for each internal points
            for pos in range(len(internal_points)):
                center = points[internal_points[pos]]
                
                # Find the associated points in the correspondance list
                mask = np.linalg.norm(inners-center,axis=1)<1e-5
                
                # Extract the corresponding mantle (outer and corresponding set of points)
                mantle = outers[mask]
                
                # Select the point of interest and its neighbours to draw a line
                if pos == 0:
                    line_candidate_ids = [internal_points[pos], internal_points[pos+1]]
                elif pos == (len(internal_points)-1):
                    line_candidate_ids = [internal_points[pos-1],internal_points[pos]]
                else:
                    line_candidate_ids = [internal_points[pos-1],internal_points[pos],
                                          internal_points[pos+1]]
                    
                centerline_points = points[line_candidate_ids]
                line_mean = centerline_points.mean(axis=0)
                
                # best fit line in the least squares sense
                _, _, vv = np.linalg.svd(centerline_points - line_mean)
                # slope or direction
                m = vv[0]
                
                # determine the distance from each point in the mantle and a line that stems
                # from the center and moves in the direction
                distance_to_reference = np.linalg.norm(np.cross(center-mantle,m)/np.linalg.norm(m),axis=1)
                
                # extract a characteristic radius
                try:
                    characteristic_radius = np.quantile(distance_to_reference,quantile_criteria)
                except:
                    print("[bid %i] WARNING! Radius not found"%bid)
                    print(" > Skipping this internal point")
                    continue
                    characteristic_radius = np.nan
                
                # length-associated weight 
                weight = np.linalg.norm(centerline_points[-1]-centerline_points[0])
                
                # Add radii and length (weights) to handle
                radii += [characteristic_radius] 
                weights += [weight]
            
            # Compute weighted average
            weights = np.array(weights); radii = np.array(radii)
            if len(radii)==0:
                print("[WARNING] Radius could not be determined for bid = %i"%bid)
            
            weighted_radius = np.sum(radii*weights)/np.sum(weights)
            self.branches[bid].radius = weighted_radius
        


    def generate_sythetic_branches(self, template_cloud, surf_ply,
                                   kmeans_coherence=False):
        
        '''
        Each application of this method will generate a new generation of
        branches, replacing a previous algorithm that generated branches 
        externally. 
        
        The template cloud is a uniform distribution of points along a 
        surface that is created externally using the function 
        naive_raycast_cloud contained in the same library.
        
        surf_ply is also used for raycasting but for the query of individual
        points when generating synthetic branches. This is required to avoid
        branches growing and extending outside the lung.
        
        If a branch extends outside the lung, its length should be reduced 
        a few times until it lies within the domain. If after some attempts 
        this does not work or its length is too small, the branch should be 
        deleted.
        
        '''
        
        # Load surface mesh for raycasting
        surf_lung = trimesh.load(surf_ply)
        
        # Generation storage
        active_points = []; length_mask = [];
        
        # Check every distal branch
        for b in self.distal_branches:
            
            # Determine the branch length; 
            #·later on , if L < branch_length_threshold, deactivate branch
            length = self.branches[b].line_length
            
            # Generate an activation mask that discriminates whether the branch
            # is larger than a minimum required length (branch_length_threshold)
            length_mask += [length > self.branch_length_threshold]
            # Store the corresponding points
            active_points += [self.branches[b].p1]
          
        # Transform into numpy array to ease manipulation
        active_points = np.array(active_points)
        length_mask = np.array(length_mask)
        active_branches = np.array(self.distal_branches)
        
        # Determine subdomains associated to each active point
        subdomains = assign_subdomains_to_points(active_points, template_cloud)    
        
        # Determine the number of points assigned to each subdomain
        bincount = np.bincount(subdomains)
        
        # Only pass through the following routine the subdomains with more than 
        # 1 point
        binsize_mask = bincount > self.minimum_subdomain_size
        
        # Unite the activation mask with the binmask
        activation_mask = np.logical_and(length_mask, binsize_mask)
        
        for e, token in enumerate(zip(active_branches,active_points,activation_mask)):
            
            # Unravel token
            branch_no, parent, activation = token
            
            # Skip points with activation == False
            if not activation: 
                # A deactivated branch not bifurcate any longer, so it will 
                # remain a distal branch forever 
                # Deactivate branch
                # print("Branch #%i deactivated"%branch_no)
                #.distal_branches.remove(branch_no)
                continue
            
            # Reduce the points to the subdomain
            # We only see the points associated with this subdomain
            cloud = template_cloud[subdomains==e]
            
            # Generate children
            children, lengths = split_cloud(cloud, parent, surf_lung,
                                            radius_of_interest=50.0,
                                            kmeans_coherence=kmeans_coherence)
            
            # Retrieve the parent branch
            parent_branch = self.branches[branch_no]
            parent_path = parent_branch.path
            
            # For each of the childrens
            for child in children:
                
                # Determine the new branch information
                child_branch_id = max(self.branches.keys())+1
                child_path = parent_path+[child_branch_id]
                
                # Let the parent branch know it is a parent (congratulations?)
                parent_branch.children += [child_branch_id]
                
                # Create and close branch
                child_branch = Branch(None, parent,child_branch_id,child_path)
                child_branch.close(None,child)
                
                # Add new branch to the tree
                self.branches.update({child_branch_id:child_branch})
                
                # Add branch to the synthetic branch list
                self.synthetic_branches += [child_branch_id]
                
                # Define the children branches as distal
                self.distal_branches += [child_branch_id]
                
            # Remove the parent from the active_branches list in the tree
            if len(children)>0: self.distal_branches.remove(branch_no)    
      
    def set_strahler_order(self, verbose=False):
        '''
        Method to define the strahler order of the branches that conform the tree
        and the tree itself. The strahler order of the tree is the strahler order
        of its root
        
        If the branch has no children, its Strahler number is one
        
        If the branch has one child with Strahler number i, and all other 
        children have Strahler numbers less than i, then the Strahler number 
        of the branch is i again.
        
        If the branch has two or more children with Strahler number i, and no 
        children with greater number, then the Strahler number is i+1.
        '''
        
        # initialize an array to store the visited branches
        visited_branches = []
        pending_parents = []
        
        # first, set the distal branches' strahler order to 1 by definition
        # and keep track of their parents
        for leaf in self.distal_branches:
            
            # Distal branches are 1 by definition
            self.branches[leaf].set_order(1)
            
            # store the branch id to avoid vising them again and to make sure the 
            # parents are visited only whenever all their children have been
            # visisted already.
            visited_branches += [leaf]
            
            # store the parent branch id in a list
            parent = self.branches[leaf].parent
            
            # if the parent wasn't in pending_parents, add it
            if parent not in pending_parents: pending_parents += [parent]
        
        # for as long as we have not cleared all the parents ...
        while(len(pending_parents)>0):
            
            # report the number of pending parents
            # if stuck, something is wrong
            if verbose:
                print(len(pending_parents))
                print(pending_parents)
            
            # move through every parent
            for leaf in pending_parents:
                
                # check if all their children have been visited, if true, 
                # proceed, otherwise skip it until a next iteration of the 
                # major while loop
                
                children_approved = True # auxiliary flag
                for child in self.branches[leaf].children:
                    if child not in visited_branches:
                        children_approved = False
                
                # use the flag children_approved to skip this leaf if necessary
                if not children_approved: 
                    continue
                    
                # if we are here, then we passed the children test, we can
                # determine the strahler order
                children_strahler = []
                for child in self.branches[leaf].children:
                    children_strahler += [self.branches[child].order]
                
                # depending on the amount of children
                if len(children_strahler)==2:
                    # now decide the parent's strahler order
                    if children_strahler[0]==children_strahler[1]:
                        parent_strahler = children_strahler[0] + 1
                    else:
                        parent_strahler = max(children_strahler)
                # there might be a branch with only one children, if that is 
                # the case just repeat the child's strahler
                elif len(children_strahler)==1:
                    parent_strahler = children_strahler[0]
                #three or more children is unexpected, let's just raise an exception
                else:
                    raise Exception("Problems at Strahler order assignation for branch %i"%leaf)
                
                # assign strahler to parent
                self.branches[leaf].set_order(parent_strahler)
                
                # update tracking lists
                visited_branches += [leaf]
                
                pending_parents.remove(leaf)
                
                if leaf != 0: # avoid the root
                    leaf_parent = self.branches[leaf].parent
                    # add to the list only if it wasn't there already
                    if leaf_parent not in pending_parents:
                        pending_parents += [self.branches[leaf].parent]


    def assign_synthetic_radii(self, RbS=2.0, RdS = None):
        '''
        
        Assigns a radius to each synthetic airway using a rule originally 
        proposed by Tawhai et al. (2004) "CT-based geometry analysis and ...".
        
        Note that RdS is a value suggested for perfectly symmetric branching
        patterns. They say that asymetric trees have lower RbH and higher RbS.
        
        It is assumed that RdS = RbS**(1/3) as Tawhai  did. This results in
        RdS = 1.260. Bordas et al. (2015) "Development and analysis of..." used
        RdH = 1.150 (based on Horsfield order) .
    
    
        '''
        # Read trachea's relevant properties
        trachea_diameter = self.branches[0].radius*2.0
        highest_order = self.branches[0].order
        
        # If no value for RdS is provided, do the same as Tawhai
        if RdS is None:
            # Compute RdS 
            RdS = RbS**(1/3)
            
        # Assign radius to the synthetic branches
        for bid in self.synthetic_branches:
            branch = self.branches[bid]
            logD = (branch.order-highest_order)*np.log(RdS)+np.log(trachea_diameter)
            branch.set_radius(np.exp(logD)*0.5)


    # Trim tree function
    def trim_generation(self):
        
        '''
        Generate a new airway tree by removing every distal branch and 
        resolving everything involved to simplify the complexity of the tree.
        '''
        
        # Extract the initial data from the 
        seed = self.seed
        points = self.points
        elem = self.elem
        branch_length_threshold = self.branch_length_threshold
        
        new_tree = Tree(seed,points,elem,
                        branch_length_threshold=branch_length_threshold,
                        trimmed=True)
        
        # Placeholders
        distal_candidates = [] # holds the potential new distal branches id
        delection_targets = [] # holds the branches that are going to be deleted
         
        # Move through every branch in the original airway tree and find the parents
        # of the old distal branches. Those parents will become the distal branches
        # of the new trimmed tree. 
        # The previous distal branches are going to be deleted.
        for b in self.branches:
            
            if b in self.distal_branches:
                
                # copy the parent which will become the new distal airway        
                distal_candidates += [self.branches[b].parent]
                delection_targets += [b]
        
        # With this, a list of all the candidates for the new tree has been created
        distal_candidates = list(np.unique(distal_candidates))    
        
        
        # Generate a dictionary with the information regarding parents (distal 
        # candidates) and their children. After purging the previous distal
        # branches, the parent candidates may still have some children. Those will 
        # be handles differently from those who have no children left.
        family_book = {}
        for b in distal_candidates:
            family_book.update({b:self.branches[b].children.copy()})
        
        # Remove children from the family book, most of the entries will be empty but
        # possibly not all of them
        for b in delection_targets:
            p = self.branches[b].parent
            family_book[p].remove(b)
            
        # Now we are about to clone most of the previous tree, excepting the distal
        # airways.
        # Placeholder for the final distal branches
        confirmed_distal = []
        for b in self.branches:
            
            # If the branch is not going to be deleted
            if b not in delection_targets:
                
                # extract the relevant information from the old branch
                old_branch = self.branches[b]
                start_id = old_branch.id0
                start_point = old_branch.p0
                branch_id = old_branch.id
                path = old_branch.path.copy()
                end_id = old_branch.id1
                end_point = old_branch.p1
                radius = old_branch.radius
                
                # Assign the new family book value which may be an empty list or a 
                # lone child
                if b in distal_candidates:
                    children = family_book[b]
                    
                    if len(children)==0:
                        confirmed_distal += [b]
                else:
                    children = old_branch.children.copy()
                
                # Generate the new branch and assign it to the new tree
                new_branch = Branch(start_id,start_point,branch_id,path)
                new_branch.close(end_id, end_point)
                new_branch.set_radius(radius)
                new_branch.set_children(children)
                new_tree.branches.update({b:new_branch})
        
        # Update the distal branches list
        new_tree.distal_branches = confirmed_distal
        
        return new_tree

    def assemble_linear_system(self, Q0 = -1e6, P0 = None, 
                               prescribed_pressure_dummy = 0.0):  
        
        '''
        
        Assemble a linear system associated with the flow within the airway 
        tree. 
        
        The prescribed_pressure_dummy assigns this value to every distal point
        to simplify calculations. A more complex data passing should be 
        implemented in real calculations. This is only to compute the Reynolds
        number
        
        '''
        
        # mu: dynamic or absolute viscosity kg/(mm-s)
        # Q0 in mm3/s
        # P0 in kPa
        
        N = len(self.branches.keys())
        # system Ax = b; unknowns x ~ 2*N+1 => Qs~N, Ps~N+1; x = [Qs,Ps]
        A = np.zeros((2*N+1,2*N+1),dtype=float)
        f = np.zeros((2*N+1,1))
        
        eqno= 0
        # for every branch
        
        for b in self.branches.keys():
            
            #retrieve branch
            elem = self.branches[b]
            
            # If this branch is not distal
            if not b in self.distal_branches:
                
                # First check if it is the initial branch
                # Boundary condition, either prescribed flow or pressure
                if b ==0:
                    if not Q0 is None and P0 is None:
                        A[eqno,0] = 1.0 # Prescribed flux at inlet
                        f[eqno,0] = Q0
                    elif not P0 is None and Q0 is None:
                        A[eqno,N] = 1.0 
                        f[eqno,N] = P0  # Prescribed pressure at inlet
                    else:
                        raise Exception("Either P0 or Q0 must be prescribed and the other must be None")
                    eqno += 1
                
                # Any branch that is not distals, bifurcates and generates children, 
                # and a local mass continuity equation must be enforced,
                
                # Retrieve dof associated to each branch
                p_dof = elem.bdof # parent dof
                    
                if len(elem.children) == 2:
                    ch0_bid,ch1_bid = elem.children # children dof
                    ch0_dof = self.branches[ch0_bid].bdof
                    ch1_dof = self.branches[ch1_bid].bdof

                    A[eqno,p_dof] = 1.0
                    A[eqno,ch0_dof] = -1.0
                    A[eqno,ch1_dof] = -1.0
                    eqno += 1
                elif len(elem.children) == 1:
                    
                    ch0_bid = elem.children[0] # children dof
                    ch0_dof = self.branches[ch0_bid].bdof
                    A[eqno,p_dof] = 1.0
                    A[eqno,ch0_dof] = -1.0
                    eqno += 1
                    
            # in distal branches we have prescribed pressure
            else:
                
                # numbering of the pressure values starts after N 'Qs' and there are
                # as many pressure values as points there are in the mesh. 
                
                # Take the distal branch and retrieve its distal point dof and add N
                pressure_id = N + elem.vdof1
                A[eqno,pressure_id] = 1.0 # activate the element in matrix
                f[eqno,0] = prescribed_pressure_dummy
                eqno += 1 
            
            # Determine the branch resistance
            L = elem.line_length
            r = elem.radius
            R = 8*self.mu*L/(np.pi*r**4)
            
            # Retrieve the branch dof ids
            proximal_id = N+elem.vdof0
            distal_id = N+elem.vdof1 
            

            # Poiseuille flow
            A[eqno,elem.bdof] = R # respective Q
            A[eqno,proximal_id] = 1.0
            A[eqno,distal_id] = -1.0
            eqno += 1
        
        self.linear_system_A = A
        self.linear_system_f = f


    def solve_linear_system(self,export_path=None,visualization_npz_path=None):
        
        ''' 
        Solve the linear system constructed when calling the method
        'assemble_linear_system' before.
        
        Once a result has been found, then a local Reynolds numbers is computed
        and assigned to the respective branches and a mesh is exported where
        the flow and pressure per branch is found
        
        '''       
        
        # Number of branches
        N = len(self.branches.keys())
        # Solve the linear system
        x = np.linalg.solve(self.linear_system_A,self.linear_system_f)
        # Retrieve data (Qs and Ps)
        qs = x[:N].flatten()
        ps = x[N:].flatten()
        # Generate empty array to compute Reynolds number per branch
        res = np.zeros_like(qs)
        
        # For every branch
        for b in self.branches.keys():
            # The branch numeration does not necesarily equal the dof after
            # trimming.
            e = self.branches[b].bdof
            
            # Extract the flow
            q_e = qs[e]
            # Determine velocity
            radius = self.branches[b].radius
            area = np.pi*radius**2
            v_e = q_e/area
            # Determine Reynolds
            re = self.rho * v_e * 2* radius / self.mu
            # Assign to the array and save in the branch object
            res[e] = re
            self.branches[b].set_re(re)
        # Reynolds in absolute value
        res = np.abs(res)
                
        if not export_path is None:
            # Reference mesh
            mesh = self.mesh
            # Append new files
            mesh = io.Mesh(points=mesh['xyz'],cells={"line":mesh['ien']},
                    point_data={"pressure":ps},cell_data={"flow":[qs], "re":[res]})
            # Write somewhere
            mesh.write(export_path)

        if not visualization_npz_path is None:
            
            # Data holder
            starts = []
            ends = []
            radii = []
            order = []
            
            # Fill up arrays
            for b in self.branches:
                branch = self.branches[b]
                starts += [branch.p0]
                ends += [branch.p1]
                radii += [branch.radius]
                order += [branch.order]
            # Transform as numpy array
            starts = np.array(starts)
            ends = np.array(ends)
            radii = np.array(radii)
            order = np.array(order)
            # Export npz data. This is opened in Paraview using a 
            # ProgrammableSource source object whose script is available
            # in the src directory.
            np.savez(visualization_npz_path, 
                     starts=starts,ends=ends,radii=radii, order=order)

    def export_distal_cloud(self, outname):
        '''
        Export the current cloud of points, useful to create subdomain meshes.
        '''
        # export a cloud of points to generate subdomains
        distal = []
        for bno in self.distal_branches:
            branch = self.branches[bno]
            distal += [branch.p1]
        # Save as outname (.npz)        
        np.savez(outname, distal=distal)


class Pipeline:
    
    '''
    This class is intended to be used in the poromechanical lung models to 
    read the information conveyed in piecewise-line arrangements of the 
    airway tree. Through this class, a mesh can be read, and a linear system
    can be built and managed.
    '''
    
    def __init__(self, mesh_path, rho = 1e-9, mu = 1.825e-8,gamma=0.327):
        '''
        Initialize the pipeline by reading a compatible linear mesh 
        representing the airway tree. The mesh is generated using the Tree
        structure also contained in this library.
        '''
        
        
        # Global parameters
        self.mesh_path = mesh_path
        self.rho = rho
        self.mu = mu
        self.gamma = gamma
        
        # Load the mesh
        mesh = io.read(mesh_path)
        
        # Mesh-relevant fields
        self.points = mesh.points
        self.elems = mesh.cells_dict['line']
        self.radius = mesh.cell_data['radius'][0]
        self.area = np.pi*self.radius**2
        self.length = mesh.cell_data['length'][0]
        self.distal_points = mesh.point_data['distal']
        self.distal_elems = mesh.cell_data['distal'][0]
        self.distal_point_ids = np.arange(len(self.points))[self.distal_points==1]  
        self.distal_elem_ids = np.arange(len(self.elems))[self.distal_elems==1]  
        self.N = len(self.elems)

        # Generate a mapping between distal element ids and distal point ids
        pages = {"p2e":{}, "e2p":{}}
        for pid in self.distal_point_ids: 
            args = np.argwhere(self.elems==pid)
            if len(args) == 1:
                pages["p2e"].update({pid:args[0,0]})
                pages["e2p"].update({args[0,0]:pid})
            else:
                raise Exception("Translation weird for point %i in 'p2e'"%pid)

        
        # Distal point id to subdomain id dictionary
        self.translate = {"p2s":{p:s for s,p in enumerate(self.distal_point_ids)},
                          "s2p":{s:p for s,p in enumerate(self.distal_point_ids)},
                          "p2e":pages["p2e"],
                          "e2p":pages["e2p"]}

        # System of equation placeholders
        self.A = np.zeros((2*self.N+1,2*self.N+1),dtype=float)
        self.f = np.zeros((2*self.N+1,1),dtype=float)
        self.x = np.zeros((2*self.N+1,1),dtype=float)
        
    def assemble_linear_system(self, Pdistal, Q0=None, P0=None, upstream_resistance=None,
                               downstream_resistances=None):
        '''
        This will allow to assemble a linear system associated to viscous flow
        through an airway tree,
        '''
        # Equation pointer
        eqno = 0
        
        for e, line in enumerate(self.elems): # N items/equations
                 
            
            # Assign dofs associated to the pressure nodes, associated to each end
            # of the element
            prox_pdof, dist_pdof = line
            prox_pdof += self.N
            dist_pdof += self.N
                
            # Poiseuille equations
            # Determine the branch resistance
            L = self.length[e]
            r = self.radius[e]
            R = 8*self.mu*L/(np.pi*r**4)
        
            # Poiseuille flow
            self.A[eqno,e] = R # respective Q-location
            self.A[eqno,prox_pdof] = 1.0 # P-proximal
            self.A[eqno,dist_pdof] = -1.0 # P-distal
            
            if e == 0 and not (upstream_resistance is None):
                self.A[eqno,e] += upstream_resistance
            
            if e in self.distal_elem_ids and not (downstream_resistances is None):
                self.A[eqno,e] += downstream_resistances[e]

            # Next equation
            eqno += 1
        
        # For every point (N+1), we will establish a new equation
        for p in np.arange(len(self.points)): # N+1 equations
            
            # Inlet
            if p == 0:
                if not Q0 is None and P0 is None:
                    self.A[eqno,0] = 1.0 # Prescribed flux at inlet
                    self.f[eqno,0] = Q0
                elif not P0 is None and Q0 is None:
                    self.A[eqno,self.N] = 1.0 
                    self.f[eqno,self.N] = P0  # Prescribed pressure at inlet
                else:
                    raise Exception("Either P0 or Q0 must be prescribed and the other must be None")
        
            # Distal point 
            elif p in self.distal_point_ids: 
                # Distal points, apply the subdomain-averaged pressure
                self.A[eqno,p+self.N] = 1.0 # activate the element in matrix
                self.f[eqno,0] = Pdistal[p]
        
            else:
                # This is mass conservation now
                for el, pos in np.argwhere(self.elems==p):
                    # Retrieve the dof associated to pressure in the vertex
                    qdof = el
                    # Distal will be 1.0 and proximal will be -1.0
                    if pos == 1:
                        self.A[eqno,qdof] = 1.0
                    else:
                        self.A[eqno,qdof] = -1.0
                                
            # Next equation
            eqno+=1
    
    def update_linear_system(self, Pdistal, Q0=None, P0=None):
        '''
        Mechanism to update the linear system of equations. The dictionary
        Pdistal has to be updated according to the average pressure in every
        subdomain and either Q0 or P0 must be specified.
        '''
        # Start counting equations from the Nth equation
        eqno = self.N
        
        # For every point (N+1), we will establish a new equation
        for p in np.arange(len(self.points)): # N+1 equations
            
            # Inlet
            if p == 0:
                if not Q0 is None and P0 is None:
                    self.A[eqno,0] = 1.0 # Activate flux eqn. at inlet 
                    self.A[eqno,self.N] = 0.0 # Make sure the pressure eq.n is not active
                    self.f[eqno,0] = Q0  # Set the indicated value 
                elif not P0 is None and Q0 is None:
                    self.A[eqno,0] = 0.0 # Make sure the flow eqn is not active
                    self.A[eqno,self.N] = 1.0  # Activate prescribed flux eqn.
                    self.f[eqno,0] = P0  # Prescribed pressure at inlet
                else:
                    raise Exception("Either P0 or Q0 must be prescribed and the other must be None")
        
            # Distal point 
            elif p in self.distal_point_ids: 
                # Distal points, apply the subdomain-averaged pressure
                self.A[eqno,p+self.N] = 1.0 # activate the element in matrix
                self.f[eqno,0] = Pdistal[p]
                
            # Mass conservation equations
            else: 
                # Skip these points as they do not change
                eqno+=1
                continue

            # Next equation
            eqno+=1
    
    def solve_linear_system(self):
        '''
        Solve the linear system of equations.
        '''
        self.x = np.linalg.solve(self.A,self.f)
    
    def export_solution(self, filename, compute_re = False):
        '''
        Export a finite element mesh corresponding to the current solution
        of the pipeline flow system.
        '''
        # Extract relevant fields from solution
        Qs = self.x[:self.N].flatten()
        Ps = self.x[self.N:].flatten()
        
        # Create data dictionaries
        point_data = {"awt_pressure":Ps}
        cell_data = {"awt_flow":[Qs]}
        
        # Compute Re if required
        if compute_re:
            Vs = Qs/self.area
            re = np.abs(self.rho*Vs*2*self.radius/self.mu)
            # Update dictionary
            cell_data.update({"awt_Re":[re],
                              "awt_velocity":[Vs]})
        # Create mesh
        out = io.Mesh(points=self.points, cells={"line":self.elems},
                      point_data=point_data,
                      cell_data=cell_data)
        
        # Write to filename
        out.write(filename)
        
    def retrieve_qs(self):
        '''
        Returns a dictionary for every "e-th subdomain", where 'e' is the 
        number of the subdomain, which are numbered according to the distal
        element id. The data in the dictionary is the flow Q for every 
        subdomain. 
        
        Note that the i-th position in the Q-vector is due to the construction
        of the system of equations, where the first N variables are the flow
        in every element of the Pipe system, numbered according to their 
        corresponding element.
        '''
        # Retrieve flows
        Qs = self.x[:self.N].flatten()
        # Note that i refers to the element numbering, which may be disordered
        # and e refers to the subdomain numbering, which is sorted according
        # to the distal element order itself.
        return {e:Qs[i] for e,i in enumerate(self.distal_elem_ids)}
    
    def retrieve_ps(self):
        '''
        Returns a dictionary for every "e-th subdomain", where 'e' is the 
        number of the subdomain, which are numbered according to the distal
        element id. The data in the dictionary is the pressure P for every
        subdomain.
        
        Note that the i-th position in the P-vector is due to the construction
        of the system of equations, where the first N variables are the flow
        in every element of the Pipe system, and the following N+1 are the 
        pressure variables, numbered according to their corresponding point.
        '''
        Ps = self.x[self.N:].flatten()
        return {e:Ps[i] for e,i in enumerate(self.distal_point_ids)}


    def update_pedley_resistances(self,upstream_resistance=None,
                                       downstream_resistances=None,):
        
        ''' 
        Updating the resistances according to Pedley's model 
        '''
        
        qs = self.x[:self.N].flatten()
        # Equation pointer
        eqno = 0
        
        for e, line in enumerate(self.elems): # N items/equations
                
           
           # Assign dofs associated to the pressure nodes, associated to each end
           # of the element
           prox_pdof, dist_pdof = line
           prox_pdof += self.N
           dist_pdof += self.N
               
           # Poiseuille equations
           # Determine the branch Poiseuille resistance
           L = self.length[e]
           r = self.radius[e]
           R = 8*self.mu*L/(np.pi*r**4)
           # Determine Reynolds
           Re = 2*self.rho*np.abs(qs[e])/(self.mu*np.pi*(r*1e-3))
           # Determine Pedley resistance
           R = max([R,self.gamma*np.sqrt(Re*2*r/L)*R])

           # Update resistance
           self.A[eqno,e] = R # respective Q-location
           
           if e == 0 and not (upstream_resistance is None):
               self.A[eqno,e] += upstream_resistance
           
           if e in self.distal_elem_ids and not (downstream_resistances is None):
               self.A[eqno,e] += downstream_resistances[e]

           # Next equation
           eqno += 1    

    def update_gamma(self, gamma):
        self.gamma = gamma

def initialize_tree_from_vtu(vtu_path):
    '''
    Fills up most of the needed structures for a Tree structure while loading
    the data from a vtu tree geometry exported beforehand.
        
    It should be integrated into the Tree object maybe but this is good enough
    for the moment.
    
    TODO: Integrate with the Tree structure
    '''
    
    # Load vtu
    airway_mesh = io.read(vtu_path)
    
    # Wrap into a function "Read From Mesh" function
    
    cells = airway_mesh.cells_dict['line']
    points = airway_mesh.points
    distal = airway_mesh.cell_data['distal'][0]
    
    radius = airway_mesh.cell_data['radius'][0]
    length = airway_mesh.cell_data['length'][0]
    
    
    TestTree = Tree(seed=points[0],points=points,elem=cells)
    
    
    for bid, cell in enumerate(cells):
        # Create data structure
        id0, id1 = cell          
        # Read the corresponding points
        p0 = points[id0]
        p1 = points[id1]
        # Generate branch structure
        branch = Branch(id0,p0,bid,[None]) # path = None
        # Fill up base branch data
        branch.p1 = p1
        branch.id1 = id1
        branch.radius = radius[bid]
#        branch.order = order[bid]
        branch.distal = distal[bid]
        branch.line_length = length[bid]
        
        # Check for parent
        upstream = np.argwhere(cells==id0)
        for fam in upstream:
            if fam[1] == 1:
                branch.parent = fam[0]
                
        # Check for childrens
        downstream = np.argwhere(cells==id1)
        for fam in downstream:
            if fam[1] == 0:
                branch.children += [fam[0]]
        
        # Check for distal condition
        if distal[bid] == 1:
            TestTree.distal_branches += [bid]
            TestTree.distal_points += [p1]
        
        TestTree.branches.update({bid:branch})    
    
    # Build paths
    for bid in range(len(cells)):
        
        if bid == 0:
            TestTree.branches[bid].path=[bid]
            continue 
        
        # initialize path
        path = [bid]
        
        # parent id
        pid = TestTree.branches[bid].parent
        
        path += [pid]
        
        while pid != 0:
            pid = TestTree.branches[pid].parent
            path += [pid]
        
        path.reverse()
        
        TestTree.branches[bid].path = path.copy()
            
    return TestTree


def tetvol(points):
    '''
    Determine the volume of a tetrahedron
    '''
    ps = np.hstack([points,np.ones((4,1))])
    return np.abs(np.linalg.det(ps))/6

def distribute_subdomains(path_to_mesh, terminals, evaluate_distribution=True,
                          verbose=False):
    '''
    
    Take a finite element mesh such as mesh000000.vtu (generated in FEniCS) and
    distribute the distal terminals through the mesh to generate subdomains. 
    
    We intend to evaluate the homogeneity of the distribution, so we count the
    amount of elements and the volume assigned to each subdomain. Those lists
    can be processed to generate metric of distribution efficiency.
    
    '''
    
    # Read the mesh
    reference_mesh = io.read(path_to_mesh)
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
    
    # Count the number of elements and the volume per subdomain
    if evaluate_distribution:
        
        # Empty arrays
        counter = []
        volumes = []
        
        # List
        nterminals = len(terminals)
        parser = np.floor(nterminals/20) # Screen messages
        
        # Evaluate every terminal
        for j in range(nterminals):
            # Progress messages
            if j%parser == 0 and verbose: print("%.0f"%(j/parser*5)+"%")
            # Mask the irrelevant domains
            sd_mask = cell_subdomain==j
            sd_cells = ien[sd_mask]
            # Evaluate the domain's volume
            vol = 0.0
            for scell in sd_cells:
                ps = xyz[scell]
                vol += tetvol(ps)    
            # Fill the lists
            counter += [np.count_nonzero(sd_mask)]
            volumes += [vol]
        
        volumes = np.array(volumes)
        counter = np.array(counter)    
    
    
        return cell_subdomain, counter, volumes
    
    else:
        
        return cell_subdomain, None, None


# %%

if __name__ == "__main__" and True:
    
# =============================================================================
#  Template for the processing of an airway tree
# =============================================================================
    root = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/"
    path_to_mesh = root+"MESH/"
    path_to_images = root+"NIFTI/"

    '''
    These are the meshes generated in CGAL
    texts = ['correspondance-lcc.polylines.txt', 
             'correspondance-poly.polylines.txt', 
             'correspondance-sm.polylines.txt', 
             'skel-lcc.polylines.txt', 
             'skel-poly.polylines.txt', 
             'skel-sm.polylines.txt']
    '''
    
    # Full path to the skeletonization
    skel_txt = path_to_mesh + "skel-sm.polylines.txt"
    corr_txt = path_to_mesh + "correspondance-sm.polylines.txt"
    
    
    # Reference Nifti image     
    nifti_lung = path_to_images+"NEW_Mask_Exp.nii.gz"
    # Straight from MATLAB tetrahedral lung
    mat_lung = path_to_mesh+"tet_lung.mat"
    # FEniCS processed tetrahedral lung
    vtu_lung = path_to_mesh+"FEniCS/mesh000000.vtu"
    # FEniCS processed triangle lung mesh 
    vtu_tri = path_to_mesh+"FEniCS/boundary_markers000000.vtu"
    # Cloud of points for airway generation purposes
    npz_cloud = path_to_mesh+"cloud.npz"
    #npz_cloud = "D:/791k_cloud.npz"

    # dummy vtu and npz for development purposes
    vtu_dummy = path_to_mesh+"dummy_geometry.vtu"
    npz_dummy = path_to_mesh+"dummy_geometry.npz"

    # Simplification of the triangle lung mesh
    surf_ply = vtu_tri.split(".vtu")[0]+".ply"

    # Simplify the surface mesh for ray casting
    rewrite_surface_mesh(vtu_tri, surf_ply)

    # Activate if you want to generate a newcloud of points for the airway 
    # generation scheme.
    if False:
        alv_density=1e-1
        cloud = distribute_point_cloud(surf_ply, return_cloud=True, 
                                       vtuname = vtu_dummy, npzname=npz_dummy,
                                       alv_density=alv_density)
        np.savez(npz_cloud,cloud=cloud)
    else:
        npz = np.load(npz_cloud)
        cloud = npz['cloud']


    # Read the txt and extract a finite element mesh (lines)
    points, elem = skel_to_mesh(skel_txt,skel_mesh=None)
    
    # Transform the skeleton mesh (traslation and scaling) using lung meshes
    points = transform_skel(mat_lung, vtu_lung, nifti_lung, points, elem, 
                            outname=path_to_mesh+"raw_skel.vtu",
                            scale_airways_with_affine=True, positive_affine=True,
                            override_offset=False)
 
    # Retrieve topological information
    out = classify_skel_nodes(points, elem)
    
    # Generate a tree using the top of the trachea as seed
    seed = out['inlet_id']
    airway_tree = Tree(seed, points,elem, branch_length_threshold=1.50)
    # Initialize the skeleton analysis
    airway_tree.activate_seed() 
    # Create branches from skeletonization
    airway_tree.grow_tree() 
    # Determine radius from the skeletonization
    airway_tree.radius_from_skeletonization(skel_txt, corr_txt, nifti_lung)
    # Export minimal tree
    airway_tree.process_mesh(path_to_mesh+"skel.vtu",include_order=False)

# %%

    # Generate synthetic branches
    for i in range(8):
        print("Generating branches, loop %i"%(i+1))
        airway_tree.generate_sythetic_branches(cloud, surf_ply,
                                               kmeans_coherence=True)
    # Assign Strahler order
    airway_tree.set_strahler_order()
    # Assign radius following Tawhai et al. (2004)
    airway_tree.assign_synthetic_radii()
    # Export mesh
    airway_tree.process_mesh(path_to_mesh+"skel.vtu",include_order=True)
    # Assemble linear system
    airway_tree.assemble_linear_system()
    # Solve linear system and export as vtu
    airway_tree.solve_linear_system(export_path=path_to_mesh+"fskel.vtu")
# %%

if __name__ == "__main__" and False:
    
# =============================================================================
#  Template for the processing of an airway tree
# =============================================================================
    
    path_to_mesh = "../testing-data/AIRWAY-MESHES/dev/"

    '''
    These are the meshes generated in CGAL
    texts = ['correspondance-lcc.polylines.txt', 
             'correspondance-poly.polylines.txt', 
             'correspondance-sm.polylines.txt', 
             'skel-lcc.polylines.txt', 
             'skel-poly.polylines.txt', 
             'skel-sm.polylines.txt']
    '''
    
    # Full path to the skeletonization
    skel_txt = path_to_mesh + "skel-sm.polylines.txt"
    corr_txt = path_to_mesh + "correspondance-sm.polylines.txt"
    
    
    # Reference Nifti image     
    nifti_lung = "../testing-data/NIFTI/NEW_Lung_Segmentation_Exp.nii.gz"
    # Straight from MATLAB tetrahedral lung
    mat_lung = "../testing-data/LUNG-MESHES/Lung.mat"
    # FEniCS processed tetrahedral lung
    vtu_lung = "../testing-data/LUNG-MESHES/mesh000000.vtu"
    # FEniCS processed triangle lung mesh 
    vtu_tri = "../testing-data/LUNG-MESHES/boundary_markers000000.vtu"
    # Cloud of points for airway generation purposes
    npz_cloud = "../testing-data/LUNG-MESHES/79k_cloud.npz"
    #npz_cloud = "D:/791k_cloud.npz"

    # dummy vtu and npz for development purposes
    vtu_dummy = "../testing-data/LUNG-MESHES/dummy_geometry.vtu"
    npz_dummy = "../testing-data/LUNG-MESHES/dummy_geometry.npz"

    # Simplification of the triangle lung mesh
    surf_ply = vtu_tri.split(".vtu")[0]+".ply"

# %%
    # Simplify the surface mesh for ray casting
    rewrite_surface_mesh(vtu_tri, surf_ply)

    # Activate if you want to generate a newcloud of points for the airway 
    # generation scheme.
    if False:
        alv_density=1
        cloud = distribute_point_cloud(surf_ply, return_cloud=False, 
                                       vtuname = vtu_dummy, npzname=npz_dummy,
                                       alv_density=alv_density)
        
    else:
        npz = np.load(npz_cloud)
        cloud = npz['cloud']


    # Read the txt and extract a finite element mesh (lines)
    points, elem = skel_to_mesh(skel_txt,skel_mesh=None)
    
    # Transform the skeleton mesh (traslation and scaling) using lung meshes
    points = transform_skel(mat_lung, vtu_lung, nifti_lung, points, elem, 
                            outname=path_to_mesh+"raw_skel.vtu",
                            scale_airways_with_affine=False)
    
    # Retrieve topological information
    out = classify_skel_nodes(points, elem)
    
    # Generate a tree using the top of the trachea as seed
    seed = out['inlet_id']
    airway_tree = Tree(seed, points,elem, branch_length_threshold=1.50)
    # Initialize the skeleton analysis
    airway_tree.activate_seed() 
    # Create branches from skeletonization
    airway_tree.grow_tree() 
    # Determine radius from the skeletonization
    airway_tree.radius_from_skeletonization(skel_txt, corr_txt, nifti_lung)
    # Generate synthetic branches
    for i in range(8):
        print("Generating branches, loop %i"%(i+1))
        airway_tree.generate_sythetic_branches(cloud, surf_ply,
                                               kmeans_coherence=True)
    # Assign Strahler order
    airway_tree.set_strahler_order()
    # Assign radius following Tawhai et al. (2004)
    airway_tree.assign_synthetic_radii()
    # Export mesh
    airway_tree.process_mesh(path_to_mesh+"skel.vtu",include_order=True)
    # Assemble linear system
    airway_tree.assemble_linear_system()
    # Solve linear system and export as vtu
    airway_tree.solve_linear_system(export_path=path_to_mesh+"fskel.vtu")

#%%  trim the tree

    # 1st trimming
    simple_tree = airway_tree.trim_generation()
    simple_tree.set_strahler_order()
    simple_tree.process_mesh(path_to_mesh+"skel2.vtu",include_order=True)
    #simple_tree.assemble_linear_system()
    #simple_tree.solve_linear_system(export_path=path_to_mesh+"fskel2.vtu")

    # recursive trimming
    for i in range(5):
        simple_tree = simple_tree.trim_generation()
        simple_tree.set_strahler_order()
        simple_tree.process_mesh(path_to_mesh+"skel%i.vtu"%(i+3),include_order=True)
        #simple_tree.assemble_linear_system()
        #simple_tree.solve_linear_system(export_path=path_to_mesh+"fskel3.vtu")
        
        # Export cloud of points to generate subdomains
        simple_tree.export_distal_cloud(path_to_mesh+"distal_points_cloud.npz")

# %% export/import module (pickled)
    
    # A reasonably sized tree was transformed into an 115 Mb object (careful!)
    
    # export    
    if False:
        save_obj = open(path_to_mesh+'simple_tree.obj','wb')
        pickle.dump(simple_tree, save_obj)
        save_obj.close()
            
    # import
    if False:
        load_obj = open(path_to_mesh+'simple_tree.obj','rb')
        simple_tree = pickle.load(load_obj)
        load_obj.close()

# %%
# =============================================================================
# Handling the airway pipeline using the Pipeline object
# =============================================================================

    if False:
        
        path_to_mesh = "../testing-data/stable/airway-tree/"
    
        pipe = Pipeline(path_to_mesh+"skel7.vtu")
        Pdistal = {p:0.0 for p in pipe.distal_point_ids}
        pipe.assemble_linear_system(Pdistal,Q0=-1e6)
        pipe.solve_linear_system()
        #pipe.export_solution(path_to_mesh+"pipe0.vtu",compute_re=True)
        
        #Pdistal = {p:np.random.normal(loc=0.5, scale=0.01) for p in pipe.distal_point_ids}
        Pdistal = {p:-1.0 for p in pipe.distal_point_ids}
        
        
# %%

if __name__ == "__main__" and False:

# =============================================================================
#  Template for the processing of an airway tree
# =============================================================================
    
    root = "C:/Users/angus/Downloads/CORNELL-GEOM/test/"

    path_to_mesh = root +  "MESH-20/"
    path_to_images = root +  "NIFTI/"
    path_to_airways = root + "AIRWAYS/"
    
    '''
    These are the meshes generated in CGAL
    texts = ['correspondance-lcc.polylines.txt', 
             'correspondance-poly.polylines.txt', 
             'correspondance-sm.polylines.txt', 
             'skel-lcc.polylines.txt', 
             'skel-poly.polylines.txt', 
             'skel-sm.polylines.txt']
    '''
    
    # Full path to the skeletonization
    skel_txt = path_to_airways + "skel-sm.polylines.txt"
    corr_txt = path_to_airways + "correspondance-sm.polylines.txt"
    
    
    # Reference Nifti image     
    nifti_lung = path_to_images+"SMOOTH_Mask_Exp.nii.gz"
    # Straight from MATLAB tetrahedral lung
    mat_lung =  path_to_mesh+"tet_lung.mat"
    # FEniCS processed tetrahedral lung
    vtu_lung = path_to_mesh+"FEniCS-trimmed/mesh000000.vtu"
    # FEniCS processed triangle lung mesh 
    vtu_tri = path_to_mesh+"FEniCS-trimmed/boundary_markers000000.vtu"
    # Cloud of points for airway generation purposes
    npz_cloud = path_to_mesh+"FEniCS-trimmed/cloud.npz"
    #npz_cloud = "D:/791k_cloud.npz"

    # dummy vtu and npz for development purposes
    vtu_dummy = path_to_mesh+"dummy_geometry.vtu"
    npz_dummy = path_to_mesh+"dummy_geometry.npz"

    # Simplification of the triangle lung mesh
    surf_ply = vtu_tri.split(".vtu")[0]+".ply"

    # Simplify the surface mesh for ray casting
    rewrite_surface_mesh(vtu_tri, surf_ply)

    # Activate if you want to generate a newcloud of points for the airway 
    # generation scheme.
    if False:
        alv_density=1e-1
        cloud = distribute_point_cloud(surf_ply, return_cloud=True, 
                                       vtuname = vtu_dummy, npzname=npz_dummy,
                                       alv_density=alv_density)
        np.savez(npz_cloud, cloud=cloud)
    else:
        npz = np.load(npz_cloud)
        cloud = npz['cloud'] 


    # Read the txt and extract a finite element mesh (lines)
    points, elem = skel_to_mesh(skel_txt,skel_mesh=None)
    
    # Transform the skeleton mesh (traslation and scaling) using lung meshes
    points = transform_skel(mat_lung, vtu_lung, nifti_lung, points, elem, 
                            outname=path_to_mesh+"raw_skel.vtu",
                            scale_airways_with_affine=False)
    
    # Retrieve topological information
    out = classify_skel_nodes(points, elem)
    
    # Generate a tree using the top of the trachea as seed
    seed = out['inlet_id']
    airway_tree = Tree(seed, points,elem, branch_length_threshold=1.50)
    # Initialize the skeleton analysis
    airway_tree.activate_seed() 
    # Create branches from skeletonization
    airway_tree.grow_tree() 
    # Determine radius from the skeletonization
    airway_tree.radius_from_skeletonization(skel_txt, corr_txt, nifti_lung)
    
    # Save current mesh
    airway_tree.process_mesh(path_to_airways+"base-skel.vtu",include_order=True)

    
    # Generate synthetic branches
    for i in range(8):
        print("Generating branches, loop %i"%(i+1))
        airway_tree.generate_sythetic_branches(cloud, surf_ply,
                                               kmeans_coherence=True)
        # Assign Strahler order
        airway_tree.set_strahler_order()
        # Assign radius following Tawhai et al. (2004)
        airway_tree.assign_synthetic_radii()
        
        # Export mesh
        airway_tree.process_mesh(path_to_airways+"base-skel-%i.vtu"%i,include_order=True)
    # Assemble linear system
    airway_tree.assemble_linear_system()
    # Solve linear system and export as vtu
    airway_tree.solve_linear_system(export_path=path_to_mesh+"fskel.vtu")


# %%

def trim_airways_with_small_subdomains(in_airways, reference_mesh, out_airways, 
                                       verbose=False, 
                                       in_analysis=False,
                                       out_analysis=False,
                                       return_out_analysis=True):
    
    TestTree = initialize_tree_from_vtu(in_airways)

    terminals = TestTree.distal_points
    
    subdomains, counts, volumes = distribute_subdomains(reference_mesh, terminals)

    airway_mesh = io.read(in_airways)    

    cells = airway_mesh.cells_dict['line']
    points = airway_mesh.points

    
    if verbose: print("Number of subdomains: %i"%len(terminals))
       
    for data, name in zip([volumes,counts],["Volumes","Counts",]):
            
        if name=="Volumes":
            continue
            
        q50 = np.median(data)
        q25 = np.quantile(data,0.25)
        q75 = np.quantile(data,0.75)
        qlow = np.quantile(data,0.10)
        qhigh = np.quantile(data,0.90)
            
        if verbose:
            print("[BEFORE] Evaluating %s:"%name)
            print(" Median (Q25 - Q75) : %.0f (%.0f - %.0f)"%(q50,q25,q75))
            print(" * MIN: %.0f"%np.min(data))
            print(" * Q10: %.0f"%qlow)
            print(" * Q90: %.0f"%qhigh)

    # These delection targets are terminals that generate small subdomains. They might
    # be merged with their pair-child into a single terminal.
    
    # Options: 
    # 1) Merge with their pair-child into a single terminal; Issue: There may be some w/o brothers
    # 2) Delection of the branch; Issue: Retraction towards the parent child may disort results
    
    # We may do both; If 1) fails then apply 2)
    
    # #These are ids from the distal_points list in the airway tree mesh
    deletion_targets = np.arange(len(terminals))[data < qlow]
    # distal_points and distal_branches are sorted correspondingly, thus we could 
    # use the distal_branches array directly
    
    # Take the distal branches from the active tree
    distal_branches = np.array(TestTree.distal_branches)
    # # Select the branches that are marked for deletion 
    # This assumes that distal_points and distal_branches follow the same ordering.
    deletion_branches = distal_branches[deletion_targets]
    
    # here we will store 
    complete_trimming = [] # Mark the parents that will become distal branches
    trimming_targets =  [] # Definitively delete these branches
    
    # Move through each one of the branches selected for deletion
    for bid in deletion_branches:
        
        # Retrieve the parent
        pid = TestTree.branches[bid].parent
        # Retrieve the list of the children
        children = TestTree.branches[pid].children.copy()
        # Remove this branch from the children list
        children.remove(bid)
                
        # Usually there are only two children, but this is not always the case
        merger = [] # Placeholder
        for child in children: # Every children but the main branch
            merger += [child in deletion_branches]
        merger = np.all(merger)      
        
        if merger:
            # Note that all children are to be deleted and these are the parent ids that will become distal pointsl
            # The children need to be removed from the distal branches and point lists.
            complete_trimming += [pid]
            trimming_targets += [bid]
            
        else:
            # Note that these are the branches ids to be deleted. At least one other children branch
            # remains connected to the parent that will remain as distal.
            trimming_targets += [bid]
            # How do we renumber?
            
    complete_trimming = np.unique(np.array(complete_trimming)) # this has parent branch ids
    trimming_targets = np.array(trimming_targets) # this has children branch ids
    
    # Invoking a new tree requires a point list and a cell list; So we need to create the new lists before-
    #TrimTree = Tree()
        
    # PARTIAL TRIMMINIG 
    deletion_point_ids = np.unique(cells[trimming_targets][:,1]) # This has point ids to be deleted
    
    point_translator = {}
    point_mask = np.zeros(len(points),dtype=bool)
    
    # This block purges the old point list and creates a new dictionary that
    # translated old ids to new ids.
    pid = 0; new_pid = 0
    while pid < len(points):
        
        # Check if point is going to be deleted
        if pid in deletion_point_ids:
            # if True (deleted point)
            point_translator.update({pid:None})
            # note that we dont update the new numbering in this scenario
        else:
            # if False (kept point)
            point_mask[pid] = True
            point_translator.update({pid:new_pid})
            new_pid +=1 # update the new numbering
        # check next point
        pid += 1
    
    # New point list
    new_points = points[point_mask]
    # Some relevant fields
    distal_point = np.zeros(len(new_points),dtype=int)
    new_cells = []
    radius = []
    length = []
    distal = []
    
    new_bid = 0
    
    # Create new cell list; Evaluate every cell from the original tree
    for bid, cell in enumerate(cells):
        
        # Extract the point ids that compose the old cell
        p0, p1 = cell # in old numbering
        
        # Cells in complete trimming are parents whose children are going to be
        # erased. Their end-points (distal) are going to be terminal points. 
        
        # If the proximal point of the branch in evaluation is marked as a 
        # new terminal point, then we will mark it as a distal, in the sense 
        # that they will be terminal points. 
        
        if len(complete_trimming)>0:
            if p0 in cells[complete_trimming][:,1]:
                # This will be executed twice, but is cheap a we don't mind
                distal_point[point_translator[p0]] = 1 # The new terminal point will be marked as so  
                # parents_into_distals += [new_bid] #  I don't trust this; new_bid renames the current cell but here we are targeting the parent branch of the cell in view. 
 
        # We should need only to compare the end-point if belongs to the deletion points
        if not p1 in deletion_point_ids:
            # If the point is not deleted, we keep the cell
            new_cell = [point_translator[p0], point_translator[p1]] # create the new cell
            new_cells += [new_cell] # store the cell
            radius += [TestTree.branches[bid].radius] # store other fields
            length += [TestTree.branches[bid].line_length] 
            distal += [TestTree.branches[bid].distal]
            
            # If the branch was a distal branch...
            if TestTree.branches[bid].distal==1:
                # Then its end-point is a distal point as well
                distal_point[point_translator[p1]] = 1
            
            # Used to keep track of the new cell ids
            new_bid += 1

    # Transform into numpy arrays
    radius = np.array(radius)
    length = np.array(length)
    distal = np.array(distal)
    distal_point=np.array(distal_point)
    new_cells = np.array(new_cells)
    
    # The ids of the terminal points
    terminal_point_ids = np.arange(len(distal_point))[distal_point==1]
    
    for tpid in terminal_point_ids:
        matches = np.argwhere(new_cells==tpid)
        
        if len(matches)==1:
            ncid, pos = matches[0]
            if pos==1:
                distal[ncid] = 1
        else:
            print("Error when trimming the airway tree")
            print(" > The terminal point (id=%i) has issues!"%tpid)
        
    # Export tree
    trimmed_tree = io.Mesh(points=new_points, cells={"line":new_cells}, 
                           cell_data={"radius":[radius],
                                      "length":[length],
                                      "distal":[distal]},
                           point_data={"distal":distal_point})
    
    trimmed_tree.write(out_airways)
   
    if out_analysis:
        terminals = new_points[distal_point==1]
        if verbose: print("Number of subdomains: %i"%len(terminals))
        subdomains, counts, volumes = distribute_subdomains(reference_mesh, terminals)
        
        for data, name in zip([volumes,counts],["Volumes","Counts",]):
            
            if name=="Volumes":
                continue
            
            q50 = np.median(data)
            q25 = np.quantile(data,0.25)
            q75 = np.quantile(data,0.75)
            qlow = np.quantile(data,0.10)
            qhigh = np.quantile(data,0.90)
            
            if verbose:
                print("[AFTER] Evaluating %s:"%name)
                print(" Median (Q25 - Q75) : %.0f (%.0f - %.0f)"%(q50,q25,q75))
                print(" * MIN: %.0f"%np.min(data))
                print(" * Q10: %.0f"%qlow)
                print(" * Q90: %.0f"%qhigh)

        if return_out_analysis: return subdomains, counts, volumes

def determine_equivalent_resistance(tree_path,mu = 1.825e-8, verbose=True, max_attempts=50):
    
    ''' 
    Open an airway mesh in format *.vtu (or equivalent) and determine the equivalent resistance for the airway tree.
    Assumes that all flows join at the end-point and that all distal pressures are equal, something akin to a closed
    electric circuit.
    
    For large trees, the variable max_attempts may need to be enlarged. This is a limit condition for
    a while loop that gets larger as more branches are available. 
    '''
    

    # Load the tree
    tree = initialize_tree_from_vtu(tree_path)
    if verbose:
        print("Number of branches: %i"%(len(tree.branches.keys())))
        print("Number of terminal points: %i"%(len(tree.distal_points)))
    # List of branches that have to be visited yet
    pending_branches = list(tree.branches.keys())
    
    # Distal branches
    distal_branches = tree.distal_branches.copy()
    
    # Here we will hold intermediate equivalent resistances results
    resistance_indexer = {}
    attempts = 0    
    equivalent_resistance = None
    
    while  attempts < max_attempts:
    
        # Retrieve the parents for the current distal branches
        parent_branches = []
        for bid in distal_branches:
            parent_branches += [tree.branches[bid].parent]
        parent_branches = np.unique(parent_branches)
        
        # For every parent, ask whether all their children are distal branches
        for pid in parent_branches:
            
            # Retrieve children
            children = tree.branches[pid].children
            # Children is distal list
            children_check = []
            for chid in children:
                children_check += [chid in distal_branches]
        
            # True if all children are distal
            children_pass = np.all(children_check)
            
            # If not all children are ready to be computed, then skip 
            if not children_pass: continue
            
            nchildren = len(children)
            
            resistances = []
            if nchildren == 1:
                chid = children[0]
                L = tree.branches[chid].line_length 
                r = tree.branches[chid].radius
                R = 8*mu*L/(np.pi*r**4)
                if chid in resistance_indexer.keys(): R += resistance_indexer[chid]
                resistances += [R]
                
            else:
                for chid in children:
                    L = tree.branches[chid].line_length 
                    r = tree.branches[chid].radius
                    R = 8*mu*L/(np.pi*r**4)
                    if chid in resistance_indexer.keys(): R += resistance_indexer[chid]
                    resistances += [R]
                
            # Valid for N resistances (including N=1)
            equivalent_resistance = 1/np.sum([1/r for r in resistances])
            
            # Store the resistance for the parent branch in dictionary
            resistance_indexer.update({pid:equivalent_resistance})
            
            # Remove the already computed branches from the distal branches tree
            for chid in children: 
                distal_branches.remove(chid)
                pending_branches.remove(chid)
            
            # Now append the parent id who is now a distal branch as well
            distal_branches += [pid]
            
        attempts += 1
        
        if 0 in resistance_indexer:
            
            L = tree.branches[0].line_length 
            r = tree.branches[0].radius
            R = 8*mu*L/(np.pi*r**4) + resistance_indexer[0]
            equivalent_resistance = R
            
            if verbose: print("The equivalent resistance is: %.3e"%equivalent_resistance)
            break
        
        if attempts == max_attempts: 
            print("Warning: Maximum attempts (%i) reached!"%max_attempts)
            
    if verbose: print("When passing 1 L/s, the pressure drop will be %.4f (cmH2O)"%(equivalent_resistance*10.1972e6))
    
    return equivalent_resistance

    
# %%

if __name__ == "__main__" and False:
    
    '''
    SANDBOX:
        Read a previously saved mesh and recreate (as well as possible)
    the original Tree structure.
    
    '''

    import shutil 
    
    root = "/home/user/Documents/MESHING/test-data/AIRWAYS-SENSIBILIZATION/BASE-GEOMETRIES/"
    send_to = "/home/user/Documents/MESHING/test-data/AIRWAYS-SENSIBILIZATION/"
    
    path_to_airways = root + "AIRWAYS/"
    airways = path_to_airways+"skel-6.vtu"
    airway_mesh = io.read(airways)    

    cells = airway_mesh.cells_dict['line']
    points = airway_mesh.points
    TestTree = initialize_tree_from_vtu(airways)

    # The mesh which is used for the subdomain analysis
    working_mesh = "MESH-20"    
    # The folder where to send the files
    target_folder = send_to+"PACK-19/"
    
    # In this block we should trim the candidates for delection terminals;
    # Probably just delete both children and keep the parents. Otherwise,
    # just delete one of the two childs. We'll see.
    
    # Remember to modify the ids for branches and stuff
    
    # 1) Generate subdomains associated to a mesh; Study distribution herein.
    
    path_to_mesh = root+"%s/FEniCS/mesh000000.vtu"%working_mesh
    terminals = TestTree.distal_points
            
    subdomains, counts, volumes = distribute_subdomains(path_to_mesh, terminals)
    
    for data, name in zip([volumes,counts],["Volumes","Counts",]):
        
        if name=="Volumes":
            continue
        
        q50 = np.median(data)
        q25 = np.quantile(data,0.25)
        q75 = np.quantile(data,0.75)
        qlow = np.quantile(data,0.10)
        qhigh = np.quantile(data,0.90)
        
        print("[BEFORE] Evaluating %s:"%name)
        print(" Median (Q25 - Q75) : %.0f (%.0f - %.0f)"%(q50,q25,q75))
        print(" * MIN: %.0f"%np.min(data))
        print(" * Q10: %.0f"%qlow)
        print(" * Q90: %.0f"%qhigh)
    
    # These delection targets are terminals that generate small subdomains. They might
    # be merged with their pair-child into a single terminal.
    
    # Options: 
    # 1) Merge with their pair-child into a single terminal; Issue: There may be some w/o brothers
    # 2) Delection of the branch; Issue: Retraction towards the parent child may disort results
    
    # We may do both; If 1) fails then apply 2)
    
    # #These are ids from the distal_points list in the airway tree mesh
    deletion_targets = np.arange(len(terminals))[data < qlow]
    # distal_points and distal_branches are sorted correspondingly, thus we could 
    # use the distal_branches array directly
    
    # Take the distal branches from the active tree
    distal_branches = np.array(TestTree.distal_branches)
    # # Select the branches that are marked for deletion 
    # This assumes that distal_points and distal_branches follow the same ordering.
    deletion_branches = distal_branches[deletion_targets]
    
    # here we will store 
    complete_trimming = [] # Mark the parents that will become distal branches
    trimming_targets =  [] # Definitively delete these branches
    
    # Move through each one of the branches selected for deletion
    for bid in deletion_branches:
        
        # Retrieve the parent
        pid = TestTree.branches[bid].parent
        # Retrieve the list of the children
        children = TestTree.branches[pid].children.copy()
        # Remove this branch from the children list
        children.remove(bid)
        
        single_parent = len(children)==0 
        
        # Usually there are only two children, but this is not always the case
        merger = [] # Placeholder
        for child in children: # Every children but the main branch
            merger += [child in deletion_branches]
        merger = np.all(merger)      
        
        if merger:
            # Note that all children are to be deleted and these are the parent ids that will become distal pointsl
            # The children need to be removed from the distal branches and point lists.
            complete_trimming += [pid]
            trimming_targets += [bid]
            
        else:
            # Note that these are the branches ids to be deleted. At least one other children branch
            # remains connected to the parent that will remain as distal.
            trimming_targets += [bid]
            # How do we renumber?
            
    complete_trimming = np.unique(np.array(complete_trimming)) # this has parent branch ids
    trimming_targets = np.array(trimming_targets) # this has children branch ids
    
    # Invoking a new tree requires a point list and a cell list; So we need to create the new lists before-
    #TrimTree = Tree()
        
    # PARTIAL TRIMMINIG 
    deletion_point_ids = np.unique(cells[trimming_targets][:,1]) # This has point ids to be deleted
    
    point_translator = {}
    point_mask = np.zeros(len(points),dtype=bool)
    
    # This block purges the old point list and creates a new dictionary that
    # translated old ids to new ids.
    pid = 0; new_pid = 0
    while pid < len(points):
        
        # Check if point is going to be deleted
        if pid in deletion_point_ids:
            # if True (deleted point)
            point_translator.update({pid:None})
            # note that we dont update the new numbering in this scenario
        else:
            # if False (kept point)
            point_mask[pid] = True
            point_translator.update({pid:new_pid})
            new_pid +=1 # update the new numbering
        # check next point
        pid += 1
    
    # New point list
    new_points = points[point_mask]
    # Some relevant fields
    distal_point = np.zeros(len(new_points),dtype=int)
    new_cells = []
    radius = []
    length = []
    distal = []
#    parents_into_distals = []
    
    new_bid = 0
    
    # Create new cell list; Evaluate every cell from the original tree
    for bid, cell in enumerate(cells):
        
        # Extract the point ids that compose the old cell
        p0, p1 = cell # in old numbering
        
        # Cells in complete trimming are parents whose children are going to be
        # erased. Their end-points (distal) are going to be terminal points. 
        
        # If the proximal point of the branch in evaluation is marked as a 
        # new terminal point, then we will mark it as a distal, in the sense 
        # that they will be terminal points. 
        
        if len(complete_trimming)>0:
            if p0 in cells[complete_trimming][:,1]:
                # This will be executed twice, but is cheap a we don't mind
                distal_point[point_translator[p0]] = 1 # The new terminal point will be marked as so  
                # parents_into_distals += [new_bid] #  I don't trust this; new_bid renames the current cell but here we are targeting the parent branch of the cell in view. 
 
        # We should need only to compare the end-point if belongs to the deletion points
        if not p1 in deletion_point_ids:
            # If the point is not deleted, we keep the cell
            new_cell = [point_translator[p0], point_translator[p1]] # create the new cell
            new_cells += [new_cell] # store the cell
            radius += [TestTree.branches[bid].radius] # store other fields
            length += [TestTree.branches[bid].line_length] 
            distal += [TestTree.branches[bid].distal]
            
            # If the branch was a distal branch...
            if TestTree.branches[bid].distal==1:
                # Then its end-point is a distal point as well
                distal_point[point_translator[p1]] = 1
            
            # Used to keep track of the new cell ids
            new_bid += 1


    # Transform into numpy arrays
    radius = np.array(radius)
    length = np.array(length)
    distal = np.array(distal)
    distal_point=np.array(distal_point)
    new_cells = np.array(new_cells)
    
    # The ids of the terminal points
    terminal_point_ids = np.arange(len(distal_point))[distal_point==1]
    
    for tpid in terminal_point_ids:
        matches = np.argwhere(new_cells==tpid)
        
        if len(matches)==1:
            ncid, pos = matches[0]
            if pos==1:
                distal[ncid] = 1
        else:
            print("Error when trimming the airway tree")
            print(" > The terminal point (id=%i) has issues!"%tpid)
        
    # Export tree
    trimmed_tree = io.Mesh(points=new_points, cells={"line":new_cells}, 
                           cell_data={"radius":[radius],
                                      "length":[length],
                                      "distal":[distal]},
                           point_data={"distal":distal_point})
    
    terminals = new_points[distal_point==1]
    subdomains, counts, volumes = distribute_subdomains(path_to_mesh, terminals)
    
    for data, name in zip([volumes,counts],["Volumes","Counts",]):
        
        if name=="Volumes":
            continue
        
        q50 = np.median(data)
        q25 = np.quantile(data,0.25)
        q75 = np.quantile(data,0.75)
        qlow = np.quantile(data,0.10)
        qhigh = np.quantile(data,0.90)
        
        print("[AFTER] Evaluating %s:"%name)
        print(" Median (Q25 - Q75) : %.0f (%.0f - %.0f)"%(q50,q25,q75))
        print(" * MIN: %.0f"%np.min(data))
        print(" * Q10: %.0f"%qlow)
        print(" * Q90: %.0f"%qhigh)

    if not os.path.isdir(target_folder):
        
        os.mkdir(target_folder)
        TestTree.process_mesh(target_folder+"testtree.vtu",include_order=True)
        shutil.copytree(root+working_mesh+"/FEniCS-trimmed/",target_folder+working_mesh+"/FEniCS/")
        trimmed_tree.write(target_folder+"skel.vtu")
    
    new_terminals = new_points[new_cells[distal==1][:,1]]
    new_subdomains, new_counts, new_volumes = distribute_subdomains(path_to_mesh, new_terminals)
    
    #  Generate figures
    
    fig, (ax0,ax1) = plt.subplots(nrows=2,figsize=(8,6))
    
    ax0.hist(counts,alpha=0.5, label="Original")
    ax0.hist(new_counts,alpha=0.3,color='r',label="Trimmed")
    ax0.set_xlabel("Number of elements per subdomain (-)")
    ax0.set_ylabel("Frequency (-)")
    ax0.legend()
    
    
    ax1.hist(volumes,alpha=0.5, label="Original")
    ax1.hist(new_volumes,alpha=0.3,color='r',label="Trimmed")
    ax1.set_xlabel("Volume per subdomain (mm3)")
    ax1.set_ylabel("Frequency (-)")
    
    
    # In this block we should activate the generative algorithm but for only
    # a subset of the cloud point, exactly that that is associated with the
    # large distal points.
    
    # Remember to modify the ids for branches and stuff
     
    deletion_targets = np.arange(len(terminals))[data > qhigh]
    # distal_points and distal_branches are sorted correspondin
    
    print("The number of subdomains to be subdivided is: %i"%(deletion_targets.shape[0]))
    

if False:
    
    root = "C:/Users/angus/Downloads/CORNELL-GEOM/test/"
    

    path_to_mesh = root +  "MESH-20/"
    vtu_tri = path_to_mesh+"FEniCS-trimmed/boundary_markers000000.vtu"
    surf_ply = vtu_tri.split(".vtu")[0]+".ply"
    npz_cloud = path_to_mesh+"FEniCS-trimmed/cloud.npz"

    
    path_to_airways = root + "AIRWAYS/"
    airways = path_to_airways+"base-skel.vtu"
    airway_mesh = io.read(airways)    

    cells = airway_mesh.cells_dict['line']
    points = airway_mesh.points
    TestTree = initialize_tree_from_vtu(path_to_airways+"base-skel.vtu")

    
    npz = np.load(npz_cloud)
    cloud = npz['cloud']


    for i in range(3):

        TestTree.generate_sythetic_branches(cloud, surf_ply,
                                           kmeans_coherence=True)
    
        TestTree.set_strahler_order(verbose=False)
    
        
        TestTree.assign_synthetic_radii()
        
        TestTree.process_mesh(path_to_airways+"base-skel-%i.vtu"%i,include_order=True)

# %%    
if False:
    
    # Read the txt and extract a finite element mesh (lines)
    points, elem = skel_to_mesh(skel_txt,skel_mesh=None)
    
    # Transform the skeleton mesh (traslation and scaling) using lung meshes
    points = transform_skel(mat_lung, vtu_lung, nifti_lung, points, elem, 
                            outname=path_to_mesh+"raw_skel.vtu",
                            scale_airways_with_affine=False)
    
    # Retrieve topological information
    out = classify_skel_nodes(points, elem)
    
    # Generate a tree using the top of the trachea as seed
    seed = out['inlet_id']
    airway_tree = Tree(seed, points,elem, branch_length_threshold=1.50)
    # Initialize the skeleton analysis
    airway_tree.activate_seed() 
    # Create branches from skeletonization
    airway_tree.grow_tree() 
    # Determine radius from the skeletonization
    airway_tree.radius_from_skeletonization(skel_txt, corr_txt, nifti_lung)
    # Generate synthetic branches
    for i in range(8):
        print("Generating branches, loop %i"%(i+1))
        airway_tree.generate_sythetic_branches(cloud, surf_ply,
                                               kmeans_coherence=True)
        # Assign Strahler order
        airway_tree.set_strahler_order()
        # Assign radius following Tawhai et al. (2004)
        airway_tree.assign_synthetic_radii()
        

        
# %% # determining equivalent resistance 

if False:
    
    ''' This block compares the pressure drop in an arbitrary airway tree as solved in the linear 
    system from the Poiseuille flow resistances to the equivalent resistance solution as implemented
    in a routine elsewhere. The pressure drops should match always. So far it does'''
    
    
    test_path =  "C:/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/BASE-GEOMETRIES/AIRWAYS/dummy-skel.vtu"
    

    pipe = Pipeline(test_path)
    Q0 = -1e6
    Pdistal = {p:0.0 for p in pipe.distal_point_ids}
    rs = 1e-5
    drs = {e:rs for e in pipe.distal_elem_ids}
    pipe.assemble_linear_system(Pdistal,Q0=Q0,upstream_resistance=1e-20, downstream_resistances=drs)
    pipe.solve_linear_system()
    print("The pressure drop in the system is %.4f (cmH2O)"%(pipe.x[pipe.N]*10.1972))
    
    determine_equivalent_resistance(test_path)
    qs = pipe.retrieve_qs()
    
    minq = None; maxq = None;
    
    for q in qs:
        if minq is None:
            minq = np.abs(qs[q])
        else:
            minq = np.min([minq,np.abs(qs[q])])
      
        if maxq is None:
            maxq = np.abs(qs[q])
        else:
            maxq = np.max([maxq,np.abs(qs[q])])   
        
    print("min(q)/max(q): %.2f"%(minq/maxq))
    print("min(q): %.0f"%minq)
    print("max(q): %.0f"%maxq)

    subdomains, counts, volumes = distribute_subdomains(reference_mesh, terminals)

# %% creating a synthetic test tree according to a sketch

if False:
    
    ''' This block creates a synthetic airway tree using a few points, it can be modified to 
    create alternative trees by hand but that is too troublesome. It might be useful someday
    though so don't delete it '''
    
    
    points = np.array([[5.0,15.0,0.0], # 0
                      [ 5.0,12.0,0.0], # 1
                      [ 5.0,10.0,0.0], # 2      
                      [ 2.0, 7.0,0.0], # 3
                      [ 8.0, 7.0,0.0], # 4    
                      [ 2.0, 4.0,0.0], # 5
                      [ 6.0, 4.0,0.0], # 6     
                      [10.0, 3.0,0.0], # 7
                      [ 0.0, 0.0,0.0],     
                      [ 5.0, 0.0,0.0], # 9
                      [ 8.0, 0.0,0.0], # 10
                      [13.0, 0.0,0.0]])
    
    ien = [[0,1], # 0
           [1,2], # 1
           [2,3], # 2
           [2,4], # 3
           [3,5], # 4
           [4,6], # 5
           [4,7], # 6
           [5,8], # 7
           [5,9], # 8
           [7,10], # 9
           [7,11], # 10          
           ]
    
    distal_point = np.array([0,0,0,0,0,0,1,0,1,1,1,1])
    distal_cell = [np.array([0,0,0,0,0,1,0,1,1,1,1])]
    order_cell = [np.array([3,3,2,2,2,1,2,1,1,1,1])]
    radius_cell = [np.array([4.0,3.5,3.0,2.8,2.0,2.0,1.5,1.0,1.3,0.5,0.5])]
    length_cell = [np.array([np.linalg.norm(points[i]-points[o]) for (i,o) in ien])]
    
    mesh = io.Mesh(points=points,cells={"line":ien},point_data={"distal":distal_point},
                   cell_data={"order":order_cell,"radius":radius_cell,
                              "length":length_cell, "distal":distal_cell})
    
    dummy_airways =  "C:/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/BASE-GEOMETRIES/AIRWAYS/dummy-skel.vtu"
    
    mesh.write(dummy_airways)


# %% 

if __name__ == '__main__' and False:
    
    import matplotlib.patches as mpatches
    
    fig, axes = plt.subplots(ncols=3,figsize=(15,6),dpi=200)

    # target geometries
    # LENOVO LAPTOP
    paths = ["C:/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/BASE-GEOMETRIES/AIRWAYS/skel-1.vtu",
             "C:/Users/angus/Downloads/CORNELL-GEOM/test/AIRWAYS-base/base-skel.vtu"]
    # MSI LAPTOP
    paths = ["C:/Users/Agustin/Documents/TemporalData/skel-1.vtu",
             "C:/Users/Agustin/Documents/TemporalData/base-skel.vtu"]
    
    # x-axis control, Strahler order or airway generation "Order" "Generation"
    strat = "Generation" # Order Generation

    
    # color control
    colors = ["tab:red","tab:blue"]
    
    # boxplot control
    dx = 0.20
    x0 = -0.10
    
    mu = 1.825e-8
    # test_path =  "C:/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/BASE-GEOMETRIES/AIRWAYS/dummy-skel.vtu"
    
    # figure limits
    ylims = {"res":(1e-10,1e-4),
             "length":(0,150),
             "radius":(-1,4)}
        
    fields = ["length","radius","res"]    
    
    # legend-related
    patch0 = mpatches.Patch(color=colors[0],label="Unstable tree")
    patch1 = mpatches.Patch(color=colors[1],label="Stable tree")
    patches = [patch0,patch1]
    
    # generate analysis and figure
    for e,(test_path,color) in enumerate(zip(paths,colors)):
        
        # load tree
        tree = initialize_tree_from_vtu(test_path)
        # determine generations
        generations = np.zeros(len(tree.branches.keys()),dtype=int)
        for bid in tree.branches:
            if bid == 0:
                generations[bid] = 0
            else:
            
                branch = tree.branches[bid]
                pid = branch.parent
                nsisters = len(tree.branches[pid].children)
                if nsisters>1:
                    generations[bid] = generations[pid]+1
                else:
                    generations[bid] = generations[pid]
            
        # extract ready-to-use data
        mesh = io.read(test_path)
        order = mesh.cell_data['order'][0]
        radius = mesh.cell_data['radius'][0]
        length = mesh.cell_data['length'][0]
        
        # adjust labels
        renamer = {"res":"Poiseuille resistance (kPa.s/mm3)",
                   "radius":"Log(Diameter)",
                   "length":"Airway length (mm)"}
        
        # determine resistance
        resistances = 8*mu*length/(np.pi*radius**4)
        
        if strat == "Generation":
            print("Using '%s' as stratificator"%strat)
            stratificator = generations
            xlabel = "Generation (-)"
            xmin = 0
        elif strat == "Order":
            print("Using '%s' as stratificator"%strat)
            stratificator = order
            xlabel = "Strahler order (-)"
            xmin = 1
        else:
            stratificator = None
            print("Invalid stratificator '%s'"%strat)
    
        # determine xlimits
        strata = np.unique(stratificator)
    
        # sort fields per strata
        sorter = {field:{s:[] for s in strata} for field in fields}
        
    
        # extract data
        for s in strata:
            mask = stratificator==s
            for field,data in zip(fields,[length,np.log(2*radius),resistances]):
                sorter[field][s] = data[mask]
        
        # determine area per strata
        sorter.update({"area":{s:np.pi*np.sum(sorter["radius"][s]**2) for s in strata}})


        # generate subplots
        for ax, field in zip(axes,fields):
            
            print("field: %s - start"%field)
            
            for s in strata:
                
                n = len(sorter[field][s])
                #ax.scatter([o-0.1]*n,sort_resistances[o],color="k")
                box = ax.boxplot(sorter[field][s], positions=[s+dx*e+x0],vert=True,showfliers=False, patch_artist=True)
            
                for b in box['boxes']:
                    b.set_alpha(0.3)
                    b.set_color(color)
                for b in box['whiskers']: b.set_alpha(0.5)
                for b in box['caps']: b.set_alpha(0.5)       
                for b in box['medians']: b.set_color(color)
    
    
            if field == "radius" and strat=="Generation":
                if max(strata) < 10:
                    ax.plot(strata, strata*-0.0438+1.3094, ls="--",color="k")
                    ax.fill_between(strata,strata*-0.0438+1.3094-0.0494,strata*-0.0438+1.3094+0.0494,color="k",alpha=0.2)

        
            if field == "length" and strat=="Generation":
                xmax = np.max(strata) if np.max(strata)<5 else 5
                xrange = np.arange(0,xmax)
                ax.plot(xrange,-7.5186*xrange+37.556, ls="--",color="k")


            yscale = "log" if field == "res" else "linear"
            ax.set_yscale(yscale)
            ax.set_xticks(strata)
            ax.set_xticklabels(["%i"%s for s in strata])
            xlims = (xmin-0.5, max(strata)+0.5)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims[field])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(renamer[field])
            
            print("field: %s - end"%field)
        
        plt.subplots_adjust(hspace=0.40)
        plt.legend(handles=patches)
        
# %%

if False:
    
    
    fig, ax = plt.subplots(figsize=(6,6),dpi=200)
    
    for s in strata:
        ax.scatter(s,sorter['area'][s],color="k")
    ax.set_yscale('log')
    ax.set_ylabel("Area (mm2)")
    ax.set_xlabel(xlabel)
    ax.set_ylim((1,1e4))
    N = max(sorter['res'].keys())
    N = np.ceil(N/2).astype(int)*2
    ax.set_xticks(np.arange(0,N+1,2))
    ax.set_xticklabels(np.arange(0,N+1,2))
    
    
# %%

if __name__=='__main__' and False:
    ''' Simple implementation of Pedleys model and comparison with Poiseuille's model '''
    
    from scipy.optimize import root
    
    # Airway geometry
    L = 160 # in mm
    D = 14 # in mm
    r = D/2 
    
    # Air properties
    mu = 1.825e-5 # air dynamic viscosity in Pa.s
    rho = 1.204 # air density in kg/
    
    # Prescribed flow
    Q = 1 # L/s
    Q_ = Q*1e-3 # m3/s
    
    # Pedley's model
    gamma = 0.327
    
    # Poiseuille resistance cmH2O/(L/s)
    R_Poiseuille = mu*1e-3*10.1972*8/np.pi*L/r**4*1e6
    
    # Reynolds number
    Re = 2*rho*np.abs(Q_)/(mu*np.pi*(r*1e-3))
    
    R_Pedley = np.max([gamma*np.sqrt(Re*2*r/L)*R_Poiseuille,R_Poiseuille])
    
    print("Reynolds number: %.1f"%Re)
    print("Poiseuille resistance: %.3f cmH2O"%R_Poiseuille)
    print("Pedley's resistance: %.3f cmH2O"%R_Pedley)
    
    
    def poiseuille(r,L,mu=1.825e-5):
         return mu*1e-3*10.1972*8/np.pi*L/r**4*1e6
    
    def pedley(r,L,Q,mu=1.825e-5):
        Re = 2*rho*np.abs(Q_)/(mu*np.pi*(r*1e-3))
        R_Poiseuille = poiseuille(r,L)
        return np.max([gamma*np.sqrt(Re*2*r/L)*R_Poiseuille,R_Poiseuille])
    
    # Define an arbitrary pipe
    L1 = 30; L2 = 25; D1 = 12; D2 = 9; r1=D1/2; r2=D2/2
    # Equation to balance pressure in terminals using Pedley's resistances
    eq = lambda q : pedley(r1,L1,q)*q-pedley(r2,L2,Q_-q)*(Q_-q)
    # Solve
    sol = root(eq,3e-4)
    q1 = sol.x[0]
    q2 = Q_-q1
    # Present results
    print("Pedley's flow distribution: ")
    print("  > q1/q0: %.2f"%(q1/Q_))
    print("  > q2/q0: %.2f"%(q2/Q_))
    
    A = np.matrix([[1,1],[poiseuille(r1,L1), -poiseuille(r2,L2)]])
    b = np.matrix([[Q_],[0]])
    qs = np.linalg.solve(A,b)
    q1_lam = qs[0,0]
    q2_lam = qs[1,0]
    
    print("Poiseuille's flow distribution: ")
    print("  > q1/q0: %.2f"%(q1_lam/Q_))
    print("  > q2/q0: %.2f"%(q2_lam/Q_))
    
    rped0=pedley(r,L,Q_)
    rped1=pedley(r1,L1,q1)
    rped2=pedley(r2,L2,q2)
    rped_eq = 1/(1/rped1+1/rped2)+rped0
    dp_ped = Q_*rped_eq
    
    print("Pedley's pressure drop: ")
    print("  > dP (a): %.3f"%(dp_ped*1000))
    print("  > dP (b): %.3f"%((Q_*rped0+q1*rped1)*1000))
    
    
    
    print("Poiseuille's pressure drop: ")
    dp_poiseuille = Q_*poiseuille(r,L)+q1_lam*poiseuille(r1,L1)
    
    print("  > dP (a): %.3f"%(dp_poiseuille*1000))


# %% # determining equivalent resistance 

if False and __name__ == "__main__":
    
    ''' This block compares the pressure drop in an arbitrary airway tree as solved in the linear 
    system from the Poiseuille flow resistances to the equivalent resistance solution as implemented
    in a routine elsewhere. The pressure drops should match always. So far it does'''
    
    root = "C:/Users/angus/Downloads/AIRWAYS-SENSIBILIZATION/"
    airways_path =  root+"/PACK-01-0/skel.vtu"
    reference_mesh = root+"/PACK-01-0/MESH-25/FEniCS/mesh000000.vtu"

    TestTree = initialize_tree_from_vtu(airways_path)
    terminals = TestTree.distal_points
       
    # Determine subdomains volume
    _, _, volumes = distribute_subdomains(reference_mesh, terminals)
    total_volume = np.sum(volumes)
    weights = volumes/total_volume

    # Mount pipeline
    pipe = Pipeline(airways_path)
    Q0 = -1e6
    Pdistal = {p:0.0 for p in pipe.distal_point_ids}
    dr = 1e-4
    ur = 5e-17
    
    downs = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]
    
    minqs = []
    maxqs = []
    ratios = []
    pdrops = []
    
    downs = [10**(-10+e) for e in range(9)]
    
    for dr in downs:
    
        drs = {e:dr*weights[i] for i,e in enumerate(pipe.distal_elem_ids)}
        pipe.assemble_linear_system(Pdistal,Q0=Q0,upstream_resistance=ur, downstream_resistances=drs)
        pipe.solve_linear_system()
        print("The pressure drop in the system is %.4f (cmH2O)"%(pipe.x[pipe.N]*10.1972))
        pdrops += [pipe.x[pipe.N]*10.1972]
        #determine_equivalent_resistance(test_path)
        qs = pipe.retrieve_qs()
        
        minq = None; maxq = None;
        
        for q in qs:
            if minq is None:
                minq = np.abs(qs[q])
            else:
                minq = np.min([minq,np.abs(qs[q])])
          
            if maxq is None:
                maxq = np.abs(qs[q])
            else:
                maxq = np.max([maxq,np.abs(qs[q])])   
            
        minqs += [minq]
        maxqs += [maxq]
        ratios += [minq/maxq ]
    
    import matplotlib.patches as mpatches
    
    colors = ["tab:blue","tab:orange"]
    patch0 = mpatches.Patch(color=colors[0],label="Maximum flow")
    patch1 = mpatches.Patch(color=colors[1],label="Minimum flow")
    patches = [patch0,patch1]
    
    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(9,6),dpi=200)
    
    for e in range(len(downs)):
        ax1.scatter(e,minqs[e],color=colors[1])
        ax1.scatter(e,maxqs[e],color=colors[0])
    
        ax2.scatter(e,pdrops[e],color="k")
    
    ax2.set_ylabel("Pressure drop (cmH2O)")
    ax1.set_ylabel("Flow (mm3/s)")
    ax1.legend(handles=patches)
    for ax in [ax1,ax2]:
        ax.set_xticks(range(9))
        ax.set_xticklabels(["$10^{-%i}$"%(10-e) for e in range(9)])
        ax.set_xlabel("Scaling factor for resistance (-)")
    
# %%
''' Implementation of the Pedley's model '''



if __name__ == '__main__' and False:
    
    from numpy.linalg import norm


    reference_mesh = "C:/Users/Agustin/Documents/TemporalData/mesh000000.vtu"
    airways_path = "C:/Users/Agustin/Documents/TemporalData/skel-2.vtu"

    TestTree = initialize_tree_from_vtu(airways_path)
    terminals = TestTree.distal_points
       
    # Determine subdomains volume
    _, _, volumes = distribute_subdomains(reference_mesh, terminals)
    total_volume = np.sum(volumes)
    weights = volumes/total_volume




    Nitmax = 40
    

    pipe = Pipeline(airways_path)
    Q0 = -1e6
    tol = 1e-12
    Pdistal = {p:0.0 for p in pipe.distal_point_ids}
    rs = 1e-4
    drs = {e:rs*weights[i] for i,e in enumerate(pipe.distal_elem_ids)}
    ur = 4e-7
    drs= None
    
    pipe.assemble_linear_system(Pdistal,Q0=Q0,upstream_resistance=ur, downstream_resistances=drs)
    pipe.solve_linear_system()
    
    qs = pipe.retrieve_qs()
    minq = None; maxq = None
    for key in qs.keys():
        q = np.abs(qs[key])
        if key == 0:
            minq = q
            maxq = q
        else:
            minq = min([minq,q])
            maxq = max([maxq,q])
    
    print("  * [Poiseuille] Ratio (minQ/maxQ): %.3f"%(minq/maxq))
    print("  * [Poiseuille] Pressure drop: %.4f (cmH2O)"%(pipe.x[pipe.N]*10.1972))
    
    x_old = pipe.x.copy()
    err = 1 # dummy
    n = 0
    
    errs = []
    
    while err>tol:
        
        # Update matrix and solve system
        pipe.update_pedley_resistances(downstream_resistances=drs,
                                       upstream_resistance=ur)
        pipe.solve_linear_system()
        # Determine and store error
        err = norm(x_old-pipe.x)/norm(pipe.x)
        errs += [err]
        # Update temporal variable and counter
        x_old = pipe.x.copy()
        n+=1
        # Exit iteration loop if maximum iterations are reached
        if n > Nitmax:
            print("Maximum iterations reached")
            break
        
    # Find out the ratio between the extreme points in the terminal points
    qs = pipe.retrieve_qs()
    minq = None; maxq = None
    for key in qs.keys():
        q = np.abs(qs[key])
        if key == 0:
            minq = q
            maxq = q
        else:
            minq = min([minq,q])
            maxq = max([maxq,q])
    
    print("  * [Pedley] Ratio (minQ/maxQ): %.3f"%(minq/maxq))
    print("  * [Pedley] Pressure drop: %.4f (cmH2O)"%(pipe.x[pipe.N]*10.1972))
    