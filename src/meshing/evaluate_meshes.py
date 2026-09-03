# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:17:53 2025

@author: angus
"""

import meshio as io
import numpy as np
import matplotlib.pyplot as plt
import os

parameters = {"PIG2-ARDSnet":{"Tsyr":0.435,
                           "Tpausa":0.560,
                           "Texp":1.870,
                           "Pplat":21.7,
                           "PEEP":11.0,
                           "Ppeak":26.8,
                           "Signal path":'model2/signals/PIG2-ARDSnet.npz',
                           "Target volume":0.3527,
                           "EELV":1936.33,
                           },
           "PIG3-ARDSnet":{"Tsyr":0.455,
                           "Tpausa":0.435,
                           "Texp":1.100,
                           "Pplat":26.3,
                           "PEEP":11.3 ,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG3-ARDSnet.npz',
                           "Target volume":0.3870,
                           "EELV":1745.23 ,
                           },
           "PIG4-ARDSnet":{"Tsyr":0.48,
                           "Tpausa":0.28,
                           "Texp":1.73,
                           "Pplat":18.5,
                           "PEEP": 10.7,
                           "Ppeak":24.8,
                           "Signal path":'model2/signals/PIG4-ARDSnet.npz',
                           "Target volume":0.411,
                           "EELV":2407.44,
                           },
           "PIG5-ARDSnet":{"Tsyr":0.375,
                           "Tpausa":0.375,
                           "Texp":1.25,
                           "Pplat":20.6,
                           "PEEP": 10.8,
                           "Ppeak":33.8,
                           "Signal path":'model2/signals/PIG5-ARDSnet.npz',
                           "Target volume":0.4011,
                           "EELV":2072.99,
                           },
           "PIG6-ARDSnet":{"Tsyr":0.540,
                           "Tpausa":0.265,
                           "Texp":1.340,
                           "Pplat":22.7,
                           "PEEP": 10.7,
                           "Ppeak":25.7,
                           "Signal path":'model2/signals/PIG6-ARDSnet.npz',
                           "Target volume":0.2995,
                           "EELV":2314.06,
                           }
           }

def tetrahedron_volume_from_array(verts):
    """
    Compute the volume of a tetrahedron from a (4, 3) array of vertices.

    Parameters:
    verts: np.ndarray
        A (4, 3) array where each row represents a vertex (x, y, z) of the tetrahedron.

    Returns:
    float
        The volume of the tetrahedron.
    """
    assert verts.shape == (4, 3), "Input must be of shape (4, 3)"
    
    a = verts[1] - verts[0]
    b = verts[2] - verts[0]
    c = verts[3] - verts[0]
    
    volume = np.abs(np.dot(a, np.cross(b, c))) / 6.0
    return volume

def determine_cell_volumes(points, cells):
    volumes = []
    for cell in cells:
        volumes += [tetrahedron_volume_from_array(points[cell])]
    return np.array(volumes)


root = "C:/Users/angus/Downloads/CORNELL-NEWGEO/"
pig_no = 4
mesh_name = "medium-fine"
case_name = "FEniCS"
mesh_path = root+"PIG%i/ARDSnet/%s/%s/"%(pig_no,mesh_name,case_name)
file_name = "mesh000000.vtu"
file_path = mesh_path+file_name

pass_flag = True

if not os.path.isdir(mesh_path) and pass_flag:
    pass_flag = False
    print("Mesh directory '%s' not found"%mesh_path)

if not os.path.isfile(file_path) and pass_flag:
    pass_flag = False
    print("Mesh directory '%s' not found"%mesh_path)
    
# 

eelv = parameters['PIG%i-ARDSnet'%pig_no]['EELV']*1e3 # cm3 to mm3

if pass_flag:
    
    print("Every file was successfully found")
    print("Proceeding to computations")
    
    mesh = io.read(file_path)
    points = mesh.points
    cells = mesh.cells_dict['tetra']
    
    print("Number of points: %i"%points.shape[0])
    print("Number of cells: %i"%cells.shape[0])
    
    volumes = determine_cell_volumes(points,cells)
    
    v50 = np.median(volumes)
    vmax = volumes.max()
    vmin = volumes.min()
    v95 = np.quantile(volumes,0.95)
    v05 = np.quantile(volumes,0.05)
    v75 = np.quantile(volumes,0.75)
    v25 = np.quantile(volumes,0.25)
    viqr = v75 - v25
    ratio_vmaxvmin = vmax/vmin
    ratio_v95v05 = v95/v05
    eelv_to_v50_ratio = eelv/v50
    print("Median volume (IQR): %.0f (%.0f)"%(v50,viqr))
    print("MinMax volume ratio: %.0f"%ratio_vmaxvmin)
    print("95/05 volume ratio: %.1f"%ratio_v95v05)
    print("EELV in L: %.2f"%(eelv*1e-6))
    print("EELV-to-V50 ratio: %.0f"%eelv_to_v50_ratio)