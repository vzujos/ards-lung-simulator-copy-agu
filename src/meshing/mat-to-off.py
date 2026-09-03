# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:03:22 2023

@author: angus
"""

from scipy.io import loadmat
import meshio as io


def tri_mat2off(path_mat,path_off):
    mat = loadmat(path_mat)
    points = mat['node']
    elem = mat['elem']-1
    out = io.Mesh(points=points,cells={'triangle':elem})
    out.write(path_off)
    

# %%

# airways processing
root = 'C:/Users/angus/Downloads/CORNELLU-PIGS/TreatmentGroup-B/Subject-02/03-ARDSnet-MV/MESH-2/'
root = 'C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/MESH-8/'
#root = 'D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/PIG2/BaselineInjury/MESH-5/'

if True:
    
    geom = "lung" #"lung" "aw"# or airways
    
    # declare directions
    folder = ''
    path = root+folder
    path_mat = path+'coarse_%s.mat'%geom
    path_off = path+"coarse_%s.off"%geom
    
    tri_mat2off(path_mat,path_off)
    