# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:20:58 2023

@author: Agustin
"""

import numpy as np
import meshio as io
import os

# %% define points

# originally defined but outdated, they must be displaced
distal_points = np.array([( 80,130,165),
                             ( 90,100,210),
                             (180,140,170),
                             (170,140,240)])

# selected visually from skel7
distal_points = np.array([(-37.7,-11.3,44.4), #
                          (-30.5, -36.6,1.2), # 
                          ( 28.6, -7.9, 45.9),
                          (52.2,3.4,-20.2)]) # go inward Y

dz = np.array([(0,0,40)])

intermediate_points = np.vstack([(distal_points[0,:]+distal_points[1,:])*0.5+dz,
                    (distal_points[2,:]+distal_points[3,:])*0.5+dz])

juncture = (intermediate_points[0,:]+intermediate_points[1,:])*0.5+dz

origin = juncture+dz*3

# define point list and element connectivity list

points = np.vstack([origin,
          juncture,
          intermediate_points,
          distal_points])

elem = np.array([[0, 1],
                [1, 2],
                [1, 3],
                [2, 4],
                [2, 5],
                [3, 6],
                [3, 7]])

length = []
for p0,p1 in elem:
    length += [np.linalg.norm(points[p0]-points[p1])]
length = np.array(length)

point_data = {"distal":np.array([0,0,0,0,1,1,1,1]),}
cell_data = {"distal":[np.array([0,0,0,1,1,1,1])],
             "radius":[np.array([5,3,3,1,1,1,1],dtype=float)],
             "length":[length]}

# generate mesh and write it to some address
mesh = io.Mesh(points=points,cells={"line":elem},
               point_data=point_data,cell_data=cell_data)
mesh.write("../testing-data/stable/airway-tree/simple-skel.vtu")


# %%


# originally defined but outdated, they must be displaced
distal_points = np.array([( 40.3,-38.4,-41.0),
                          ( 51.1,-17.1,-28.7),
                          ( 29.2, 12.4,  9.8),
                          ( 26.3,  3.3, 52.7),
                          (-35.4,-45.2,-41.0),
                          (-36.5,-37.5,-0.36),
                          (-37.7,-11.3, 44.3),
                          (-35.5,-29.8, 32.8)])

dz = np.array([(0,0,20)])

intermediate_points = np.vstack([(distal_points[0,:]+distal_points[1,:])*0.5+dz,
                                 (distal_points[2,:]+distal_points[3,:])*0.5+dz,
                                 (distal_points[4,:]+distal_points[5,:])*0.5+dz,
                                 (distal_points[6,:]+distal_points[7,:])*0.5+dz,])
                                
upper_points = np.vstack([(intermediate_points[0,:]+intermediate_points[1,:])*0.5+dz,
                          (intermediate_points[2,:]+intermediate_points[3,:])*0.5+dz,])

juncture = (upper_points[0,:]+upper_points[1,:])*0.5+dz

origin = juncture+dz*3

# define point list and element connectivity list

points = np.vstack([origin,
                    juncture,
                    upper_points,
                    intermediate_points,
                    distal_points])

elem = np.array([[0, 1],
                 [1, 2],
                 [1, 3],
                 [2, 4],
                 [2, 5],
                 [3, 6],
                 [3, 7],
                 [4, 8],
                 [4, 9],
                 [5, 10],
                 [5, 11],
                 [6, 12],
                 [6, 13],
                 [7, 14],
                 [7, 15]])

length = []
for p0,p1 in elem:
    length += [np.linalg.norm(points[p0]-points[p1])]
length = np.array(length)

point_data = {"distal":np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]),}
cell_data = {"distal":[np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])],
             "radius":[np.array([5,3,3,1,1,1,1,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3],dtype=float)],
             "length":[length]}

# generate mesh and write it to some address
mesh = io.Mesh(points=points,cells={"line":elem},
               point_data=point_data,cell_data=cell_data)
mesh.write("../testing-data/stable/airway-tree/regular-skel.vtu")

