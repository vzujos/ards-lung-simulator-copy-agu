# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:12:49 2026

@author: angus
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import meshio as io

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

# We'll interpolate intensities towards a different mesh
subject = 5
protocol = 'ARDSnet'
key = (subject,protocol)
r_root = "D:/ARAOS-PIGS/CORNELLU-PIGS-GROUPED/"
mesh_quality = 'medium'

s_mesh_path = "C:/Users/angus/Downloads/CORNELL-NEWGEO/PIG%i/%s/"%key+"%s/"%mesh_quality

in_registration_mesh = s_mesh_path+"reg_anim_ready.vtu"

mesh = io.read(in_registration_mesh)
mass = mesh.point_data['Mass']
xyz = mesh.points
nROI = 10

directions = {'BA':np.mat([0.,0.,1.]).T,
              'VD':np.mat([0.,1.,0.]).T,}

direction = 'VD'
nrois = 10

_, ids = IsoVolumetricSegmentation(directions[direction], mass, xyz, nROI)


outmesh = io.Mesh(points=xyz,cells=mesh.cells_dict, point_data={direction:ids})
outmesh.write(s_mesh_path+"temp.vtu")