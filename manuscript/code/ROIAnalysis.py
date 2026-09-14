#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:01:59 2018

@author: pablo
"""

import numpy as np
import matplotlib.pyplot as plt
import meshio as io

def ISO_ROIAnalysis(mesh_file, field, nro_rois, direction, usePhi = False):
	'''
	Purpose: prepare fields of subject in directory for using computeROIvalues
	Input  : mesh_file:  Pointer to the subject's lungs mesh (.npz) file
			 field	: desire field
			 nro_rois : number of ROIs
	Output  :	rois_stats: stats for each ROI (mean and STD)
				 vol_roi  : volume for each ROI
				 mass_roi : mass for each ROI
	'''
		
   # Load mesh
	mesh= io.read(mesh_file)
	# Read data
	field = mesh.point_data[field]
	w = mesh.point_data['Mass']   
	xyz = mesh.points

	if usePhi:
		xyz += mesh.point_data['DispField']
	
	curve, id_roi = IsoVolumetricSegmentation(direction, w, xyz,nro_rois)


	roi_results = computeISO_ROIvalues(field, w, id_roi)

	return roi_results, id_roi


def ISO_ROIAnalysis_COVID(xyz,field, mass, nro_rois, direction, copd=False):
	'''
	Purpose: prepare fields of subject in directory for using computeROIvalues
	Input  : mesh_file:  Pointer to the subject's lungs mesh (.npz) file
			 field	: desire field
			 nro_rois : number of ROIs
	Output  :	rois_stats: stats for each ROI (mean and STD)
				 vol_roi  : volume for each ROI
				 mass_roi : mass for each ROI
	'''

	curve, id_roi = IsoVolumetricSegmentation(direction, mass, xyz,
											  nro_rois)
	roi_results = computeISO_ROIvalues(field, mass, id_roi)

	return roi_results, id_roi



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

	plot_cum_mass = False	 # Set True if cummulative mass plot is desired
	if plot_cum_mass:
#		print min(d), max(d)
#		print np.round((np.sort(d)[lim_index]-min(d))/(max(d)-min(d)), 2)
		plt.plot(np.sort(d), sum_mass)
		plt.plot(np.sort(d)[lim_index], sum_mass[lim_index], 'o')
		plt.xlabel('z [voxel size]')
		plt.ylabel('Cumulative volume [voxel size$^3$]')
#		print [0, np.round((np.sort(d)[lim_index]-min(d))/(max(d)-min(d)), 2)]
		plt.figure()
		plt.plot(np.ones(11), np.hstack((0, np.round((np.sort(d)[lim_index] -
													 min(d))/(max(d)-min(d)),
													 2))), 'o')
	id_roi_output = [None]*len(id_roi)
	for i, j in zip(id_roi, np.argsort(d)):
		id_roi_output[j] = int(i)

	return [sum_mass, np.sort(d)], np.array(id_roi_output)


def computeISO_ROIvalues(field, w, id_roi):
	'''
	Purpose :Compute statistical measures by dividing mesh into desired
			 ROIs
	Input   : field: array with FE field
			  y	: gravitational height of mesh
			  w	: weight of each node
			  rho  : rho of each element
			  nro_rois: desired number of ROIs

			  Output  :	rois_stats: stats for each ROI (mean and STD)
						   vol_roi  : volume for each ROI
						   mass_roi : mass for each ROI
	'''

	m = np.zeros(max(id_roi)+1)
	s = np.zeros(max(id_roi)+1)
	for i, j, k in zip(field, w, id_roi):
		m[k] += i*j
	m = m/np.sum(w)*(max(id_roi)+1)
	for i, j, k in zip(field, w, id_roi):
		s[k] += j*(i - m[k])**2
	s = (s/np.sum(w)*(max(id_roi)+1))**0.5

	return np.array([[i, j] for i, j in zip(m, s)])


def Get_Intersect(n1, p1, n2, p2, n3, p3):
	intersect = np.linalg.solve(np.c_[n1, n2, n3].T,
								np.array([np.dot(n1, p1), np.dot(n2, p2),
										  np.dot(n3, p3)]))
	return intersect


def RotatePlanes(n_ori, n_rot, p_rot, n1, p1, n2, p2):

	""" Purpose:  - Given a rotation from n_ori to n_rot, rotates two planes
					by the same rotation using as origin the 3-planes
					intersection point
		Input:	- n_ori:  Normal of the original plane
				  - n_rot:  Normal of the rotated plane
				  - p_rot:  Point defining the rotated plane
				  - n1, p1: Normal and point defining first plane
				  - n2, p2, Normal and point defining the second plane
		Output:   - n1_rec, n2_rec:  Normals of the recorrected planes
				  - intersect: Intersection of the three planes
	"""

	intersect = Get_Intersect(n_rot, p_rot, n1, p1, n2, p2)
	v = np.cross(n_ori, n_rot)
	s = np.linalg.norm(v)
	c = np.dot(n_ori, n_rot)
	vx = np.mat([[0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
	R = np.eye(3) + vx + np.dot(vx, vx)*(1 - c)/s**2

	n1_rec = np.array(np.dot(R, n1)).reshape(3,)
	n1_rec = n1_rec/np.linalg.norm(n1_rec)
	n2_rec = np.array(np.dot(R, n2)).reshape(3,)
	n2_rec = n2_rec/np.linalg.norm(n2)

	return n1_rec, n2_rec, intersect


def PlaneToFunction(n_, p_, i, x):
	""" Purpose:  -Returns x_i value from a plane function in terms of
				   the 2 next coordinates
		Input:	- n_:  Normal of the plane
				  - p_:  Point of the plane
				  - i :   n of the coordinate to obtain its value
				  - x :   Remaining coordinates in vector e.g. i=2 >>
						  (x_1, False, x_3); i=1 >> (False, x_2, x_3)
	"""

	x_ = p_[i]
	for j in [0, 1, 2]:
		if j != i:
			x_ -= n_[j]*(x[j] - p_[j])/n_[i]
	return x_


if __name__ == '__main__':
	print('Debugging...')
