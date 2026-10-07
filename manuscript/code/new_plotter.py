#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:18:30 2019

@author: user
"""
import statsmodels.stats.weightstats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as sp_sts
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from xlwt import Workbook
from PIL import Image
import numpy as np
import pandas as pd


#------------------------------------------------------------------------------

## @package new_plotter
# This package contains routines dedicated to generate all the plots needed by
# BioMechLib in any of its instances. Everything related to a plot moves 
# through here. It has some redundances that are being worked on. This is not
# intended to be directly used but it should be code 'under the hood'.


## Creates an scatter plot given compatible arrays 'x' and 'y'. The parameters
# required are related to the configuration of the plot and can be set at the 
# method 'compareUsingScatter' from the branch 'globalOperations'
# @param x: Lists of numpy arrays containing the data to for the x-coordinate.
# @param y: Lists of numpy arrays containing the data to for the y-coordinate.
# @param xbounds: Tuple that contains the bounds for the x axis.
# @param ybounds: Tuple that contains the bounds for the y axis.
# @param legends: The legend for each curve involved. 
# @param colors: The colors to be used in a format compatible with matplotlib.
# @param markers: The markers symbols in a format compatible with matplotlib.
# @param xlabel: The name to be placed for the x-axis.
# @param ylabel: The name to be placed for the y-axis.
# @param save: Boolean flag to activate the saving of the image.
# @param dst: Route to the saved image, including it's name excepting its
# format.
# @param fmt: The format of the image such as 'pdf' or 'png'.
# @param figsize: The size in inches of the image. Default is (8,6).
# @param amplification: Boolean flag to refer that ki amplification is being 
# plotted if True. Default is True.
# @param includeSubjectLinearFit: Activates the fitting of linear curves for 
# each dataset contained. Default is False.
# @param includeGlobalLinearFit: Activates the application of a global fit
# for all the data contained in the x and y variables. Default is True. 

def scatterPlot(x,y, xbounds, ybounds, legends, colors, markers,xlabel, ylabel,
	save= False, dst = "", fmt = "pdf", figsize = (8,6), amplification = True,
	includeSubjectLinearFit = False, includeGlobalLinearFit = True):
	
	fig, ax = plt.subplots(figsize = figsize)
	names = legends # copy to use later

	ax.set_xlim(xbounds);	ax.set_ylim(ybounds)
	x_base = np.linspace(xbounds[0],xbounds[1])
	
	if includeSubjectLinearFit:
		n_leg = len(legends)
		legends = legends * 2
		for i in range(n_leg):
			 legends[i] += " Fit"
	
	if includeGlobalLinearFit:
		global_x = []; global_y = []
		dummy = ["Global Fit"]
		n_leg = len(legends)
		for i in range(n_leg):
			 dummy.append(legends[i])
		legends = dummy

	if not amplification:
		ax.axhline(y = 0.012, xmin = xbounds[0]-5.0, xmax=xbounds[1], color = "r",
			  linewidth = 1.0, linestyle='--')
		dummy = ["Ki = 0.012"]
		n_leg = len(legends)
		for i in range(n_leg):
			 dummy.append(legends[i])
		legends = dummy

	c_t = 1
	D_x = (xbounds[1] - xbounds[0])/100;	D_y = (ybounds[1] - ybounds[0])/100
	
	for c,m,r in zip(colors,markers,range(len(x))):
		ax.scatter(x[r],y[r], color=c,marker=m)

		if includeGlobalLinearFit:
			for dx, dy in zip (x[r],y[r]):
				global_x.append(dx); global_y.append(dy)
				
		if includeSubjectLinearFit:			
			m, c, rvalue, pvalue, stderr = sp_sts.linregress(x[r],y[r])
#			ax.text(xbounds[0] + D_x * 3, ybounds[1] - D_y * 5.0 * c_t,
#		   names[r] + ": $R^2$ = %4.2f \quad p = %4.3f"%(rvalue**2,pvalue),fontsize=16)
			
			c_t += 1
			ax.plot(x_base,m*x_base+c, label="Fit")
	
	if includeGlobalLinearFit:
		m, c, rvalue, pvalue, stderr = sp_sts.linregress(global_x,np.array(global_y))
		ax.text(xbounds[0] + D_x * 32, ybounds[1] - D_y * 5.0,
		   "Global Fit: $R^2$ = %4.2f \quad p = %4.3f"%(rvalue**2,pvalue),fontsize=16)
		ax.plot(x_base,m*x_base+c)
	
	ax.set_xlabel(xlabel);	ax.set_ylabel(ylabel)
	plt.subplots_adjust(top=0.90, bottom=0.1, left=0.10, right=0.75)
	ax.legend(legends, bbox_to_anchor=(0.5, 0.0, 0.88, 0.8))
	if save: plt.savefig(dst+"."+fmt,format=fmt)
	
#------------------------------------------------------------------------------
	
## Generates a different histogram per ROI. Returns a bigger image which holds
# each of the single generated distribution plots. Used to look into the shape
# of the distribution within a ROI for a given field.
# @param dicROI: Data holder that contains the data to be plotted. It is
# generated in the 'generateROIHistogram' method from the Mesh objects.
# package and in few other similars. 
# @param bins: The bins to be used in the histograms.
# @param figsize: The size of the image. Default is (10,30).
# @param fmt: The format of the saved image.
# @param dst: Route to the saved image, including its name and excepting its
# format.
# @param weightedHistogram: Boolean flag which activates the used of weighted
# statistics in the histogram creation. Default is True.
# @param save: Boolean flag which activates the saving of the image. Default is
# True.
# @param hist_range: Internal variable which sets the range of the histogram.
# Default is None.
# @param n_rois: The number of ROIs used in the histograms. Default is 10.

def ROIHistogram(dicROI, bins, figsize = (10,30),  fmt='pdf', 
			 dst = "", weightedHistogram = True, save = False, 
			 hist_range = None, n_rois=10):


	fig, axes = plt.subplots(nrows=(n_rois)//2,ncols=2, figsize=figsize)
	
	av = np.zeros(n_rois)
	ylims = []
	
	for ax, r in zip(axes.flatten(), range(n_rois)):
		
		d = dicROI[r][0];		m = dicROI[r][1]
	
		if weightedHistogram:
			ax.hist(d,bins,weights=m,density=True,\
			  rwidth=0.9,alpha=0.75, range=hist_range)
		else:
			ax.hist(d,bins,density=True, rwidth=0.9, \
			  alpha=0.75, range=hist_range)
			
		av[r] = np.average(d,weights=m)
		ax.grid(axis='y', alpha=0.2)
		ax.set_title("ROI %i"%(r+1))
		ax.set_xlabel("Volumetric Strain [\%]")
		ax.set_ylabel("Frecuency [-]")
		ylims.append(ax.get_ylim()[1])
			
	ymax = np.ceil(np.max(ylims)*1000)/1000
	
	for ax, r in zip(axes.flatten(), range(n_rois)):
		ax.set_ylim([0.0,ymax]); ymin = 0.0
		ax.set_xlim((bins[0],bins[-1]));
		
		dy = ymax - ymin
		xlim = ax.get_xlim(); xmin = xlim[0]; 
		xmax = xlim[1]; dx = xmax-xmin; 
	
		meanText = "Weighted average: %5.3f"%av[r]+"\%"
		ax.text(xmin+0.55*dx,ymin+0.85*dy,meanText,fontsize=10)

	plt.subplots_adjust(top=0.72, bottom=0.1, left=0.05, right=0.94, \
					  hspace=0.40, wspace=0.30)
	
	if save: plt.savefig(dst+"."+fmt,format=fmt)

#------------------------------------------------------------------------------

## Used to create a single histogram that contains multiple stacked bars for a 
# given field such as Aeration or Delta Gas Fraction. Bins must be given which
# contain the limits of each relevant category.
# @param masterDict: Holds the information being plotted. This dictionary is
# created in 'globalAerationHistograms_perState' and other similar methods 
# from the database package. 
# @param kind: Defines the kind of histogram. Kind 0 represents aeration 
# histograms and kind 1 represents 'Delta Gas Fraction'.
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param figsize: The size of the image in inches. Default is (10,10).
# @param bins: The bins for the image being created. Default is [].
# @param readyDict: Dictionary that skips the processing of the raw data that 
# comes from masterDict. Default is None.
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format.
# @param returnData: Returns the information that constructed the plot. Default
# is False.

def singleBarHistogram(masterDict, kind, fmt='pdf',figsize = (10,10), bins=[],
					   readyDict = None, save = False, dst="",
					   returnData = False, n_rois = 10):

	# kind = 0 => AERATION HISTOGRAM
	# kind = 1 => DELTA GAS FRACTION
	
	if kind == 0:
		legend = ["Hyperinflated tissue", "Normally aerated tissue",\
			   "Poorly aerated tissue", "Non aerated tissue"]
		suptitle = "Aeration distribution per ROI"
		
		if len(bins) == 0:
			bins =  [-1000.0, -900.0,-500.0, -100.0, +100.0]
	
	elif kind == 1:
		legend = ["$\Delta\,GF < 0\%$",	"$0\% <\Delta\,GF \leq 20\%$",\
		"$20\% <\Delta\,GF \leq 40\%$", "$40\% <\Delta\,GF \leq 60\%$", \
		"$\Delta\,GF > 60\%$"]
		suptitle = "$Delta\,Gas\,Fraction\,[\Delta GF]\,per\,ROI$"
		if len(bins) == 0:
			bins = [-1.0, 0.0, 0.2, 0.4, 0.6, 1.0]
	else:
		print("Aborting routine 'singleBarHistogram' due to kind error.")
		return
		
	#Generate plotting structures.
	fig,ax = plt.subplots(nrows=2,ncols=1,figsize=figsize)
		
	# Recover information from the master dictionary.
	dirs = ['VD','BA']
	
	# Generate useful variables to be used later.
	rois = np.arange(n_rois);	x_axis = np.linspace(1, n_rois, n_rois)
				
	# Tuning parameters for caption location
	y1 = -0.10
	
	holder = {}
	
	for (d,a) in zip(dirs,ax):	
			
		# Captions are used later to introduce text into the image.
		captions = getCaptions(d)
		
		ranges = []
		for l in range(len(bins)-1):	
			ranges.append([])
		
		if readyDict == None: # this is a quicker dictionary
	
			for r in rois:
					
				hist, bin_edges = np.histogram(masterDict[d][r][0],bins, \
									   weights=masterDict[d][r][1])
				hist /= sum(hist)
				cl = 0
				for l in range(len(bins)-1):
					ranges[cl].append(hist[cl]); cl += 1
		else:
			ranges = readyDict[d]
			
		holder.update({d:ranges})
		
		a.set_xticks(np.arange(11)+1)
		a.set_xticklabels([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
				
		a.text(-0.4,y1,captions[0], horizontalalignment='center')
		a.text(11.4,y1,captions[1], horizontalalignment='center')
	
		if kind == 0:
			a.bar(x_axis, ranges[0], color = "k") 
			a.bar(x_axis, ranges[1],bottom=ranges[0], color="k", alpha = 0.5, edgecolor="k")
			a.bar(x_axis, ranges[2],bottom=[i+j for i,j in zip(ranges[0],ranges[1])], color = "k", alpha = 0.25, edgecolor="k")
			a.bar(x_axis, ranges[3],bottom=[i+j+k for i,j,k in zip(ranges[0],ranges[1],ranges[2])], color = "w", edgecolor="gray", alpha = 0.999)
			
		elif kind == 1:
			a.bar(x_axis, ranges[0], color = "w", edgecolor = "k") 
			a.bar(x_axis, ranges[1],bottom=ranges[0], fill = False, hatch = "....")
			a.bar(x_axis, ranges[2],bottom=[i+j for i,j in zip(ranges[0],ranges[1])], color = "grey", edgecolor = "k")
			a.bar(x_axis, ranges[3],bottom=[i+j+k for i,j,k in zip(ranges[0],ranges[1],ranges[2])], fill = False, hatch = "////")
			a.bar(x_axis, ranges[4],bottom=[i+j+k+l for i,j,k,l in zip(ranges[0],ranges[1],ranges[2],ranges[3])], color = "k", edgecolor="k")
					
		a.set_title("%s to %s direction"%(captions[0],captions[1]))
		a.set_xlabel("ROI")
		a.set_ylabel("Fraction [-]")
			
		a.legend(legend, loc='best', bbox_to_anchor=(0.5, 0.0, 0.98, 1.0))
	
	fig.suptitle(suptitle, fontsize=20)
	plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.70, hspace=0.30, wspace=0.40)

	if save:
		 plt.savefig(dst+"."+fmt,format=fmt)
		 if kind == 0:
			 recolumns = {0:"Hyperinflated tissue", 
				1:"Normally aerated tissue", 
				2:"Poorly aerated tissue",
				3:"Non-aerated tissue"}
			 
			 writer = pd.ExcelWriter(dst+".xls")
			 for k in holder.keys():
				 df = pd.DataFrame(np.array(holder[k]).T)
				 df.rename(columns=recolumns).to_excel(writer,k)
			 writer.save()
 
	
	
	
	if returnData: return holder
#------------------------------------------------------------------------------

## Used to create a single histogram that contains multiple stacked bars for
# a given field such as Aeration or Delta Gas Fraction. Bins must be given
# which contain the limits of each relevant category.
# @param masterDict: Holds the information being plotted.
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param figsize: The size of the image in inches. Default is (15,20).
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format.

def singleBarHistogram_ROI(masterDict,  fmt='pdf',figsize = (15,20), dst="",
					   save = False):
	
	legend = ["Hyperinflated tissue", "Normally aerated tissue",\
			   "Poorly aerated tissue", "Non aerated tissue"]

		
	# Recover information from the master dictionary.
	dirs = ['VD','BA']
	states = ['00h','03h','06h','09h','12h']

	for d in dirs:
		
		fig,ax = plt.subplots(nrows=5,ncols=2,figsize=figsize)
		captions = getCaptions(d)
		fig.suptitle("%s to %s direction"%(captions[0],captions[1]),fontsize=20)
	
		# Tuning parameters for caption location
		x_axis = np.linspace(1, 5, 5)
		for (i,a) in zip(np.arange(10),ax.flatten()):
			holder = []
			for r in range(4):
				sameCompartment = []
				for s in states:
					sameCompartment.append(masterDict[s][d][r][i])
				holder.append(sameCompartment)

			a.bar(x_axis, holder[0], color = "k") 
			a.bar(x_axis, holder[1],bottom=holder[0], color="k", alpha = 0.5, edgecolor="k")
			a.bar(x_axis, holder[2],bottom=[i+j for i,j in zip(holder[0],holder[1])], color = "k", alpha = 0.25, edgecolor="k")
			a.bar(x_axis, holder[3],bottom=[i+j+k for i,j,k in zip(holder[0],holder[1],holder[2])], color = "w", edgecolor="gray", alpha = 0.999)
			a.set_title("ROI %i"%(i+1))
			a.set_xticks(np.arange(5)+1)
			a.set_xticklabels(states)	
			a.set_xlabel("State")
			a.set_ylabel("Fraction [-]")
			if i%2!=0:
				a.legend(legend, loc='best', bbox_to_anchor=(-0.25, 0.85, 0.0, 0.0))

		plt.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.90, hspace=0.40, wspace=1.0)

		if save: plt.savefig(dst+"_%s"%d+"."+fmt,format=fmt)

#------------------------------------------------------------------------------
		
## Returns snaps of different slices of a nifti array. The directions are
# "A": Axial,, "S": Sagittal and "C": Coronal.
# @param image: A nibabel image. 
# @param bounds: A tuple that contains three different tuples, one for each
# dimension of the image.
# @param snaps: The number of images. Default is 6.
# @param width: The width of the image in inches. A characteristic length for
# the image. Default is 3.
# @param direction: The selected direction. "A": Axial, "S": Sagittal, "C":
# Coronal. Default is "A".
# @param threshold: The limits for the image intensity. Default is (-1000,0).

def showSlices(image, bounds, snaps = 6, width = 3, direction = "A", 
			   threshold=(-1000,0)):

	if direction not in ["A","S","C"]:
		print("   > Chosen direction '%s' not in allowable directions."%direction)
		print("   > Aborting method.")
	
	idata = image.getData()
	affine = image.affine
	aspectRatio = np.abs(affine[2,2]/affine[1,1])

	xmin = int(bounds[0][0]);	xmax = int(bounds[0][1])
	ymin = int(bounds[1][0]);	ymax = int(bounds[1][1])
	zmin = int(bounds[2][0]);	zmax = int(bounds[2][1])

	plt.gray()
#	
	if direction == "S":	
		figsize = (width*2,width*aspectRatio*3)
		fig, ax = plt.subplots(nrows=3,ncols=2,figsize=figsize)
		i = 0; j = 0
	
		for n in np.linspace(xmin,xmax,num=snaps+2).astype(int)[1:(snaps+1)]:
			ax[j][i].imshow(np.flip(idata[n,ymin:ymax,zmin:zmax],1).T,\
				  aspect=aspectRatio, vmax=threshold[1], vmin=threshold[0])
			ax[j][i].set_axis_off()
			ax[j][i].set_title("X: %i out of %i"%(n,image.shape[2]))
			i +=1
			if i == 2:
				j += 1;	i = 0
				
	if direction == "C":	
		figsize = (width*2,width*aspectRatio*3)
		fig, ax = plt.subplots(nrows=3,ncols=2,figsize=figsize)
		i = 0; j = 0
		
		for n in np.linspace(ymin,ymax,num=snaps+2).astype(int)[1:(snaps+1)]:
			
			ax[j][i].imshow(np.flip(idata[xmin:xmax,n,zmin:zmax],1).T, \
					 aspect=aspectRatio, vmax=threshold[1], vmin=threshold[0])
			ax[j][i].set_axis_off()
			ax[j][i].set_title("Y: %i out of %i"%(n,image.shape[2]))
			i +=1
			if i == 2:
				j += 1;	i = 0

	if direction == "A":
		figsize = (width*2,width*3)
		i = 0; j = 0
		fig, ax = plt.subplots(nrows=3,ncols=2,figsize=figsize)
		
		for n in np.linspace(zmin,zmax,num=(snaps+2)).astype(int)[1:(snaps+1)]:
			
			ax[j][i].imshow(idata[xmin:xmax,ymin:ymax,n].T, aspect=1.0, \
				  vmax=threshold[1], vmin=threshold[0])
			ax[j][i].set_axis_off()
			ax[j][i].set_title("Z: %i out of %i"%(n,image.shape[2]))
			i +=1
			
			if i == 2:
				j += 1;	i = 0
				
	plt.subplots_adjust(top=0.95, bottom=0.05, \
					 left=0.02, right=0.98, hspace=0.10, wspace=0.05)
	plt.show()

#------------------------------------------------------------------------------
## Takes in a dictionary and plots a bar graph showing the amount of air
# contained in every state of a set of images. The information can be 
# complemented with information from experimental measurements. 
# @param dic: The holder of the information.
# @param figsize: The size of the image in inches. Default is (10,8).
# @param bars: Boolean flag which activates using bars instead of continuous 
# lines. Default is True.
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param figsize: The size of the image in inches. Default is (10,8).
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format. Default is "".
# @param bbox_to_anchor: Used to configure the placement of the nomenclature. 
# It's not intended to be changed but it may be necessary if the figsize
# changes. Default is (0.985,0.895).
# @param root: Currently deprecated. TODO: Remove. 
# @param measurements: List of measured data to be compared with the current
# set. Default is None.

def singleSubjectMaskVolumeInTime(dic, labels, figsize = (10,8), bars = True,\
								 save =False,bbox_to_anchor=(0.985,0.895), \
								 dst = "image", fmt="pdf", root="a", \
								 measurements = None):
	
	null = 0.0 if bars else None
	
	times = dic['times'];	exists = dic['exists'];	valid = dic['valid']
	vE = dic['mVE']; vI = dic['mVI'];	dV = vI-vE
	
	npoints = len(vE)
	x = np.linspace(1,npoints,npoints)

	fig, ax = plt.subplots(nrows=1,ncols=1, figsize=figsize)
	ax.set_ylim(0,2000)
	ax.set_xlim(0.5,npoints+0.5)
	ax.set_xticks(np.arange(1,6,1))
	legend = ["Exp (Valid)", "Exp (Invalid)", "Insp (Valid)",\
		    "Insp (Invalid)", "Delta (Valid)", "Delta (Invalid)"]
	ax.set_xticklabels(labels)
	ax.set_ylabel('Volume [ml]');	ax.set_xlabel('Time')
	
	ax2 = ax.twinx() # Clone the first 
	
	d1 = []; d2 = []; d3 = [];  d1_ = []; d2_ = []; d3_ = []; GVS = []
	
	for (l, e, v) in zip(labels, exists,valid):
		
		if not e: # if it wasn't at 'times' then it's not available.
			d1_.append(None); d2_.append(None); d3_.append(null)
			d1.append(None) ; d2.append(None) ; d3.append(null)
			GVS.append(None)
			continue
		
		i = times.index(l)
		GVS.append(100.*(vI[i]-vE[i])/vE[i])

		if not v:
			d1_.append(vE[i]); d2_.append(vI[i]); d3_.append(dV[i])
			d1.append(None) ; d2.append(None) ; d3.append(null);
		else:
			d1.append(vE[i]); d2.append(vI[i]); d3.append(dV[i])
			d1_.append(None) ; d2_.append(None) ; d3_.append(null)			
	
	ax.plot(x,d1,color='#ff0000', linewidth = 0, marker = "o")
	ax.plot(x,d1_,color='w',linewidth=0,marker="o",markeredgecolor='#ff0000')

	ax.plot(x,d2,color='#0000ff', linewidth = 0, marker = "o")
	ax.plot(x,d2_,color='w', linewidth=0,marker="o",markeredgecolor='#0000ff')
	
	ax2.plot(x,GVS,color='y',linewidth=1.0,marker="o",markeredgecolor='y')
	ax2.set_ylim(0,150)
	ax2.set_ylabel("Global Volumetric Strain [\%]")
	
	if bars:
		ax.bar(x,d3,hatch = "...", fill=False)
		ax.bar(x,d3_,hatch = "///", fill=False)
	else:
		ax.plot(x,d3,color='#208000', linewidth = 0, marker = "o")
		ax.plot(x,d3_,color='w',linewidth=0,marker="o",markeredgecolor='#208000')
	
	ax.set_title("Mask volume for different states")	

	plt.subplots_adjust(top=0.90, bottom=0.15, left=0.10, right=0.70, \
					  hspace=0.40, wspace=0.30)
	
	ax.legend(legend, loc='best', bbox_to_anchor=bbox_to_anchor, \
			 bbox_transform=plt.gcf().transFigure)
	
	if save: plt.savefig(dst+"."+fmt,format=fmt)
	
#------------------------------------------------------------------------------

## Takes in a dictionary and plots a summary for volume indicators at different 
# states. 
# @param data: The holder of the data. 
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param figsize: The size of the image in inches. Default is (10,8).
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format. Default is "".
# @param bbox_to_anchor: Used to configure the placement of the nomenclature. 
# It's not intended to be changed but it may be necessary if the figsize
# changes. Default is (0.985,0.895).

def airDistributionPerState(data,figsize=(10,8), save=False, dst="", fmt="pdf",
							bbox_to_anchor=(0.98,0.895), GVS_range = (-20, 20), 
							air_range = (0, 2000)):

	aE = data['aE']; 	aI = data['aI']; 	wE = data['wE']; 	wI = data['wI']
	times = data['times'];	vE = aE + wE;	vI = aI + wI
	width = 0.25; dw = width*0.55
	GVS = (vI-vE)/vE*100.0
	
	npoints = len(aE);	x = np.linspace(1,npoints,npoints)
	fig, ax = plt.subplots(nrows=1,ncols=1, figsize=figsize)
	ax2 = ax.twinx() # Clone the first 

	ax.set_ylim(air_range)
	ax.set_xlim(0.5,npoints+0.5)
	ax.set_xticks(np.arange(1,npoints+1,1))
	
	ax.bar(x-dw, wE, color = "whitesmoke", width=width, edgecolor="k") 
	ax.bar(x-dw, aE,bottom=wE, color = "grey", width=width, edgecolor="k")
	

	ax.bar(x+dw, wI, color = "whitesmoke", width=width,  hatch = "//",edgecolor="k") 
	ax.bar(x+dw, aI,bottom=wI, color = "grey", width=width,hatch = "//", edgecolor="k")
	
	ax2.plot(x,GVS,color='r',linewidth=1.0,marker="o", markeredgecolor='k')
	ax2.set_ylim(GVS_range)
	ax2.set_ylabel("Global Volumetric Strain [\%]")
	
	ax.set_xticklabels(times)
	ax.set_ylabel('Volume [ml]');	ax.set_xlabel('State')
	
	legend = ["Tissue @ Exp","Air @ Exp", "Tissue @ Insp", "Air @ Insp"]
	
	ax.set_title("Summary for volume indicators at different states")	

	
	ax.legend(legend, loc='best', bbox_to_anchor=bbox_to_anchor, \
			 bbox_transform=plt.gcf().transFigure)
	ax2.legend(["Global Volumetric Strain [Seg]"])

	plt.subplots_adjust(top=0.90, bottom=0.15, left=0.10, right=0.70, \
					  hspace=0.40, wspace=0.30)

	if save: plt.savefig(dst+"."+fmt,format=fmt)

	
#------------------------------------------------------------------------------

## Takes in a dictionary and plots a summary for volume indicators at different 
# states. 
# @param figsize: The size of the image in inches. Default is (10,8).
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format. Default is "".
# @param bbox_to_anchor: Used to configure the placement of the nomenclature. 
# It's not intended to be changed but it may be necessary if the figsize
# changes. Default is (0.985,0.895).
# @param root: Currently deprecated. TODO: Remove. 
# @param bars: Use bars instead of a line. Default is True.
# @param dic: The data holder. 
def singleSubjectAirVolumeInTime(dic, figsize = (10,5), bars = True,  \
								 save =False,bbox_to_anchor=(0.98,0.895), \
								 dst = "image", fmt="pdf", root="a"):

	if root not in ["a","w"]: return
	title = "Air" if root == "a" else "Water"	
	null = 0.0 if bars else None
	width = 0.25; dw = width*0.55
	
	labels = ['00h','03h','06h','09h','12h']
	times = dic['times'];	exists = dic['exists'];	valid = dic['valid']
	aE = dic['%sE'%root]; aI = dic['%sI'%root];	dA = np.abs(aI-aE)

	npoints = len(aE)
	x = np.linspace(1,npoints,npoints)

	fig, ax = plt.subplots(nrows=1,ncols=1, figsize=figsize)
	ax.set_ylim(0,1600)
	ax.set_xlim(0.5,npoints+0.5)
	ax.set_xticks(np.arange(1,6,1))
	legend = ["Air @ Exp", "Air @ Insp",  "Numerical $\Delta V$","Experimental $\Delta V$"]
	ax.set_xticklabels(labels)
	ax.set_ylabel('Volume [ml]');	ax.set_xlabel('State')
	
	ax.plot(x+dw,aE,color='b', linewidth = 0, marker = "o")
	ax.plot(x+dw,aI,color='r', linewidth = 0, marker = "o")
	ax.bar(x+dw,dA, fill=False, width=width)
	
	if 'VT_measurements' in dic.keys():
		VTm = dic['VT_measurements']
		ax.bar(x-dw,VTm, hatch = "//",fill=False, width=width)

		
	ax.set_title("%s volume"%title)	

	ax.legend(legend, loc='best', bbox_to_anchor=bbox_to_anchor, \
			 bbox_transform=plt.gcf().transFigure)
	
	plt.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.75, \
					  hspace=0.40, wspace=0.30)

	if save: plt.savefig(dst+"."+fmt,format=fmt)

#------------------------------------------------------------------------------

## Generates a plot for the whole group of subjects. Will plot the volume 
# at inspiration, expiration and the variation of the volume for all the 
# different states within the database.
# @param dic: The data holder
# @param addScatter: Boolean flag which activates the plotting of the different
# raw data in form of an scatter plot. Default is False.
# @param fillBetween: Activates the plot of the STD in a shaded area next to 
# the mean lines. 
# @param figsize: The size of the image in inches. Default is (10,5).
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format. Default is "".
# @param bbox_to_anchor: Parameter that moves the legend within the figure 
# space. Default is (0.99,0.895).
	
def AllSubjectsVolumeInTime(dic, figsize = (10,5), addScatter = False, \
							 fillBetween = True, bbox_to_anchor=(0.99,0.895), 
							 dst = "", fmt="pdf", save=False):

	keys = dic['keys']	
	npoints = len(keys)
	
	x = np.linspace(1,npoints,npoints)
	
	vE_mean = dic['mVE_mean']
	vI_mean = dic['mVI_mean']
	dV = vI_mean-vE_mean
	
	vE_std = dic['mVE_std']
	vI_std = dic['mVI_std']
	
	legend = ["$V_{E,\,MEAN}$", "$V_{I,\,MEAN}$", "$V_{E,\,STD}$", 
		   "$V_{I,\,STD}$", "$\Delta V$"]
	
	fig, ax = plt.subplots(nrows=1,ncols=1, figsize=figsize)
	ax.set_xlim(0.5,npoints+0.5)
	ax.set_xticks(np.arange(1,6,1))
	ax.set_xticklabels(keys)
	ax.set_ylabel('Volume [ml]');	ax.set_xlabel('Time')
	ax.set_ylim(0,2000)
	
	c1 = '#ff0000'; c2 = '#0000ff'
	ax.plot(x,vE_mean,color=c1, linewidth = 1.5, marker = "o", linestyle='--')
	ax.plot(x,vI_mean,color=c2, linewidth = 1.5, marker = "o", linestyle='--')
	
	if fillBetween:
		ax.fill_between(x,vE_mean-vE_std,vE_mean+vE_std,color=c1,linewidth=1.0, alpha = 0.2)
		ax.fill_between(x,vI_mean-vI_std,vI_mean+vI_std,color=c2,linewidth=1.0, alpha = 0.2)

	if addScatter:
		for (x0, k) in zip(x, keys):
			shape = len(dic[k][0]); x0_ = np.full(shape, x0)
			ax.scatter(x0_, dic[k][0], marker= "o", c=c1)
			ax.scatter(x0_, dic[k][1], marker= "o", c=c2)

	ax.bar(x,dV,hatch = "/", fill=False, width=0.4)
	ax.set_title("Mean mask volume at different states")
	
	ax.legend(legend, loc='best', bbox_to_anchor=bbox_to_anchor, \
			 bbox_transform=plt.gcf().transFigure)
	
	plt.subplots_adjust(top=0.90, bottom=0.15, left=0.10, right=0.80, \
					  hspace=0.40, wspace=0.30)

	if save: plt.savefig(dst+"."+fmt,format=fmt)


# =============================================================================

## Generates a matrix of multiple dual plots. The plot is configured in 
# the database package. This is only intended to automatize the plotting step.
# This method uses a color palette and is less complex than 
# subjectWithMultipleFields_v2.
# @param data: A dictionary that holds the data. 
# @param subject: The subject identifier.
# @param direction: The direction in study. Either 'VD' or 'BA'. 
# @param valid: A numpy array with boolean flags that refers to whether the
# information contained in data should be plotted.
# @param figsize: The size of the image in inches. Default is (10,20).
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format. Default is "".
# @param ND_Ki: Flag that refers to whether is the raw Ki (False) or the Ki
# amplification is being plotted (True). Default is False.
7
def subjectWithMultipleFields_v1(data,subject,direction,valid,figsize =(10,20), \
							 fmt='pdf', dst = "", save=False, ND_Ki = False ):

	apnd = "_AK" if ND_Ki else ""
	
	fields = data.keys()
	possibleStates = ['00h','03h','06h','09h','12h']
	colors = ["r","b","y"]
	captions = getCaptions(direction)
	
	r_range = [0,2,4,6,8,1,3,5,7,9]
	
	# prepare information
	master = {}
	for roi in r_range:
		handle = {}
		for f in fields:
			means = [];	stds = []; x = []; bottom = []; top = []; 
			x_invalid = []; means_invalid = []; norm_m = []; norm_s = []
			
			for p in possibleStates:
				counter = possibleStates.index(p)
				if p in data[f].keys():
					
					if (not valid[counter]) and f != 'Ki':
						x_invalid.append(float(counter)+1.0)
						means_invalid.append(data[f][p][0][roi])
						continue
					
					x.append(float(counter)+1.0)
					means.append(data[f][p][0][roi])# mean
					stds.append(data[f][p][1][roi])# std
					bottom.append(data[f][p][0][roi]-data[f][p][1][roi])
					top.append(data[f][p][0][roi]+data[f][p][1][roi])
					norm_m.append(data[f][p][0][roi]/data[f]['00h'][0][roi])
					norm_s.append(data[f][p][1][roi]/data[f]['00h'][0][roi])

			handle.update({f:(means,stds,np.array(x),np.array(bottom),\
							 np.array(top), x_invalid, means_invalid, \
							 np.array(norm_m), \
							 np.array(norm_m)+np.array(norm_s), \
							 np.array(norm_m)-np.array(norm_s))})
	
		master.update({roi:handle})
	
	ycount = 0;	xcount = 0;	count = 0
	
	fig, ax = plt.subplots(nrows=5,ncols=2, figsize=figsize)
	
	for roi in r_range:		

		ax[xcount,ycount].set_xlim(0.5,5.5)
		ax[xcount,ycount].set_xticks(np.arange(1,6,1))
		ax[xcount,ycount].set_xticklabels(possibleStates)
		ax[xcount,ycount].set_ylim(-50,200)
		ax[xcount,ycount].set_title("ROI %i"%(roi+1))
		ax2 = ax[xcount,ycount].twinx()
		
		if ND_Ki:
			ax2.set_ylim(0.0,10)
		else:
			ax2.set_ylim(0.0,0.10)
			
		handle = master[roi]	
	
		for (f,c) in zip(fields,colors):
			
			if f != "Ki":
			
				ax[xcount,ycount].plot(handle[f][2],handle[f][0],marker="o",\
					linestyle='--', linewidth = 1.5, color = c)
				ax[xcount,ycount].fill_between(handle[f][2],handle[f][3], \
					   handle[f][4], color = c, linewidth=1.0, alpha = 0.2)
				ax[xcount,ycount].set_ylabel(getTitle(f))

			else:
				
				# if the lines don't cross
				if not ND_Ki: 
					ax2.plot(handle[f][2],handle[f][0],linewidth=1.5,marker="o", \
					linestyle='--', color=c)
					
					ax2.fill_between(handle[f][2],handle[f][3],handle[f][4],\
							color = c, linewidth=1.0, alpha = 0.2)
				else:
					ax2.plot(handle[f][2],handle[f][7],linewidth=1.5,marker="o", \
					linestyle='--', color=c)
					
					ax2.fill_between(handle[f][2],handle[f][8],handle[f][9],\
							color = c, linewidth=1.0, alpha = 0.2)
					
				ax2.set_ylabel(getTitle(f,ND_Ki=ND_Ki))

			if f != "Ki":
				if len(handle[f][5]) != 0:
					ax[xcount,ycount].plot(handle[f][5],handle[f][6],color=c, \
					   linewidth = 0.0,marker="x")
				
		count += 1
		xcount += 1
		
		if xcount == 5:
			xcount = 0; ycount += 1
	
	plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.90, \
					  hspace=0.30, wspace=0.60)

	if save: plt.savefig(dst+apnd+"."+fmt,format=fmt)

#------------------------------------------------------------------------------

## Generates a matrix of multiple dual plots. The plot is configured in 
# the database package. This is only intended to automatize the plotting step.
# @param data: A dictionary that holds the data. 
# @param subject: The subject identifier.
# @param direction: The direction in study. Either 'VD' or 'BA'. 
# @param valid: A numpy array with boolean flags that refers to whether the
# information contained in data should be plotted.
# @param figsize: The size of the image in inches. Default is (10,20).
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format. Default is "".
# @param ND_Ki: Flag that refers to whether is the raw Ki (False) or the Ki
# amplification is being plotted (True). Default is False.
# @param ki_mask: Aditional masking array for the Ki data. Default is None.

def subjectWithMultipleFields_v2(data,subject,direction,valid,figsize =(10,20),
				 fmt='pdf', dst = "", save=False, ND_Ki = False, ki_mask=None):
	
	apnd = "_NEW" if ND_Ki else ""
	
	fields = data.keys()
	possibleStates = ['00h','03h','06h','09h','12h']
#	colors = ["r","b"]
	colors = ["k","k"]
	alphas = [0.5, 0.9]
	captions = getCaptions(direction)
	r_range = [0,2,4,6,8,1,3,5,7,9]
	
	# prepare information
	master = {}
	for roi in r_range:
		handle = {}
		for f in fields:
			means = [];	stds = []; x = []; bottom = []; top = []; 
			x_invalid = []; means_invalid = []; norm_m = []; norm_s = []
			
			for p in possibleStates:
				counter = possibleStates.index(p)
				if p in data[f].keys():
					
					if (not valid[counter]) and f != 'Ki':
						x_invalid.append(float(counter)+1.0)
						means_invalid.append(data[f][p][0][roi])
						continue
					
					x.append(float(counter)+1.0)
					means.append(data[f][p][0][roi])# mean
					stds.append(data[f][p][1][roi])# std
					bottom.append(data[f][p][0][roi]-data[f][p][1][roi])
					top.append(data[f][p][0][roi]+data[f][p][1][roi])
					norm_m.append(data[f][p][0][roi]/data[f]['00h'][0][roi])
					norm_s.append(data[f][p][1][roi]/data[f]['00h'][0][roi])

			handle.update({f:(means,stds,np.array(x),np.array(bottom),\
							 np.array(top), x_invalid, means_invalid, \
							 np.array(norm_m), \
							 np.array(norm_m)+np.array(norm_s), \
							 np.array(norm_m)-np.array(norm_s))})
	
		master.update({roi:handle})
	
	ycount = 0;	xcount = 0;	count = 0
	
	fig, ax = plt.subplots(nrows=5,ncols=2, figsize=figsize)
	
	for roi in r_range:		

		ax[xcount,ycount].set_xlim(0.5,5.5)
		ax[xcount,ycount].set_xticks(np.arange(1,6,1))
		ax[xcount,ycount].set_xticklabels(possibleStates)
		ax[xcount,ycount].set_ylim(-50,200)
		ax[xcount,ycount].set_title("ROI %i"%(roi+1))
		ax2 = ax[xcount,ycount].twinx()
		
		ax2.set_ylim(0.0,8)
			
		handle = master[roi]	
	
		for (f,c,a) in zip(fields,colors, alphas):
			
			if f != "Ki":
			
				ax[xcount,ycount].plot(handle[f][2],handle[f][0],marker="o",\
					linestyle='--', linewidth = 1.5, color = c, alpha = a )
				ax[xcount,ycount].fill_between(handle[f][2],handle[f][3], \
					   handle[f][4], color = c, linewidth=1.0, alpha = a - 0.3)
				ax[xcount,ycount].set_ylabel(getTitle(f))

			else:
				
				if ki_mask[roi]:
				
						ax2.plot(handle[f][2],handle[f][0],linewidth=1.5,marker="o", \
						linestyle='--', color=c, alpha=a)
						
						ax2.fill_between(handle[f][2],handle[f][3],handle[f][4],\
								color = c, linewidth=1.0, alpha = a- 0.3)

						ax2.set_ylabel(getTitle(f,ND_Ki=True))

			if f != "Ki":
				if len(handle[f][5]) != 0:
					ax[xcount,ycount].plot(handle[f][5],handle[f][6],color=c, \
					   linewidth = 0.0,marker="x", alpha = a)
				
		count += 1
		xcount += 1
		
		if xcount == 5:
			xcount = 0; ycount += 1
	
	plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.90, \
					  hspace=0.30, wspace=0.60)

	if save: plt.savefig(dst+apnd+"."+fmt,format=fmt)
	
#------------------------------------------------------------------------------

## Creates a dual plot after splitting the lung per compartment. 
# @param data: A dictionary holding the data.
# @param figsize: The size of the image in inches. Default is (10,5).
# @param bbox_to_anchor1: Used to configure the placement of the nomenclature. 
# It's not intended to be changed but it may be necessary if the figsize
# changes. Default is (0.985,0.895).
# @param bbox_to_anchor2: Used to configure the placement of the nomenclature. 
# It's not intended to be changed but it may be necessary if the figsize
# changes. Default is (0.985,0.895).
# @param ND_Ki: Boolean flag which activates the non-dimentialization of the 
# Ki value. Default is True.

def perCompartmentDualPlot(data, figsize=(14,8), bbox_to_anchor1=(0.95,1.05),
						    bbox_to_anchor2=(0.95,1.10), ND_Ki = True):

	states = list(data.keys())
	n_states = len(states)
	fields = list(data[states[0]].keys())
	titles = ["Hyperinflated tissue", "Normally aerated tissue",\
			   "Poorly aerated tissue", "Non aerated tissue"]
	
	bbta = [bbox_to_anchor1,bbox_to_anchor2]
	
	compartmentHolder = {}		
	for i in range(4):	
		fieldHolder = {}
		for f in fields:	
			means = []; stds = []
			for s in states:
				if ND_Ki and f == 'Ki':
					means.append(data[s][f][0][i]/data[states[0]][f][0][i])		
					stds.append(data[s][f][1][i]/data[states[0]][f][0][i])					
				else:
					means.append(data[s][f][0][i]);  stds.append(data[s][f][1][i])
			fieldHolder.update({f:(np.array(means),np.array(stds))})
		compartmentHolder.update({i:fieldHolder})
		
	fig, axes = plt.subplots(nrows=2,ncols=2,figsize=figsize)	
	x = np.linspace(1,n_states,n_states)
	
	colors = ('black','grey')
	if not ND_Ki:
		bounds = [(-50,200), (0,0.12)]
	else:
		bounds = [(-50,200), (0,8)]	
	
	for i,ax,t in zip(range(4),axes.flatten(),titles):
		
		ax.set_title(t)
		ax2=ax.twinx()
		ax.set_xlim(0.5,float(n_states)+0.5)
		ax.set_xticks(x)
		ax.set_xticklabels(states)
		
		for a,f,c,b,bb in zip([ax,ax2],fields,colors,bounds,bbta):
			top = compartmentHolder[i][f][0] + compartmentHolder[i][f][1]
			bottom = compartmentHolder[i][f][0] - compartmentHolder[i][f][1]
			a.plot(x,compartmentHolder[i][f][0], color=c)
			a.fill_between(x,bottom,top, alpha=0.35, color=c)
			a.set_ylim(b)
			a.set_ylabel(getTitle(f,ND_Ki=ND_Ki))
			
			if (i+1)%2== 0:
				a.legend([getTitle(f,ND_Ki=ND_Ki)], loc='best',\
					bbox_to_anchor=bb, \
					bbox_transform=plt.gcf().transFigure)
	
	plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.90, \
					  hspace=0.30, wspace=0.60)	

	return 

#------------------------------------------------------------------------------

## A method which takes in an axis (from the matplotlib structures) and fills
# the most common attributes such as the y-axis label, x-axis label, legends
# and other fields. 
# @param ax: The working axis.
# @param ylabel: A string which is intended to be placed at the y-axis. Default
# is "".
# @param xlabel: A string which is intended to be placed at the x-axis. Default
# is "ROI".
# @param direction: A combination of 'V' and 'D' or 'B' and 'A' that permits
# the placing of a text next to the x axis. Default is "".
# @param legends: A list of strings which contains the different legends for
# everything being plotted in the axis. Default is None.
# @param ybounds: A tuple which contains in [0] the minimum value and in [1] 
# the maximum value of the field. Default is None.
# @param bbox_to_anchor: Used to configure the placement of the nomenclature. 
# It's not intended to be changed but it may be necessary if the figsize
# changes. Default is (0.985,0.895).
# @param xLen: The number of ROIs. Default is 10. 
	
def basicROIGraphStructure(ax, ylabel="", xlabel="ROI", direction="", \
						   legends=None, ybounds = None, \
						   bbox_to_anchor=(1.05,0.895), xLen = 10):
	
	ax.set_xticklabels([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
	ax.set_xticks(np.arange(11)+1)
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)
	ax.set_xlim(0.5,10.5)
	if ybounds==None:
		ybounds=ax.get_ylim()
	else:
		ax.set_ylim(ybounds)
		
	if ybounds == None:
		ybounds = ax.get_ylim()
		
	y0=ybounds[0]; y1=ybounds[1]; dy=(y1-y0)*0.01

	if direction != "":
		if direction in ["VD", "DV", "BA", "AB"]:
			captions = getCaptions(direction)
			
			ax.text(-0.3,y0-9.0*dy,captions[0], horizontalalignment='center')
			ax.text(11.3,y0-9.0*dy,captions[1], horizontalalignment='center')
			ax.set_title("%s to %s direction"%(captions[0],captions[1]))
			
	if legends != None:
		ax.legend(legends, loc='best', bbox_to_anchor=bbox_to_anchor, \
			 bbox_transform=plt.gcf().transFigure)
		

#------------------------------------------------------------------------------

## Receives the codename of the field according to the numpy mesh file and 
# returns an extended ylabel, an extended string related to the field.
# @param field: The codename of the field.
# @param ND_Ki: Boolean flag that refers to non-dimensionalization of the Ki,
# if activated, will assume the field is Ki amplification, else is the raw Ki.
# Default is False.

def getTitle(field, ND_Ki = False):

	if field == 'VS':
		title = 'Volumetric Strain [\%]'
	elif field == 'Ki':
		if not ND_Ki:
			title = "18F-FDG net uptake rate"
		else:
			title = "Ki Amplification Factor (-)"
	else:
		title = "Add new exceptions to getTitle routine"
	return title

#------------------------------------------------------------------------------

## Creates a bidirectional matrix plot. 
# @param data: Holds the data and the validity matrix to be plotted. 
# @param mask: Boolean matrix in order to mask some data. Default is None.
# @param vmin: The minimum value for the color bar. Default is 0.
# @param vmax: The maximum value for the color bar. Default is 150.
# @param colorTicks: The numbers of colors for the palette. Default is 10.
# @param cmap: The color palette to be used. Default is 'jet'.
# @param figsize: The size of the image. Default is (5,5).
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format. Default is "".

def bidirectionalMatrix(data,mask=None,vmin=0,vmax=150, colorTicks = 10, 
				cmap='jet',figsize=(5.5,5), save=False, dst = "", fmt = "pdf",
				bar_title = ""):

	cmap = plt.get_cmap(cmap, 14)
	
	if not mask is None:
		data = np.where(mask, data, np.nan)
	
	fig, ax = plt.subplots(figsize=figsize)
	ax.matshow(data, cmap = cmap, vmin=vmin, vmax=vmax)

	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_xticks([])
	ax.set_yticks([])
	trans = ax.get_xaxis_transform()
	
	ax.annotate("", xy=(-1.0,0.85), xytext=(-1.0, 0.15), xycoords=trans, 
			 arrowprops=dict(arrowstyle="<|-|>"))
	
	ax.annotate("", xy=(1.0,-0.05), xytext=(8.0, -0.05), xycoords=trans, 
			 arrowprops=dict(arrowstyle="<|-|>"))
	
	ax.annotate("A",xy= (0.0,-0.07), xycoords=trans, size=18, weight = 'bold')
	ax.annotate("B",xy= (8.5,-0.07), xycoords=trans, size=18, weight = 'bold')
	ax.annotate("D",xy= (-1.35,0.05), xycoords=trans, size=18, weight = 'bold')
	ax.annotate("V",xy= (-1.35,0.90), xycoords=trans, size=18, weight = 'bold')
	
	if bar_title != "":
		ax.set_title(bar_title)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.1)
	im = ax.imshow(data, cmap=cmap)
	im.set_clim(vmin,vmax)
	fig.colorbar(im, cax=cax, orientation='vertical')
		
	plt.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.80)	
	
	if save: plt.savefig(dst+"."+fmt,format=fmt)

#------------------------------------------------------------------------------

## Alternative method to create a bidirectional matrix plot. Oriented to global
# plots. 
# @param g_results: Holds the data and the validity matrix to be plotted. 
# @param plot_kind: The kind of plot to be created. This directs inner 
# configuration variables to follow preestablished values. The allowable kinds
# are "SPI", "dSPI", "KiA", "SHI", "VS", "IE" or "DGF". New values must be
# introduced by modifying this code.
# @param colorTicks: The numbers of colors for the palette. Default is 10.
# @param cmap: The color palette to be used. Default is 'jet'.
# @param fig_side: The size of the image. Default is 5.0.
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format. Default is "".

def bidirectionalMatrix2(g_results,plot_kind, colorTicks = 10, 
				 cmap='jet', fig_side=5.0, fmt="pdf", save = False, dst = ""):
	
	mask = g_results["g_validity"]
	
	if plot_kind == "SPI":
		cmap = plt.get_cmap(cmap, 14)
		data = []
		ncols = 2; nrows = 2; figsize = (ncols*fig_side,nrows*fig_side)
		for p in range(4):
			raw_data = g_results["means"][p+1]/g_results["means"][0]
			data.append(np.where(mask, raw_data, np.nan))
			titles = ["03h","06h","09h", "12h"]
			bar_title = "Strain Progression Index (SPI) \n"
		vmin = 0.0; vmax = 3.0
	
	if plot_kind == "dSPI":
		cmap = plt.get_cmap(cmap, 14)
		data = []
		ncols = 2; nrows = 2; figsize = (ncols*fig_side,nrows*fig_side)
		for p in range(4):
			raw_data = g_results["means"][p+1]/g_results["means"][p]
			data.append(np.where(mask, raw_data, np.nan))
			titles = ["03h","06h","09h", "12h"]
			bar_title = "Strain Progression Index (SPI) \n"
		vmin = 0.0; vmax = 2.0
	
	elif plot_kind == "KiA":
		cmap = plt.get_cmap(cmap, 14)
		data = []
		ncols = 3; nrows = 1; figsize = (ncols*fig_side,nrows*fig_side)
		for p in range(3):
			raw_data = g_results["means"][p]
			data.append(np.where(mask, raw_data, np.nan))
			titles = ["00h","06h", "12h"]
			bar_title = "Ki Amplification Factor (-) \n"
		vmin = 1.0; vmax = 4.5


	elif plot_kind == "SHI": # take information from I3_U not VS
		cmap = plt.get_cmap(cmap, 14)
		data = []
		ncols = 3; nrows = 2; figsize = (ncols*fig_side,nrows*fig_side)
		for p in range(5):
			raw_data = g_results["stds"][p]/g_results["means"][p]
			data.append(np.where(mask, raw_data, np.nan))
			titles = ["00h","03h","06h","09h", "12h"]
			bar_title = "Strain Heterogeneity Index (SHI) \n"
		vmin = 0.0; vmax = 0.6
		
	elif plot_kind == "VS":
		cmap = plt.get_cmap(cmap, 14)
		data = []
		ncols = 3; nrows = 2; figsize = (ncols*fig_side,nrows*fig_side)
		for p in range(5):
			raw_data = g_results["means"][p]
			data.append(np.where(mask, raw_data, np.nan))
			titles = ["00h","03h","06h","09h", "12h"]
			bar_title = "Volumetric Strain (\%) \n"
		vmin = -50; vmax = 150

	elif plot_kind == "IE":
		cmap = plt.get_cmap("gray", 14)
		data = []
		ncols = 3; nrows = 2; figsize = (ncols*fig_side,nrows*fig_side)
		for p in range(5):
			raw_data = g_results["means"][p]
			data.append(np.where(mask, raw_data, np.nan))
			titles = ["00h","03h","06h","09h", "12h"]
			bar_title = "Intensity at EE (HU) \n"
		vmin = -1000; vmax = 0
		
	elif plot_kind == "DGF":
		cmap = plt.get_cmap("jet", 14)
		data = []
		ncols = 3; nrows = 2; figsize = (ncols*fig_side,nrows*fig_side)
		for p in range(5):
			raw_data = g_results["means"][p]
			data.append(np.where(mask, raw_data, np.nan))
			titles = ["00h","03h","06h","09h", "12h"]
			bar_title = "Gas Fraction Variation (-) \n"
		vmin = 0.0; vmax = 0.5
		
	fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
	
	for (d, ax, c, t) in zip(data, axes.flatten(),range(nrows*ncols), titles):
		
		ax.matshow(d, cmap = cmap, vmin=vmin, vmax=vmax)
		ax.set_title(bar_title+t)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_xticks([])
		ax.set_yticks([])
		trans = ax.get_xaxis_transform()
		
		ax.annotate("", xy=(-1.0,0.85), xytext=(-1.0, 0.15), xycoords=trans, 
				 arrowprops=dict(arrowstyle="<|-|>"))
		
		ax.annotate("", xy=(1.0,-0.05), xytext=(8.0, -0.05), xycoords=trans, 
				 arrowprops=dict(arrowstyle="<|-|>"))
		
		ax.annotate("A", xy= (0.0, -0.07), xycoords=trans, size=18, weight = 'bold')
		ax.annotate("B", xy= (8.5, -0.07), xycoords=trans, size=18, weight = 'bold')
		ax.annotate("V", xy= (-1.35, 0.05), xycoords=trans, size=18, weight = 'bold')
		ax.annotate("D", xy= (-1.35, 0.9), xycoords=trans, size=18, weight = 'bold')
		
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.25)
#		cax.set_xlabel(bar_title)
#		if ((c+1)%ncols==0):
		im = ax.imshow(d, cmap=cmap)
		fig.colorbar(im, cax=cax, orientation='vertical')
		im.set_clim(vmin,vmax)
	
#	if plot_kind != "SPI" or plot != :
#		rgb_pig = Image.open("/home/user/Documents/BMA/FIGURES/pig.tif")
#		bw_pig = np.array(rgb_pig.convert("L"))
#		ax = axes.flatten()[5]
#		ax.matshow(bw_pig,cmap = "gray", vmin=0, vmax=255)
#		ax.set_xticklabels([])
#		ax.set_yticklabels([])
#		ax.set_xticks([])
#		ax.set_yticks([])
#		ax.axis(False)
		
	plt.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.90,
					 		  hspace=0.150, wspace=0.30)	
	
	if save: plt.savefig(dst+"."+fmt,format=fmt)

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------

## Given a data holder such as a list, a numpy array or matrix, finds the
# maximum value and fits it of a closed number (in relation to the yfactor)
# in order to create the upper bound for a graph.
# @param data: The data array.
# @param deviation: Deviations for the data array.
# @param yfactor: A factor intended to delete unnecessary decimals or other 
# numbers.

def findYmax(data, deviation, yfactor=1.):
	return np.ceil(np.max(np.mat(data)+np.mat(deviation))/yfactor)*yfactor

#------------------------------------------------------------------------------

## Given a data holder such as a list, a numpy array or matrix, finds the
# minimum value and fits it of a closed number (in relation to the yfactor)
# in order to create the lower bound for a graph.
# @param data: The data array.
# @param deviation: Deviations for the data array.
# @param yfactor: A factor intended to delete unnecessary decimals or other 
# numbers.

def findYmin(data, deviation, yfactor=1.):
	return np.floor(np.min(np.mat(data)-np.mat(deviation))/yfactor)*yfactor

#------------------------------------------------------------------------------

## Given a direction such as "VD" or "BA", returns a tuple of strings with the
# complete name of the directions in order to place them into the plots. 
# @param direction: The name of the direction. Must be a combination of "VD" or
# "BA".

def getCaptions(direction):

	captions = []
	for l in range(2):
		if direction[l] == "V":
			captions.append("Ventral")
		elif direction[l] == "D":
			captions.append("Dorsal")
		elif direction[l] == "B":
			captions.append("Basal")
		elif direction[l] == "A":
			captions.append("Apical")
	return captions


#------------------------------------------------------------------------------

## A flexible generic plotting function in order to make plots.
# @param dictionary: Dictionary that holds the data do be plotted and some 
# configuration parameters.
# @param fmt: The format of the saved image such as 'pdf' or 'png'. Default is
# 'pdf'.
# @param save: Boolean flag which activates the saving of the image. Default is
# False.
# @param dst: Route to the saved image, including its name and excepting its
# format. Default is "".
# @param figsize: Default size of the image. Size in inches. Default is (8,6).

def cleanPlotter(dictionary, figsize = (8,6),  save = False, dst = "", 
				 fmt = "pdf"):
	
	# If 'secondary' is in the dictionary keys then the plot is dual
	dualPlot = True if "secondary" in dictionary else False
		
	# Create plot structures
	fig, ax = plt.subplots(nrows=1,ncols=1, figsize=figsize)
	
	# Plot the primary data in the given axis.
	composePlot(ax, dictionary["primary"], dualPlot = dualPlot)
	
	# If the secondary plot is enabled then plot it in the cloned axis.
	if "secondary" in dictionary:
		 composePlot(ax.twinx(), dictionary["secondary"], primary=False, 
			   dualPlot = dualPlot)
	
#-----------------------------------------------------------------------------

## Reads the contents of the plotting dictionary which holds both the data and
# the configuration information is contained.
# @param dictionary: Dictionary that holds the data do be plotted and some 
# configuration parameters.
def readDictionary(dictionary):
	
	# The 'curves' name explains itself. Each contains a set of data that will 
	# be plotted independently. Contains the identifier for each set.
	if "identifiers" not in dictionary:
		print("The input dictionary does not follow the required format.")
		print("A field in the most external dictionary named 'subjects' must" )
		print("be included to identify the relevant keys to be plotted.")
		return
	else:
		curveIdentifier = dictionary["identifiers"]

	# The legends contains the names for each curve that will appear in a box 
	# in a side of the graph. 
	if "legends" not in dictionary:
		legends = np.array(curveIdentifier)
	else:
		legends = np.array(dictionary["legends"])

	# The name of the field that it's being plotted. 
	if "field" not in dictionary:
		field = ""
	else:
		field = dictionary["field"]

	# Used to crop the bounds of the curves being plotted.
	if "forced_ybounds" in dictionary:
		yboundsFlag = True
		ybounds = dictionary['forced_ybounds'] 
	else:
		yboundsFlag = False
		ybounds = None

	if "force_bounds" in dictionary:
		boundsFlag = True
		forcedBounds = dictionary['force_bounds']
	else:
		boundsFlag = False
		forcedBounds = None
	
	if "force_ylabel" in dictionary:
		ylabel = dictionary["force_ylabel"]
	else:
		ylabel = None
	
	# The keys to be plotted. With 'means' and 'stds' typical variation curves
	# may be plotted but aditionally, coefficient of variation may be plotted
	# as well as other future fields.
	if "keys" in dictionary:
		plotName, fillName = dictionary["keys"]
	else:
		plotName = "means"; fillName = "stds"
		
	if "save" in dictionary:
		save = True
		dst = dictionary["save"][0]
		fmt = dictionary["save"][1]
	else:
		save = False; dst = ""; fmt = "pdf"
		
	if "horizontalLine" in dictionary:
		horizontalTuple = dictionary["horizontalLine"]
	else:
		horizontalTuple = None
		
	colors = dictionary["force_colors"] if "force_colors" in dictionary else []
	
	lines = dictionary["force_lines"] if "force_lines" in dictionary else []
		
	
	# Label for the x axis.
	xlabel = dictionary["xlabel"] if "xlabel" in dictionary else ""

	# The names that will be displayed at the x axis.
	xticklabels = None if "xticks" not in dictionary else dictionary["xticks"]

	# If the graph has a directionality, this field becomes relevant. Direction
	# related texts will be included if this has a none null value.
	direction = None if "direction" not in dictionary else dictionary["direction"]
	
	# Habilitates the fill curves which contain standard dev or standard error.
	fillBetween = True if "fillBetween" in dictionary else False
	
	plotAmplification = True if "plotAmplification" in dictionary else False
	
	if "saveExcel" in dictionary:
		writer = pd.ExcelWriter(dst+".xls")
		for i in curveIdentifier:
			dataframe = pd.DataFrame(dictionary[i])
			dataframe.to_excel(writer,sheet_name=i)
		writer.save()
		
			
	
	return curveIdentifier, legends, field, boundsFlag, forcedBounds, \
	plotName, fillName, xlabel, ylabel, xticklabels, yboundsFlag, ybounds, \
	direction, fillBetween, plotAmplification, save, dst, fmt, horizontalTuple, \
	colors, lines
	
#-----------------------------------------------------------------------------

## A axis will filled here according to the indications given at the dictionary
# that will be processed with the method 'readDictionary'.
# @param ax: The working axis. 
# @param dictionary: The holder of the information.
# @param primary: Boolean flag which refers to whether the data is related to 
# the primary axis or not (that is, it refers to the secondary axis). Default 
# is True.
# @param dualPlot: Boolean flag which refers to whether the plot is dual or 
# not. Default is False.
# @param linewidth: Defines the width of the plotted lines. Default is 1.5.

def composePlot(ax, dictionary, primary = True, dualPlot = False,
				linewidth = 1.5):
	
	# The code branches when is a dual plot.
	if dualPlot:

		if primary:
			bbox_to_anchor = (1.15, 0.895)
			marker = "."
			color = "darkblue"
			hatch = None

		else:
			bbox_to_anchor = (1.15, 0.600)
			marker = "o"
			hatch = "/"
			color = "darkred"
			
	else:
		bbox_to_anchor = (1.01, 0.895)
		marker = "."
		color = "k"
		hatch = None
		
	# Recover the options that were introduced in the dictionary.
	curveIdentifier, legends, field, boundsFlag, forcedBounds, plotName,  \
	fillName, xlabel, ylabel, xticklabels, yboundsFlag, ybounds, direction, \
	fillBetween, plotAmplification, save, dst, fmt, horizontalTuple, \
	colors, lines = readDictionary(dictionary)

	if len(colors) == 0:
		for r in range(len(curveIdentifier)):
			colors.append(color)
		


	# Create the space in the X axis where the data will be plotted and the
	# ticks/xticklabels will be placed.
	xLen = len(dictionary[curveIdentifier[0]]['means'])
	
	# Create the palette for the different curves 
	ncurves = len(curveIdentifier)
	
## DONE: REMEMBER TO UNCOMMENT THESE LINES AFTER ARAOS RESULTS
	base_alphas = np.linspace(0.20, 0.90, ncurves)
	fb_alphas = np.linspace (0.1, 0.3, ncurves)

# TODO: REMOVE ALPHAS BY COMMAND 
	
#	base_alphas = np.linspace(0.80, 0.80, ncurves)
#	fb_alphas = np.linspace (0.00, 0.000, ncurves)

	# Coordinates in the x axis.
	x = np.linspace(1,xLen,xLen)
	l_mask = []
	
	for (i,a1,a2,color) in zip(curveIdentifier,base_alphas,fb_alphas,colors):
		
		avg = np.array(dictionary[i][plotName]) # The main line.
		mask = np.array(dictionary[i]['validity']) # Used to block points.
		block = not np.any(mask) # Used to block complete curves.
		x_l = x # The space to plot the data.
		l_mask.append(not block) # Used to remove curves from legends.
		
		if block: # If block, well... block.
			continue
		
		if boundsFlag: # To arbitrarily remove the extremes of the curves.
			l_bound, r_bound = forcedBounds[i]
		else:
			l_bound, r_bound = 0, len(avg)
		
		# The data reduction happens at the method reduceData.
		avg = reduceData(avg, mask, l_bound, r_bound)
		x_l = reduceData(x_l, mask, l_bound, r_bound)
		
		# Plot the main curve.
		ax.plot(x_l, avg, color=color, linewidth=linewidth, marker=marker,
		  alpha=a1)
		
		# Add some bars that represent standard deviation or similar stuff.
		if fillBetween:
			std = np.array(dictionary[i][fillName]) 
			std = reduceData(std, mask, l_bound, r_bound)
			top_y = avg + std;	bot_y = avg - std # top and bottom curves. 
			ax.fill_between(x_l,top_y,bot_y, color=color, linewidth=1.0,
				   alpha=a2, hatch = hatch) # Actual plotting.

	# Add labels, ticks, place the legends, directions, etc. 
	basicStructure(ax, field, xticklabels, xLen, xlabel, bbox_to_anchor, 
			direction=direction, legends=legends[l_mask], 
			plotAmplification = plotAmplification, ylabel = ylabel,
			forcedBounds = ybounds, boundsFlag = yboundsFlag)
	
	if horizontalTuple != None:
		ymin, ymax = ax.get_ylim(); yrange = ymax - ymin
		
		ax.text(1.08* xLen, horizontalTuple[1] - yrange * 0.008,
		   horizontalTuple[0], fontsize=12, 
		   horizontalalignment='left')
		
		ax.axhline(y = horizontalTuple[1], xmin = 0, xmax=xLen, color = "grey",
			  linewidth = 1.0, linestyle = '--')
	
	
	
	plt.subplots_adjust(top=0.88, bottom=0.10, left=0.13, right=0.80)
	
	if save: plt.savefig(dst+"."+fmt,format=fmt)


#------------------------------------------------------------------------------

## Filter the data arrays.
# @param data: The original data array.
# @param mask: A mask to filter the original array.
# @param l_bound: A forced counter to remove the points at the left.
# @param r_bound: A forced counter to remove the points at the right. 

def reduceData(data, mask, l_bound, r_bound):
	data = data[l_bound:r_bound]
	return data[mask[l_bound:r_bound]]

#------------------------------------------------------------------------------

## Create full names and ranges for the plot for a given field. 
# @param field: The name of the field of interest.
# @param amplification: Boolean flag to check whether is the Ki amplification
# the field of interest or not. Only relevant if field = 'Ki'. Default is True.
	
def getRelevantData(field, amplification = True):
	
	# Third invariant. Jacobian.
	if field == "I3_U":
		title = "$RDI_{vol}$"
		ylim = (0.5, 3.0)
	
	elif field == "I2_U":
		title = "$RDI_{sup}$"
		ylim = (0.75, 2.25)
		
	elif field == "I1_U":
		title = "$RDI_{lin}$"
		ylim = (0.9, 1.5)
	
	# Volumetric Strain
	elif field == 'VS':
		title = 'Volumetric Strain (\%)'
		ylim = (-50,200)
	
	# FDG-18 ... Amplification currently.
	elif field == 'Ki':
		if amplification:
			title = "Ki Amplification Factor"
			ylim = (0, 10)
		else:
			title = "18F-FDG Net Uptake Rate"
			ylim = (0.0, 0.15)
	
	# Intensity at end of expiration (EE).
	elif field == 'Intensity_Exp':
		title = "Intensity at EE (HU)"
		ylim = (-1000,100)

	elif field == 'Intensity_Insp':
		title = "Intensity at EI (HU)"
		ylim = (-1000,100)
		

	# Delta Gas Fraction
	
	elif field == 'Delta_Gas_Fraction':
		title = "Gas Fraction Variation [-]"
		ylim = (-0.10, 0.60)
		
	else:
		title = None; ylim = (None, None)
	
	return title, ylim
#------------------------------------------------------------------------------

## Place the names, ticks and similar stuff within the relevant axis.
# @param ax: The axis that is being worked on.
# @param field: The name of the field being worked with.
# @param xticklabels: A list that contains the name that is being placed over
# the ticks of the x-axis.
# @param xLen: A length of the x-axis.
# @param xlabel: The name to place under the x-axis.
# @param bbox_to_anchor: A tuple that is related to the position of the legend
# box within the image.
# @param direction: The direction of the graph, if available. Default is None.
# @param legends: A list of names for the plotted data. Default is None.
# @param plotAmplification: Boolean flag which, if True, informs that the data
# being worked is the Ki amplification instead of the raw Ki. Only relevant if
# field == 'Ki'. Default is True.
# @param ylabel: The label to be placed at the y-axis. Default is None.
# @param boundsFlag: Boolean flag which refers to whether crop the raw data
# according to forcedBounds. This activate the cropping. Scarcely used. Default
# is None.
# @param forcedBounds: Tuple containing the (integer) for the left bounds and
# right bound where the data is cropped. This is relevant only when boundsFlag
# is True.

def basicStructure(ax, field, xticklabels, xLen, xlabel, bbox_to_anchor, 
				  direction=None, legends=None, plotAmplification = True,
				  ylabel = None, forcedBounds = None, boundsFlag = False,
				  interveneLegends = False):
	
	# Labels, placed wherever there is a tick.
	ax.set_xticklabels(xticklabels)
	# Placing the tick locations.
	ax.set_xticks(np.arange(xLen + 1)+1)
	
	# Get name for the y-axis and its bounds.
	if ylabel == None:
		ylabel, ybounds = getRelevantData(field, amplification = plotAmplification)
	else:
		_, ybounds = getRelevantData(field, amplification = plotAmplification)

	if boundsFlag: ybounds = forcedBounds
	
	if interveneLegends:
		if not legends is None:
			if len(legends)>0:
				leg = []
				for l in legends:
					if '_' in l:
						leg.append(l.replace('_','-'))
					else:
						leg.append(l)
				legends = leg
			
			
	print(field, ylabel, ybounds)
	# Labels for the x-axis and y-axis.
	ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
	
	# Force the limits for the x-axis and y-axis.
	ax.set_xlim(0.5,float(xLen) + 0.5); ax.set_ylim(ybounds)
	y0=ybounds[0]; y1=ybounds[1]; dy=(y1-y0)*0.01
	
	# Place the names of the directions within the graph.
	if direction != None:
		if direction in ["VD", "DV", "BA", "AB"]:
			captions = getCaptions(direction)
			ax.text(-0.3,y0-9.0*dy,captions[0], horizontalalignment='center')
			ax.text(11.3,y0-9.0*dy,captions[1], horizontalalignment='center')
#			ax.set_title("%s to %s direction"%(captions[0],captions[1]))
	
	# Box with legends at the side of the 
	if not legends is None:
		if len(legends)>0 :
			ax.legend(legends, loc='best', bbox_to_anchor=bbox_to_anchor, 
				 bbox_transform=plt.gcf().transFigure)

#------------------------------------------------------------------------------