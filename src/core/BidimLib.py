import numpy as np
import os
import ROIAnalysis as ROI

from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import linregress
from new_plotter import getRelevantData
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.family'] = 'Helvetica'

# =============================================================================
#  CLASS frame
# =============================================================================

class frame():
    # Relevant identification for a state of any given subject
    def __init__(self, 
              keyword,
              number,
              state,
              petct_available = False,
              subject_rootname = "%i",
              image_dir = "NIFTY",
              mesh_dir = "MESH",
              reg_dir = "REGISTRATION",
              path_to_dir = ""):
        
        self.keyword = keyword
        self.number = number
        self.subject = subject_rootname%(number)
        self.state = state
        self.petct_available = petct_available
        self.codename = "%s/"%(self.subject)
        self.address = "%s/"%(path_to_dir)
        self.image_dir = image_dir
        self.mesh_dir = mesh_dir
        self.reg_dir = reg_dir
        self.path_to_dir = path_to_dir
        
    def identify(self):
        self.update()
        print(self.codename)
        
    def update(self):
        self.codename = "%s/%s"%(self.subject, state)
        self.address = "%s/%s/"%(self.path_to_dir,self.codename)
# =============================================================================
# CLASS point
# =============================================================================

class point():
    
    def __init__(self, pid, coords, omit_coords = False):
        self.pid = pid
        self.data = {}
        self.color = "k"
        self.marker = "o" 
        self.fillstyle = "full"
        self.markersize = 5
        self.markerfacecoloralt="k"
        self.sorterkey = None
        self.coordinated = not omit_coords
        
        if not omit_coords:
            if len(coords) == 3:
                self.coordinate_vd = coords[0]
                self.coordinate_ba = coords[1]
                self.coordinate_rl = coords[2]
            else:
                self.coordinate_vd = coords[0]
                self.coordinate_ba = coords[1]
                self.coordinate_rl = np.nan
        else:
            self.coordinate_ba = np.nan
            self.coordinate_vd = np.nan
            self.coordinate_rl = np.nan
            
    def resetMarker(self):
        self.color = "k"
        self.marker = "o" 
        self.fillstyle = "full"
        self.markersize = 5
        self.markerfacecoloralt="k"
        self.sorterkey = None
        
    def addData(self, field_name, state_name, data, valid = True):
        if field_name not in self.data.keys():
            self.data.update({field_name:{}})
        self.data[field_name].update({state_name:(data, valid)})
        
    def modifyColor(self, color):
        self.color = color
        
    def modifyMarker(self, marker):
        self.marker = marker


# =============================================================================
# CLASS matrix
# =============================================================================


class matrix():

    ## Initiates a Matrix structure. In order to load data, a method name 
    # 'load' is to be invoked specifying the data origins and related 
    # information. 
    # @param nrois_ba: Specifies the number of divisions to be used in the
    # basal to apical direction.
    # @param nrois_vd: Specifies the number of divisions to be used in the
    # ventral to dorsal direction.
    # @param nrois_rl: Specifies the number of divisions to be used in the
    # right to left direction.
    
    def __init__(self, subject,  nrois_ba = 10, nrois_vd = 10, nrois_rl = 3,
              reverse_i = False, reverse_j = True, reverse_k = True, 
              ki_bins = 6):
        
    # The reverse_* were chosen as to mimic the extendedROIMatrixs routine
    # from the database suite. 
    # when reverse_* is applied, as the name says, the numbering of rois is 
    # reversed. In the base setting:
    # i moves from dorsal to ventral
    # j moves from apical to basal
    # k moves from left to right
    # I retain centain mistrust regarding how these should be used. When using
    # the bidimensional matrix they work alright. Compatibility is to be tested
    
    ## @var grid: Structure that holds the points that create the matrix.
        self.grid = {} 
    ## @var nrois_ba: Number of divisions within the basal-apical direction.
        self.nrois_ba = nrois_ba
    ## @var nrois_vd: Number of divisions within the ventral-dorsal direction.
        self.nrois_vd = nrois_vd
    ## @var nrois_rl: Number of divisions within the right-left direction.
        self.nrois_rl = nrois_rl
    ## @var grid_length: Total number of divisions within the mesh.
        self.grid_length = self.nrois_ba * self.nrois_vd * self.nrois_rl
    ## @var point_keys: A list that can be used to move through the matrix.
        self.point_keys = list(range(self.grid_length))
    ## @var directory: Specifies the fields and states contained within.
        self.directory = {}
    ## @var bin: TODO
        self.bin = None
    ## @var subject: Specifies the subject for this matrix
        self.subject = subject
    ## @var Ki_bounds: Specifies the bounds for the Ki fields. 
        self.Ki_bounds = {}
    ## @var VS_bounds: Specifies the bounds for Volumetric Strain.
        self.VS_bounds = {}
    ## @var Specifies whether a particular sorting was used for the matrix.
        self.sortname="None"
    ## @var A color guide for to be used when scatter plotting.
        self.sort_colors = {"None":"k"}
        
        self.ki_bins = ki_bins
        
        self.global_validity = None
        
        l = 0
        for k in range(self.nrois_rl):#
            for i in range(self.nrois_vd):
                for j in range(self.nrois_ba):
                    self.grid.update({l:point(l, (i,j,k))})
                    l += 1

# Load data into a matrix.

    def load_slice(self, frame, field, data, subroi_size_validity):
        
        if self.subject != frame.number:
            print(self.subject, frame.number)
            print("load_slices failed: the subjects don't match")
            return
        
        self.sortlist = {self.sortname:self.point_keys}

        if field not in self.directory:
            self.directory.update({field:[frame.state]})
        
        if frame.state not in self.directory[field]:
            self.directory[field].append(frame.state)

        data = np.array(data).flatten()
        end_validity = np.logical_and(np.logical_not(np.isnan(data)).flatten(),
                                 np.array(subroi_size_validity).flatten())
        
        for i in self.point_keys:
            self.grid[i].addData(field, frame.state, data[i], end_validity[i])

# Sorting is intended to classify the points within a single matrix using some
# criteria.

    def resetSorting(self):
        for i in self.point_keys:
            self.grid[i].resetMarker()
        self.bin=None
        self.sortname="None"
        self.sortlist = {self.sortname:self.point_keys}
        self.sort_colors = {"None":"k"}

# Scatter plot the data according to a some criteria. 

    def scatterBase(self, field_x, state_x, field_y, state_y, 
                 includeIdentity = False, figsize = (6,6), ybounds = None,
                 xbounds =None, force_bounds = False, line45degree = False,
                 save=False, dst_dir = "", fmt = "pdf", 
                 linregress_sort_flag = False, linregress_sets = ["None"]):
        
        # Quick check for the availability of the info
        if field_x in ["Intensity_Insp", "Intensity_Exp"] and force_bounds:
            xbounds = (-1000,100)
            
        if field_y in ["Intensity_Insp", "Intensity_Exp"] and force_bounds:
            ybounds = (-1000,100)
        
        if field_x == 'VS' and force_bounds:
            xbounds = (-50,200)
            
        if field_y == 'VS' and force_bounds:
            ybounds = (-50,200)
        
        if field_x == 'Delta_Gas_Fraction' and force_bounds:
            xbounds = (0.0,0.50)
            
        if field_y == 'Delta_Gas_Fraction' and force_bounds:
            ybounds = (0.0,0.50)
        
        for field, state in zip([field_x, field_y], [state_x, state_y]):
            if not field in self.directory: 
                print("Missing field %s"%field)
            else:
                if not state in self.directory[field]:
                    print("Missing state '%s' in field '%s'"%(state,field))
                    return
                
        # Create figure
        fig, ax = plt.subplots(figsize = figsize)
        
        if not xbounds is None:
            ax.set_xlim(xbounds)
        if not ybounds is None:
            ax.set_ylim(ybounds)
        
        # load the data into arrays
        for i in self.point_keys:
                # check whether both points are valid
            if self.grid[i].data[field_x][state_x][1] and self.grid[i].data[field_y][state_y][1]:
                point = self.grid[i]
                ax.plot(point.data[field_x][state_x][0], 
                                point.data[field_y][state_y][0],
                                marker = point.marker,
                                color = point.color,
                                fillstyle = point.fillstyle,
                                markersize = point.markersize, 
                                alpha=0.4)
        
        tx, _ = getRelevantData(field_x)
        ty, _ = getRelevantData(field_y) 
        
        ax.set_xlabel("%s [%s]"%(tx,state_x))
        ax.set_ylabel("%s [%s]"%(ty,state_y))
        
        if (field_x == field_y) or line45degree:
            ylim = ax.get_ylim()
            ax.plot(ylim, ylim, color = "grey", linestyle = '--', alpha=0.5)
        
        if linregress_sort_flag:
            lr = self.createLine(field_x, state_x,
                                 field_y, state_y, linregress_sets)
            
            for s in linregress_sets:
                xcoords = np.linspace(lr[s]["xmin"],lr[s]["xmax"],2)
                ax.plot(xcoords, xcoords*lr[s]["slope"] + lr[s]["intercept"],
             color = self.sort_colors[s], alpha=0.5, linestyle="--")
                print("Set %s: \nR2: %5.3f\np-value: %5.3f"%(s,lr[s]["r2"],lr[s]["pvalue"]))
            
        

        if save: plt.savefig("%s%s%s_%s%s_sortedBy_%s.%s"%(dst_dir, field_x,
                    state_x, field_y, state_y, self.sortname,fmt),format=fmt)
    

# Extract the data, while considering the sorting that it has. If sorter == 
# None, then all the data is returned without modification.

    def returnSortedData(self, field, state, sorter="None", field_ = 'dgf'):
        
        if not sorter in self.sortlist: return
        
        holder = []; validator = []
        
        if field == 'label':
            labels = list(self.sortlist.keys())
            labels.remove('None')
            
        for i in self.sortlist[sorter]:
            if field != 'label':
                holder.append(self.grid[i].data[field][state][0])
                validator.append(self.grid[i].data[field][state][1])
            else:
                for l in labels:
                    if i in self.sortlist[l]:
                        if l != 'NOT':
                            holder.append(l)
                        else:
                            holder.append(-1)
                        break
                validator.append(self.grid[i].data[field_][state][1])
        return np.array(holder), np.array(validator)
    
    def createLine(self, field_x, state_x, field_y, state_y, sets):
        
        # sets are the groups that are to be plotted.  ["0","3","4"]
        data_x = {}
        data_y = {}
        statistics = {}
        for s in sets:
            data_x.update({s:[]})
            data_y.update({s:[]})
            statistics.update({s:{}})
            
            for pair in self.sortlist[s]:
                if self.grid[pair].data[field_x][state_x][1] and self.grid[pair].data[field_y][state_y][1]:
                    data_x[s].append(self.grid[pair].data[field_x][state_x][0])
                    data_y[s].append(self.grid[pair].data[field_y][state_y][0])
                
            lr = linregress(data_x[s],data_y[s])
            statistics[s].update({     "xmin":np.min(data_x[s]),
                                        "xmax":np.max(data_x[s]), 
                                        "ymin":np.min(data_y[s]), 
                                        "ymax":np.max(data_y[s]),
                                        "slope":lr.slope,
                                        "intercept":lr.intercept,
                                        "r2":lr.rvalue**2,
                                        "pvalue":lr.pvalue})
        return statistics
    
# Sort the points in the matrix using one of the criteria within the routine.
# Criterias: 'Gattinoni', 'Strain', 'GasFraction', 'Ki', 'PCA'.
    
    def sort(self, field, state, criteria, markersize = 5, 
                fillstyle="full", pca_pointlist = None):
        if criteria == 'Gattinoni':
            bins = [-1000,-900,-500,-100,100]
            # shape "s":square "o":circle "^":upTriangle "v":downTriangle
            # format is marker,color,fillstyle
            marker_scheme = [("s","k", "full",markersize), 
                    ("o","k","full", markersize), 
                    ("^","gray","full", markersize), 
                    ("v", "lightgray","full", markersize),
                    ("o","k","none", markersize), 
                    ("s","k" ,"none", markersize)]

        elif criteria == 'Strain':
            bins = self.VS_bounds[state]
            marker_scheme = [("o","blue", "none",markersize), 
                    ("o","blue","full", markersize), 
                    ("^","green","full", markersize), 
                    ("v", "orange","full", markersize),
                    ("o","red","full", markersize), 
                    ("o","red" ,"none", markersize)]

        elif criteria == 'GasFraction':
            bins = [0.0,0.15,0.30,0.45]
            marker_scheme = [("o","blue", "none",markersize), 
                    ("o","blue","full", markersize), 
                    ("^","green","full", markersize), 
                    ("v", "orange","full", markersize),
                    ("o","red","full", markersize), 
                    ("o","red" ,"none", markersize)]

        elif criteria == 'Ki':
            bins = self.Ki_bounds[state]
            marker_scheme = [("o","blue", "none",markersize), 
                    ("o","blue","full", markersize), 
                    ("^","green","full", markersize), 
                    ("v", "orange","full", markersize),
                    ("o","red","full", markersize), 
                    ("o","red" ,"none", markersize)]
            
        elif criteria == "PCA":
            marker_scheme = [("o","blue", "none",markersize), 
                    ("o","red","full", markersize), 
                    ("o","k","full", markersize), 
                    ("o", "g","full", markersize),
                    ("o","y","full", markersize), 
                    ("o","c" ,"full", markersize)]
            self.sortname = criteria
            for name in list(pca_pointlist.keys()):
                self.sortlist.update({name:pca_pointlist[name]}) 
            return
        else:
            return
        
        if not field in self.directory: 
            print("Missing field %s"%field)
        else:
            if not state in self.directory[field]:
                print("Missing state '%s' in field '%s'"%(state,field))
                return
        
        self.sortlist = {}
        for s,m in zip(range(len(bins)+1),marker_scheme):
            self.sortlist.update({s:[]})
            self.sort_colors.update({s:m[1]})

        for i in self.point_keys:
                s = np.digitize(self.grid[i].data[field][state][0], bins)
                self.grid[i].marker = marker_scheme[s][0]
                self.grid[i].color = marker_scheme[s][1]
                self.grid[i].fillstyle = marker_scheme[s][2]
                self.grid[i].markersize = marker_scheme[s][3]
                self.grid[i].sorterkey = s
                self.sortlist[s].append(i)

        self.bin = bins
        self.sortname="%s%s"%(field,state)

        print("Sorting criteria: %s"%criteria)
        for b,r in zip(bins,range(len(bins))):
            print("Bin limit #%i: %4.1f"%(r+1,b))

# Load a mask for the matrix that is valid at every state in analysis

    def load_global_mask(self, v):
        self.global_validity = v

# Generate a 2D matrix plot. This plot can be saved somewhere by toggling the
# flag save = True at the destination 'dst'.

    def plot(self, field, state, side = 6, cmap = "jet", vmin = None, 
          vmax = None, ncolors = 10, use_global_validity = False, save = False,
          dst = "", fmt_ = "pdf",
          b_top=0.99, b_bottom=0.01,
          b_left=0.10, b_right=0.85,
          b_hspace=0.10, b_wspace=0.30, 
          asterisks_coords =[], dpi=300,
          norm = None, shape_ratio=1.0, flip_vd=False):
        
        
        if field == "Intensity_Exp" or field == "Intensity_Insp":
            cmap = "gray"
            vmin, vmax = (-1000, 100)
            ncolors = 11
        elif field == "VS":
            if vmin == None and vmax == None:
                vmin, vmax = (-50, 100)
            ncolors = 12
        elif field == "Ki" or field == "qKi" :
            cmap = "plasma"
            ncolors = self.ki_bins -1
        elif field == 'SHI':
            if vmin == None and vmax == None:
                ncolors = 10
                vmin =  0.0
                vmax = 1.0
            
        def func(x, pos):
            return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")
        
        d, v = self.returnSortedData(field, state)
        d = d.reshape(self.nrois_rl, self.nrois_vd, self.nrois_ba).astype(float)
            
        
        if use_global_validity and self.global_validity is not None:
            v = self.global_validity.reshape(self.nrois_rl, self.nrois_vd, self.nrois_ba)
        else:
            v = v.reshape(self.nrois_rl, self.nrois_vd, self.nrois_ba)
        d[np.logical_not(v)] = np.nan

        if vmax is None:
            vmax = np.max(d[np.logical_not(np.isnan(d))]) 
        
        if vmin is None:
            vmin = np.min(d[np.logical_not(np.isnan(d))]) 
        
        figsize = (side, side * self.nrois_rl * shape_ratio)
        
        if flip_vd:
            d = np.flip(d,axis=1)
            v = np.flip(v,axis=1)
            len_vd = d.shape[1]
            
        fig, ax = plt.subplots(ncols = 1, nrows = self.nrois_rl, 
                         figsize = figsize, dpi=dpi)

        cmap = plt.get_cmap(cmap, ncolors)
        
        for k in range(self.nrois_rl):
            
            if self.nrois_rl > 1:
                axis = ax[k]
            else:
                axis = ax

            axis.matshow(d[k,:,:], cmap = cmap, vmin=vmin, vmax=vmax)
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            axis.set_xticks([])
            axis.set_yticks([])
            
            trans = axis.get_xaxis_transform()
            axis.annotate("", xy=(-0.75,0.85), xytext=(-0.75, 0.15), xycoords=trans, 
                 arrowprops=dict(arrowstyle="<|-|>")) # VD - left side

            
            
            if False: # 1x3x2
                axis.annotate("B", xy= (-0.5, -0.16), xycoords=trans, size=18, weight = 'bold')
                axis.annotate("A", xy= (2.5, -0.15), xycoords=trans, size=18, weight = 'bold') #5x5
                axis.annotate("", xy=(0.0,-0.05), xytext=(2.0, -0.05), xycoords=trans, 
                     arrowprops=dict(arrowstyle="<|-|>")) # BA - below 
            if True: # 10x10
                axis.annotate("B", xy= (-0.5, -0.16), xycoords=trans, size=18, weight = 'bold')
                axis.annotate("A", xy= (8.5, -0.15), xycoords=trans, size=18, weight = 'bold') #5x5
                axis.annotate("", xy=(0.0,-0.05), xytext=(8.0, -0.05), xycoords=trans, 
                     arrowprops=dict(arrowstyle="<|-|>")) # BA - below 
# 1x10x10           
            if not flip_vd:
                axis.annotate("D", xy= (-1.35, 0.05), xycoords=trans, size=18, weight = 'bold')
                axis.annotate("V", xy= (-1.35, 0.9), xycoords=trans, size=18, weight = 'bold')
            else:
                axis.annotate("V", xy= (-1.35, 0.05), xycoords=trans, size=18, weight = 'bold')
                axis.annotate("D", xy= (-1.35, 0.9), xycoords=trans, size=18, weight = 'bold')  

# 1x3x2          
#            if not flip_vd:
#                axis.annotate("D", xy= (-0.85, 0.05), xycoords=trans, size=18, weight = 'bold')
#                axis.annotate("V", xy= (-0.85, 0.9), xycoords=trans, size=18, weight = 'bold')
#            else:
#                axis.annotate("V", xy= (-0.85, 0.05), xycoords=trans, size=18, weight = 'bold')
#                axis.annotate("D", xy= (-0.85, 0.9), xycoords=trans, size=18, weight = 'bold')  
            

            if len(asterisks_coords)>0:
                for (kk,y,x) in asterisks_coords:
                    if flip_vd: y = len_vd-1-y
                    
                    if kk==k:
                        axis.annotate("*", xy=(x-0.25,y+0.25),size=25,weight='bold')
            
            
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size='5%', pad=0.25)
            
            if norm is None:
                im = axis.imshow(d[k,:,:], cmap=cmap)
                fig.colorbar(im, cax=cax, orientation='vertical')
                im.set_clim(vmin,vmax)
            else:
                im = axis.imshow(d[k,:,:], cmap=cmap, norm=norm)
                fig.colorbar(im, cax=cax, orientation='vertical')
                #im.set_clim(vmin,vmax)

            
        plt.subplots_adjust(top=b_top, bottom=b_bottom, left=b_left, 
                      right=b_right, hspace=b_hspace, wspace=b_wspace)

        if save: plt.savefig(dst+"."+fmt_,format=fmt_)

# Generate a plot labed, which is used to map the k-means clusters. Created
# to be used with at most three different clusters. 

    def plot_label(self, state, side = 6, vmin = None, 
          vmax = None, use_global_validity = True, n_clusters = 3, 
          colorlist = ["r","k","b"], 
          taglist = {"b":"Blue","k":"Black","r":"Red","white":"Invalid"},
          save = False, dst = "", fmt_ = "pdf",
          b_top=0.99, b_bottom=0.01,
          b_left=0.10, b_right=0.85,
          b_hspace=0.10, b_wspace=0.30):
        
        if colorlist[0] != "white":
            colorlist.insert(0,"white")

        d, v = self.returnSortedData('label', state)
        d = d.reshape(self.nrois_rl, self.nrois_vd, self.nrois_ba).astype(float)
        
        if use_global_validity and self.global_validity is not None:
            v = self.global_validity.reshape(self.nrois_rl, self.nrois_vd, self.nrois_ba)
        else:
            v = v.reshape(self.nrois_rl, self.nrois_vd, self.nrois_ba)
        d[np.logical_not(v)] = np.nan
        d[np.isnan(d)] = -1

        if vmax is None:
            vmax = np.max(d[np.logical_not(np.isnan(d))]) 
        
        if vmin is None:
            vmin = np.min(d[np.logical_not(np.isnan(d))])
        
        def func(x, pos):
            return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")
        
        vmin = -1
        vmax = n_clusters
        
        figsize = (side, side * self.nrois_rl)
        fig, axes = plt.subplots(ncols = 1, nrows = self.nrois_rl, 
                         figsize = figsize)        
        
        for a, ax in zip(range(self.nrois_rl), axes):
            cmap = ListedColormap(colorlist[0:(n_clusters+1)])
            cmap.set_over('0.25')
            cmap.set_under('0.75')
            tags = []
            for c in colorlist[0:(n_clusters+1)]:
                tags.append(taglist[c])
            qrates = np.array(tags)
            bounds = np.arange(-1, n_clusters+1, 1)
            
            norm = BoundaryNorm(bounds, cmap.N)
            fmt = FuncFormatter(lambda x, pos: qrates[norm(x)])
                    
            ax.matshow(d[a], cmap = cmap, norm = norm)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.25)
            im = ax.imshow(d[a], cmap=cmap, norm =norm)
            im.set_clim(vmin,vmax)
    
            cbar = fig.colorbar(im, cax=cax, orientation='vertical', \
              spacing='proportional', \
              ticks=np.linspace(-0.5,float(n_clusters)-0.5,n_clusters + 1), \
              format=fmt)
            
            cbar.minorticks_off()
            trans = ax.get_xaxis_transform()
        
            ax.annotate("B", xy= (0.0, -0.07), xycoords=trans, size=18, weight = 'bold')
            ax.annotate("A", xy= (8.5, -0.07), xycoords=trans, size=18, weight = 'bold')
            ax.annotate("D", xy= (-1.35, 0.05), xycoords=trans, size=18, weight = 'bold')
            ax.annotate("V", xy= (-1.35, 0.9), xycoords=trans, size=18, weight = 'bold')
        
        plt.subplots_adjust(top=b_top, bottom=b_bottom, left=b_left, 
                      right=b_right, hspace=b_hspace, wspace=b_wspace)
    
        if save: plt.savefig(dst+"."+fmt_,format=fmt_)

# Created a matrix plot for quantified Ki.

    def plot_qki(self, state, side = 6, use_global_validity = True, n_ki = 6, 
          save = False, dst = "", fmt_ = "pdf",
          b_top=0.99, b_bottom=0.01,
          b_left=0.10, b_right=0.85,
          b_hspace=0.10, b_wspace=0.30):
        

        d, v = self.returnSortedData('qKi', state)
        d = d.reshape(self.nrois_rl, self.nrois_vd, self.nrois_ba).astype(float)
        
        if use_global_validity and self.global_validity is not None:
            v = self.global_validity.reshape(self.nrois_rl, self.nrois_vd, self.nrois_ba)
        else:
            v = v.reshape(self.nrois_rl, self.nrois_vd, self.nrois_ba)
        d[np.logical_not(v)] = np.nan
        d[np.isnan(d)] = n_ki+1

        vmax = np.max(d[np.logical_not(np.isnan(d))]) +0.5
        vmin = np.min(d[np.logical_not(np.isnan(d))]) -0.5
        
        def func(x, pos):
            return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")
        #        
        figsize = (side, side * self.nrois_rl)
        fig, axes = plt.subplots(ncols = 1, nrows = self.nrois_rl, 
                         figsize = figsize)        
        
        for a, ax in zip(range(self.nrois_rl), axes):
#            cmap = ListedColormap(colorlist[0:(n_clusters+1)])
#            cmap.set_over('0.25')
#            cmap.set_under('0.75')
            cmap='plasma'
            ncolors= n_ki+1
            cmap_dummy = plt.get_cmap(cmap,ncolors-1)
            cmap = plt.get_cmap(cmap, ncolors)
            cmap.colors[0:n_ki] = cmap_dummy.colors[0:n_ki]
            cmap.colors[n_ki][0:3] = 1.0
            
            tags = []
            for i in range(n_ki):
                 tags.append("Q%i"%(i+1))
            tags.append('Invalid')
            qrates = np.array(tags)
            
            bounds = np.arange(0.5, n_ki +2.5, 1)
            
            norm = BoundaryNorm(bounds, cmap.N)
            fmt = FuncFormatter(lambda x, pos: qrates[norm(x)])
                    
            ax.matshow(d[a], cmap = cmap, norm = norm)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.25)
            im = ax.imshow(d[a], cmap=cmap, norm =norm)
            im.set_clim(vmin,vmax)
    
            cbar = fig.colorbar(im, cax=cax, orientation='vertical', \
              spacing='proportional', \
              ticks=np.linspace(1.0, float(n_ki)+1.0, n_ki+1), \
              format=fmt)
            
            cbar.minorticks_off()
            trans = ax.get_xaxis_transform()
        
            ax.annotate("B", xy= (0.0, -0.07), xycoords=trans, size=18, weight = 'bold')
            ax.annotate("A", xy= (8.5, -0.07), xycoords=trans, size=18, weight = 'bold')
            ax.annotate("D", xy= (-1.35, 0.05), xycoords=trans, size=18, weight = 'bold')
            ax.annotate("V", xy= (-1.35, 0.9), xycoords=trans, size=18, weight = 'bold')
        
        plt.subplots_adjust(top=b_top, bottom=b_bottom, left=b_left, 
                      right=b_right, hspace=b_hspace, wspace=b_wspace)
    
        if save: plt.savefig(dst+"."+fmt_,format=fmt_)

# =============================================================================
# CLASS mesh
# =============================================================================

# Attempt to simplify the routines from the database. A mesh manager for later
# operations.

def bin_aeration_compartments(data, mass, use_mass=False, bins=[0.0,0.1,0.5,0.9,1.0]):
        
    if use_mass:
        
        N = len(data)
        
        digit = np.digitize(data,bins)
        counter = {i:None for i in range(6)}
        
       
        for bin_ in range(6):
            counter[bin_] = np.count_nonzero(digit==bin_)
            
        nat = (counter[0]+counter[1])/N
        pat = counter[2]/N
        at = counter[3]/N
        hit = (counter[4]+counter[5])/N
        
    else:
                
        total_mass = np.sum(mass)
        digit = np.digitize(data,bins)
        mass_handler = {i:None for i in range(6)}
        
       
        for bin_ in range(6):
            mass_handler[bin_] = np.sum(mass[digit==bin_])
            
        nat = (mass_handler[0]+mass_handler[1])/total_mass
        pat = mass_handler[2]/total_mass
        at = mass_handler[3]/total_mass
        hit = (mass_handler[4]+mass_handler[5])/total_mass
        
    return nat, pat, at, hit

class mesh():
    def __init__(self,
              frame,
              npz_name = "Exp_NEW.npz",
              vtk_name = "Exp.vtk",
              verbose = True, minimum_binsize = 20):
        
        self.frame = frame
        
        self.npz_dir = "%s%s/%s"%(frame.address, frame.mesh_dir,
                        npz_name)
        
        if (not os.path.isfile(self.npz_dir) and verbose):
            print("error: npz file not found for '%s'"%(frame.codename))
            self.npz_size = None
        else:
            self.npz_size  = os.path.getsize(self.npz_dir)/1024**2
        self.npz = None
        self.npz_fields = None
        self.npz_masks = None
        self.npz_len = None
        self.roi_ids = {}
        
        self.ki_exclusion_criteria = 1e-8
        self.ki_bins = None
        
        self.matrix_ids = None
        self.matrix_validity = None
        self.matrix_generated = False
        self.matrix_minimum_binsize = minimum_binsize
        
        self.vtk_dir = "%s%s/%s"%(frame.address, frame.mesh_dir, vtk_name)
        if (not os.path.isfile(self.vtk_dir) and verbose):
            print("error: vtk file not found for '%s'"%(frame.codename))
            self.vtk_size = None
        else:
            self.vtk_size  = os.path.getsize(self.vtk_dir)/1024**2
        self.vtk = None

# Load a FE mesh saved as *.npz. such as 'Exp_NEW.npz'
    def load_npz(self):
        self.npz = np.load(self.npz_dir)
        self.npz_fields = list(self.npz.keys())
        self.npz_masks = {} 
        self.npz_len = self.npz['M'].shape[0]

# Compute the global Ki for a subject. Used for PET/CT data. 
    def compute_gki(self):
        # Make sure the subject was complete loaded.
        if 'Ki' not in self.npz_fields:
            return np.nan
        else:
            mask = self.npz['Ki'] > self.ki_exclusion_criteria
            s = DescrStatsW(self.npz['Ki'][mask], weights=self.npz['M'][mask])
            return s.mean

    def introduce_yourself(self):
        print("Hello, I'm %i"%self.subject)

# Generate ROI ids in a given direction.
    def generate_ids(self, field='Mass', ba = [0.,0.,1.], vd = [0.,1.,0.], \
                rl = [1.,0.,0.], nrois_ba = 10, nrois_vd = 10, nrois_rl = 3):
        # Generate directions. 
        ba = np.mat(ba).T;         vd = np.mat(vd).T;        rl = np.mat(rl).T
        
        _, ba_id = ROI.ISO_ROIAnalysis(self.vtk_dir, field, nrois_ba, ba)
        _, vd_id = ROI.ISO_ROIAnalysis(self.vtk_dir, field, nrois_vd, vd)
        _, rl_id = ROI.ISO_ROIAnalysis(self.vtk_dir, field, nrois_rl, rl)
        self.roi_ids = {"ba":ba_id,"vd":vd_id,"rl":rl_id, "nba":nrois_ba,
                  "nvd":nrois_vd, "nrl":nrois_rl}


# Binize the data into a matrix-shaped array. Used for the 2- or 3- dimensional
# matrix studies. 
# rev_(dir): Flip the data in a given direction.
# blocked_list: Allows to skip some basal-to-apical ROIS that which have
# missing data or poor segmentations. 
# block_ba: Flag for the blocking of the basal-to-apical ROIS.

    def generate_matrix(self, rev_vd = False, rev_ba = False, rev_rl = False,
                     block_ba = True, blocked_list = [0,1,8,9]):
        n_vd = self.roi_ids["nvd"]
        n_ba = self.roi_ids["nba"]
        n_rl = self.roi_ids["nrl"]
        
        basis = np.arange(self.npz_len) # Dummy set for comparison
        base_holder = {} # Mask holder
        for i in range(n_vd): # VD
            loc_vd = set(basis[self.roi_ids["vd"]==i])
            if rev_vd: i = n_vd - i - 1
            for j in range(n_ba):  # BA
                local = loc_vd.intersection(set(basis[self.roi_ids["ba"]==j]))
                if rev_ba: j = n_ba - j -1
                base_holder.update({(i,j):local})
        
        holder = {}
        for k in range(n_rl): # for each of the rl rois
            local_rl = set(basis[self.roi_ids["rl"]==k])
            if rev_rl: k = n_rl - k - 1
            for i in range(n_vd):
                for j in range(n_ba):
                    local = local_rl.intersection(base_holder[(i,j)])
                    holder.update({(i,j,k):local})
        
        # Empty 10x10 matrix filled with False values.
        validity = []
        for k in range(n_rl):
            l_validity = np.zeros((n_vd,n_ba), dtype=bool)
            # Check every maskholder, if there is something there, then True.
            for i in range(n_vd):
                for j in range(n_ba):
                    if len(holder[(i,j,k)])>self.matrix_minimum_binsize:
                        l_validity[i,j] = True
                    if block_ba:
                        if j in blocked_list:
                            l_validity[i,j] = False
                        
            validity.append(l_validity)

        self.matrix_ids = holder
        self.matrix_validity = np.array(validity)
        self.matrix_generated = True
        
# To quantize Ki is an idea to compare Ki between different subjects. Data 
# showed that this scalar is very subject-dependent. To quantize them is
# basically to express their belonging to a quantile rather than looking into 
# their specific values.

    def matrix_quantize_ki(self, nbins = 6):
        self.ki_bins = nbins
        means, _, masses, _ =  self.extract_matrix_data('Ki')
        mask = np.logical_not(np.isnan(np.array(means)))
        means = np.array(means); masses = np.array(masses)
        stats = DescrStatsW(means[mask], weights=masses[mask])
        bins = []
        for q in np.linspace(0.0, 1.0, nbins + 1):
            bins.append(float(stats.quantile(q)))
        bins[nbins] *= 1.01
        digits = np.digitize(means, bins)
        return digits, bins

# To compute the Strain Heterogeneity Index std(field/mean(field).
# It is computed originally with 'I3_U' = Jacobian.

    def matrix_compute_shi(self, field = 'I3_U'):
        means, stds, masses, _ = self.extract_matrix_data(field)
        shi = np.array(stds)/np.array(means)
        return shi


# Read the mesh and generate the matrix-shaped data.

    def extract_matrix_data(self, field):
        
        if not self.matrix_generated: 
            print("'matrix_generated=False' which leads to failure")
            return

        # Recover the data and the mass.
        data = self.npz[field]
        mass = self.npz['M']
        means = []; stds = []; masses = []; counts = []
        n_vd = self.roi_ids["nvd"]
        n_ba = self.roi_ids["nba"]
        n_rl = self.roi_ids["nrl"]

        # Holders for the means, stds and mass.
        for k in range(n_rl):
            l_means = np.full((n_vd,n_ba),np.nan) 
            l_stds = np.full((n_vd,n_ba),np.nan) 
            l_mass = np.full((n_vd,n_ba),np.nan) 
            l_count = np.zeros((n_vd,n_ba)) 
            
            for i in range(n_vd):
                for j in range(n_ba):
                    if self.matrix_validity[k][i,j]:
                        # Local mask holder
                        mask = list(self.matrix_ids[(i,j,k)])
                        empty = False
                        # Compute the weighted statistics.
                        # If field is 'Ki', divide it with its global value.
                        if field == 'Ki':
                            sample_data = data[mask]/self.compute_gki()
                            sample_mass = mass[mask]
                            submask = data[mask] > self.ki_exclusion_criteria
                            sum_l_mass = np.sum(sample_mass[submask])
                            counter = np.shape(sample_mass[submask])[0]
                            
                            if len(submask)>self.matrix_minimum_binsize:
                                stats = DescrStatsW(sample_data[submask],
                                             weights=sample_mass[submask])
                            else:
                                empty = True
                        else:
                            stats = DescrStatsW(data[mask],
                                             weights=mass[mask])
                            
                            sum_l_mass = np.sum(mass[mask])
                            counter = np.shape(mass[mask])[0]
                        if not empty:
                            l_means[i,j] = stats.mean
                            l_stds[i,j] = stats.std
                            l_mass[i,j] = sum_l_mass
                            l_count[i,j] = counter
                        else:
                            l_means[i,j] = np.nan
                            l_stds[i,j] = np.nan
                            l_mass[i,j] = 0.0
                            l_count[i,j] = len(self.matrix_ids[(i,j,k)])

            means.append(l_means)
            stds.append(l_stds)
            masses.append(l_mass)
            counts.append(l_count)

        return np.array(means),np.array(stds),np.array(masses),np.array(counts)

    def extract_matrix_aeration_compartments(self, field='eigf', use_mass=False):
        
        compartments = ['nat','pat','at','hit']
        
        if not self.matrix_generated: 
            print("'matrix_generated=False' which leads to failure")
            return

        # Recover the data and the mass.
        data = self.npz[field]
        mass = self.npz['M']
        means = []; stds = []; masses = []; counts = []
        n_vd = self.roi_ids["nvd"]
        n_ba = self.roi_ids["nba"]
        n_rl = self.roi_ids["nrl"]
        
        # Compartment handler
        compartment_handler = {comp:[] for comp in compartments}
        
        # Holders for the means, stds and mass.
        for k in range(n_rl):
            
            # Compartment handlers
            l_compartments = {comp:np.full((n_vd,n_ba),np.nan)  for comp in compartments}
            
            for i in range(n_vd):
                for j in range(n_ba):
                    if self.matrix_validity[k][i,j]:
                        # Local mask holder
                        mask = list(self.matrix_ids[(i,j,k)])
                        
                        empty = not len(mask)>self.matrix_minimum_binsize
                        
                        # Compute the weighted statistics.
                        nat, pat, at, hit = bin_aeration_compartments(data[mask], mass[mask], use_mass=use_mass)
                            
                        if not empty:
                            l_compartments['nat'][i,j] = nat
                            l_compartments['pat'][i,j] = pat
                            l_compartments['at'][i,j] = at
                            l_compartments['hit'][i,j] = hit
            
            for comp in compartments:
                compartment_handler[comp].append(l_compartments[comp])

        return compartment_handler


# =============================================================================
# CLASS bound
# =============================================================================

# A bound helps you to keep all your matrices together and use their data for 
# an extended analysis. 

class bound():

    def __init__(self, kmeans_clusters = 3):
        self.matrices = {}
        self.subjects = []
        self.tempsets = {}
        self.marker_list = ["o","o","o", "o", "o"]
        self.color_list = ["r","b","k","g","c"]
        self.marker_radius = 4
        self.markersize = np.pi * self.marker_radius**2
        self.alpha = 0.3
        self.grid_length = 0
        self.nrois_ba = None
        self.nrois_vd = None
        self.nrois_rl = None
        self.sortedsubjects = []
        self.kmeans_clusters = kmeans_clusters

# Load a matrix into the bound

    def load(self, M):
        self.matrices.update({M.subject:M})
        self.subjects.append(M.subject)
        if self.grid_length == 0:
            self.grid_length = M.grid_length
            self.nrois_ba = M.nrois_ba
            self.nrois_vd = M.nrois_vd
            self.nrois_rl = M.nrois_rl

# Apply a sorting scheme to all the matrices within the bound
        
    def sort(self, field, state, criteria, markersize = 5, fillstyle="full"):
        for mk in self.matrices:
            if state in self.matrices[mk].directory:
                if field in self.matrices[mk].directory[field]:
                    self.matrices[mk].sort(field, state, criteria,
                         markersize = markersize, fillstyle=fillstyle)

# Remove any traces of a sorting in the bound.

    def reset(self):
        for mk in self.matrices:
            self.matrices[mk].resetSorting()

# A tempset will relate information from different states, for instance, a 
# tempset might be called 'start' and other 'end'. There, an internal
# dictionary should relate subject:state for each tempset. Then, you can 
# use that information for further analysis.

    def add_tempset(self,tempset):
        self.tempsets.update(tempset)

# Extract a field from a tempset, this considers all the subjects within the
# tempset.
        
    def extract_ts_data(self, ts_tag, field):
        if not ts_tag in self.tempsets.keys():
            return 
        
        data = np.empty(0)
        for s in self.subjects:
            if s in self.tempsets[ts_tag].keys():
                state = self.tempsets[ts_tag][s]
                data = np.append(data, self.matrices[s].returnSortedData(field, state))
        
        return data

# This crosses the information from two different tempsets. This is useful
# whenever you are interested, for instance, in a scatter plot in which 
# for each coordinate you will use data from a different state. This will
# exclude subjects that have no valid counterpart in the other tempset and 
# will remove invalid points within a subject. This should return you clean
# data ready for analysis.

    def cross_ts_data(self, ts_tag0, ts_tag1, fields0, fields1, 
                   banned_subjects = []):
        
        if not ts_tag0 in self.tempsets.keys():
            return 
        if not ts_tag1 in self.tempsets.keys():
            return 
        
        set0 = set(self.tempsets[ts_tag0]) # Take the subjects in t0
        set1 = set(self.tempsets[ts_tag1]) # Take the subjects in t1

        subjects = list(set0.intersection(set1)) # Intersect
        subjlist = np.empty(0) # Tool to identify datapoints to subjects
        data0 = {}; valid0 = {} # holders
        data1 = {}; valid1 = {} # holders
        pointlist = np.empty(0)
        
        for f in fields0:
            data0.update({f:np.empty(0)}) # create empty storages
            valid0.update({f:np.empty(0)}) #  idem
        for f in fields1:
            data1.update({f:np.empty(0)}) # idem
            valid1.update({f:np.empty(0)})  #idem

        for s in subjects:
            
            if not s in banned_subjects:  # A list to exclude some subjects
        
                state = self.tempsets[ts_tag0][s] # Extract the state 
                for f in fields0: # For each required field at t1
                    d, v = self.matrices[s].returnSortedData(f, state) # Extract raw data
                    data0[f] = np.append(data0[f],  d)  # place it
                    valid0[f] = np.append(valid0[f],  v) # Validators to include or not the subROI
                
                state = self.tempsets[ts_tag1][s] # Same as t1 but to t2
                for f in fields1:
                    d, v = self.matrices[s].returnSortedData(f, state)
                    data1[f]= np.append(data1[f], d)
                    valid1[f]= np.append(valid1[f], v)
                
                # Keep track of which data point belongs to which subject
                subjlist = np.append(subjlist, np.full(d.shape,s))
                pointlist = np.append(pointlist, np.array(self.matrices[s].point_keys))
                 
        # Establish a mask
        v = np.full(data1[list(data1.keys())[0]].shape,True)
        
        # Through "And" operations, get the final mask that is completely consistent
        for f in list(valid0.keys()): 
            v = np.logical_and(v, valid0[f])
        for f in list(valid1.keys()):
            v = np.logical_and(v,valid1[f])
        
        for f in list(data0.keys()):
            data0[f] = data0[f][v]
        for f in list(data1.keys()):
            data1[f] = data1[f][v]

        return data0, data1, subjlist[v], pointlist[v].astype(int)

# Returns a global validity matrix that considers infromation for every 
# subject that is valid for both temporal sets in study.

    def cross_validity(self, set0='start', set1='fire_start', field = 'VS', 
                    banned_subjects = []):
        s0 = set(self.tempsets[set0])
        s1 = set(self.tempsets[set1])
        subjects = list(s0.intersection(s1))
        for s in subjects:
            
            if s not in banned_subjects:
                state0 = self.tempsets[set0][s]
                state1 = self.tempsets[set1][s]
                _, v0 = self.matrices[s].returnSortedData(field, state0)
                _, v1 = self.matrices[s].returnSortedData(field, state1)
                v = np.logical_and(v0,v1)
                self.matrices[s].global_validity = v
            

# This performs a k-means analysis for data from two tempsets of interest.
# First, a principal component analysis and then the k-means operations. This
# has been tailor made for the dataset of the VILI pigs. This could be modified
# for your needs.

    def defineClusters(self, set0='start', set1='fire_start',
                     plotData = False, returnData = True, include_ki = False,
                     ki_field = 'qKi', banned_subjects = [], 
                     do_not_recalc = False, read_ki = False,
                     save = True, fmt = 'pdf', dst = '', root = "kmeans.%s"):
        
        list0 = ['VS', 'Intensity_Exp','Intensity_Insp','Delta_Gas_Fraction']
        list1 = ['Intensity_Exp','Intensity_Insp','Delta_Gas_Fraction', 'VS', 
            ki_field]
        
        if not read_ki:
            list1.remove(ki_field)
        
        # Extracting the data
        d0, d1, subjlist, pointlist = self.cross_ts_data(set0, set1,  \
            list0 , list1, banned_subjects=banned_subjects)
        
        # Early dataset
        vs0 = d0['VS'].reshape(-1,1)
        ie0 = d0['Intensity_Exp'].reshape(-1,1)
        ii0 = d0['Intensity_Insp'].reshape(-1,1)
        gf0 = d0['Delta_Gas_Fraction'].reshape(-1,1)
        # Late dataset
        ie1 = d1['Intensity_Exp'].reshape(-1,1)
        ii1 = d1['Intensity_Insp'].reshape(-1,1)
        gf1 = d1['Delta_Gas_Fraction'].reshape(-1,1)
        vs1 = d1['VS'].reshape(-1,1)
        
        if read_ki:
            ki1 = d1[ki_field].reshape(-1,1)
        else:
            ki1 = vs1.copy()
            
        # Normalization
        vs0_ = StandardScaler().fit_transform(vs0)
        ie0_ = StandardScaler().fit_transform(ie0)
        gf0_ = StandardScaler().fit_transform(gf0)
        ie1_ = StandardScaler().fit_transform(ie1)
        gf1_ = StandardScaler().fit_transform(gf1)
        vs1_ = StandardScaler().fit_transform(vs1)
        ki1_ = StandardScaler().fit_transform(ki1)
        
        # Formatting
        if include_ki:
            X = np.hstack([vs0_,ie0_, gf0_, vs1_, ie1_, gf1_, ki1_])
        else:
            X = np.hstack([vs0_,ie0_, gf0_, vs1_, ie1_, gf1_])
            
        # Setting up the PCA analysis
        pca = PCA(n_components=3)
        pca.fit(X)
        
        # Explained variance and component definition
        print("Explained Variance:")
        print(pca.explained_variance_)
        self.explained_variance = pca.explained_variance_
        print("Principal components:")
        print(pca.components_)
        self.components = pca.components_

        # Extracting the new variables (PCA_{i})
        projected = pca.fit_transform(X)
        
        # K-Means and extracting the new labels
        model = KMeans(n_clusters= self.kmeans_clusters)
        model.fit(projected[:,:2])
        
        if not do_not_recalc: # this flag avoids recomputing the flags
            # this allows for a replot that does not changes the colors
            # notice that the color changes if recomputated. model starting
            # seeds are random, thus changing every time the method is called.
            self.kmeans_labels = model.labels_


        self.sortedsubjects = list(np.unique(subjlist).astype(int))
        
        # Take each subject
        for subject in self.sortedsubjects:
            # Take a generic point list
            reference_pointlist = list(range(self.grid_length))
            # Extract a mask that isolates a subject from the group
            mask = subjlist == subject
            # Isolate the points belonging to the studied subject.
            points = pointlist[mask].astype(int)
            # Isolate the labels that apply to such subject
            label = self.kmeans_labels[mask].astype(int)
            # Find those points that were left out
            not_group = list(set(reference_pointlist) - set(points))
            # A dictionary is created that holds the label information
            pca_pointlist = {}
            for l in np.unique(self.kmeans_labels).astype(int):
                pca_pointlist.update({l:points[label==l]})
            pca_pointlist.update({"NOT":not_group})
            # Such dictionary is loaded to the matrix
            self.matrices[subject].sort("dummy","dummy","PCA",
                 pca_pointlist = pca_pointlist)
    
        if plotData:
        #plt.scatter(projected[:,0], projected[:,1], color='black',alpha=0.2)
            nr = 3; nc = 1; scale_factor = 6
            fig, ax = plt.subplots(nrows=nr, ncols =nc, \
                                     figsize = (scale_factor*nc, scale_factor* nr))
            ax=ax.flatten()
            ax[0].set_title("No clustering")
            ax[0].scatter(projected[:,0],projected[:,1], alpha=self.alpha, 
                 color="k", s=self.markersize)

            ax[0].set_ylabel("PC2")
            ax[0].set_xlabel("PC1")
            
            
            ax=ax.flatten()
            ks = range(1,10)
            inertias = []
            for k in ks:
                model = KMeans(n_clusters=k)
                model.fit(projected)
                inertias.append(model.inertia_)
            ax[1].plot(ks, inertias, '-o', color='k')
            ax[1].set_xlabel('Number of clusters, k')
            ax[1].set_ylabel('Inertia')
            ax[1].set_xticks(ks)
            
            model = KMeans(n_clusters = self.kmeans_clusters)
            model.fit(projected[:,:2])
            
            for n,z in zip(range(self.kmeans_clusters),self.color_list):
                mask = self.kmeans_labels == n
                ax[2].scatter(projected[:,0][mask],projected[:,1][mask], 
                      alpha=self.alpha, color=z, s=self.markersize)
                
            ax[2].set_ylabel("PC2")
            ax[2].set_xlabel("PC1")
            ax[2].set_title("K-Means clustering")
            
            plt.subplots_adjust(top=0.90, bottom=0.10, left=0.13, right=0.95,
                                           hspace=0.20, wspace=0.20)    
            
            if save: plt.savefig(dst+root%fmt, format=fmt)

        if returnData:
            return {"vs0":vs0, "vs1":vs1, "gf0":gf0, 
              "gf1":gf1,"ie0":ie0,"ie1":ie1, "ii0":ii0, "ii1":ii1, "ki":ki1,
              "subjlist":subjlist, "pointlist":pointlist, 
              "labels":self.kmeans_labels}

# Automatically load the data specified into the bound using the given subject
# list and the timesets (tempset) information and the fields specified.

def load_bound(b, subjlist, timesets, F, fields,
               nrois_ba = 10, nrois_vd = 10, nrois_rl = 3):
    
    for subject in subjlist:
        for ts in list(timesets.keys()):
            
            state = timesets[ts][subject]
            print("Subject %i, state %s... uploading."%(subject,state))


            f_ = frame(F.keyword, subject, state, F.cut,
                         image_filter = F.image_filter,
                         path_to_dir = F.path_to_dir)
            

            m = mesh(f_)
            m.load_npz()
            m.generate_ids(nrois_vd=nrois_vd, nrois_ba = nrois_ba)
            m.generate_matrix(block_ba=False)

            if subject not in b.matrices:
                M = matrix(m.frame.number, nrois_ba = nrois_ba, 
               nrois_vd=nrois_vd, nrois_rl = nrois_rl)

            for f in fields:
                if f == 'SHI': continue
                means, stds, masses, counts = m.extract_matrix_data(f)
                M.load_slice(m.frame, f, means, m.matrix_validity)
            
            # parche chasquilla para calcular SHI
            f = 'I3_U'
            shi = []
            means, stds, masses, counts = m.extract_matrix_data(f)
            for mean, std in zip(means, stds):
                shi.append(std/mean)
            M.load_slice(m.frame, 'SHI', shi, m.matrix_validity)
            b.load(M)

# Generate annotations into the plots using categories from Gattinoni's papers. 
# Creates vertical and/or horizontal lines that help you distinguish a regime
# of intensities from another. 

def annotate_aeration(ax, xlines=False, ylines=False, alpha = 0.3, color = "k",
                      delta_factor = 0.0, 
                      x_rotation = 45, y_rotation = 45, text_factor = 0.02,
                      region_names = ["Hyperinflated tissue",
                      "Normally-aerated tissue", "Poorly-aerated tissue",
                     "Non-aerated tissue"], linestyle = '--',
                      region_locations = [-950, -700, -300, 0],
                      lines_locations = [-900,-500,-100], ):
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()
    
    Dx = (xrange[1] - xrange[0])
    Dy = (yrange[1] - yrange[0])

    if xlines:
        y_base = yrange[1] + text_factor * Dy
        delta = Dx * delta_factor
        for r,l in zip(region_names, region_locations):
            ax.text(l+delta,y_base,r, rotation=x_rotation, color = color, alpha = alpha)
        for l in lines_locations:
            ax.axvline(l, color=color, linestyle=linestyle, alpha = alpha)
        
    if ylines:
        
        delta = Dy * delta_factor
        x_base = xrange[1] + text_factor * Dx
        for r,l in zip(region_names, region_locations):
            ax.text(x_base, l+delta,r, rotation=y_rotation, color = color, alpha = alpha)
        for l in lines_locations:
            ax.axhline(l, color=color, linestyle=linestyle, alpha = alpha)

# A variation of the routine that crosses the temporal data.

def cross_primal_variables(b, subject_list, side = 8, 
                           ts0 = 'start', ts1='NOKI-LF',
                           alpha = 0.25, banned_subjects = [], save = False,
                           dst_dir = "", root = "%s_against_%s___labeled.%s",
                           fmt = 'pdf', ki_field = 'qKi',
                           b_top=0.80, b_bottom=0.10,
                           b_left=0.10, b_right=0.80,
                           b_hspace=0.10, b_wspace=0.30,
                           read_ki = True, correlation = False,
                           annotate_intensities = True, verbose = False):
    
    if verbose: print(1)
    list0 = ['VS', 'Intensity_Exp','Intensity_Insp', 'Delta_Gas_Fraction', 'SHI']
    list1 = ['VS', 'Intensity_Exp','Intensity_Insp', 'Delta_Gas_Fraction', 'SHI', ki_field]
    if verbose: print(2)

    if not read_ki:
        list1.remove(ki_field)
    if verbose: print(3)

    d0, d1, subjlist, pointlist = b.cross_ts_data(ts0, ts1, list0, list1,
                                        banned_subjects = banned_subjects)
    if verbose: print(4)

    # Early dataset
    vs0 = d0['VS'].reshape(-1,1)
    ie0 = d0['Intensity_Exp'].reshape(-1,1)
    ii0 = d0['Intensity_Insp'].reshape(-1,1)
    gf0 = d0['Delta_Gas_Fraction'].reshape(-1,1)
    sh0 = d0['SHI'].reshape(-1,1)

    # Late dataset
    ie1 = d1['Intensity_Exp'].reshape(-1,1)
    ii1 = d1['Intensity_Insp'].reshape(-1,1)
    gf1 = d1['Delta_Gas_Fraction'].reshape(-1,1)
    vs1 = d1['VS'].reshape(-1,1)
    sh1 = d1['SHI'].reshape(-1,1)

    if read_ki:
        ki1 = d1[ki_field].reshape(-1,1)
    else:
        ki1 = vs1.copy() # dummy
    if verbose: print(5)

    label_vs = "Volumetric Strain (%s)"
    label_gf = "Delta Gas Fraction (%s)"
    label_ie = "Intensity Exp (%s)"
    label_ii = "Intensity Insp (%s)"
    label_ki = "Ki quantile (%s)"
    label_shi = "Strain Heterogeneity Index (%s)"
    l1 = "Late"; l0 = "Early"
    if verbose: print(6)

    range_vs = (-100,250)
    range_gf = (0,0.5)
    range_i = (-1000,100)
    range_ki = (0,6.5)
    range_shi = (0.0, 1.0)
    
    data = {"vs0":vs0,"ie0":ie0, "ii0":ii0,"gf0":gf0, "vs1":vs1, "ie1":ie1, "ii1":ii1, "gf1":gf1,
         "ki1":ki1, "sh0":sh0, "sh1":sh1}
    
    ranges = {"vs0":range_vs,"ie0":range_i, "ii0":range_i, "gf0":range_gf,  \
           "vs1":range_vs, "ie1":range_i, "ii1":range_i, "gf1":range_gf, \
           "ki1":range_ki, "sh0":range_shi, "sh1":range_shi}
    
    labels = {"vs0":label_vs, "ie0":label_ie, "ii0":label_ii, "gf0":label_gf, \
           "vs1":label_vs, "ie1":label_ie, "ii1":label_ii, "gf1":label_gf, \
           "ki1":label_ki, "sh0":label_shi, "sh1":label_shi}

    if verbose: print(7)

    for field_x in list(data.keys()):
        
        a = 0
        
        for field_y in list(data.keys()):
            
            if not field_x == field_y: 
            
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize = (side,side))
                
                for k_,z, m in zip(range(b.kmeans_clusters),b.color_list, ["o","s","^", "D"]):
                    mask = b.kmeans_labels == k_
                    print (k_, field_x, field_y)
                    print(data[field_x].shape, data[field_y].shape)
                    ax.scatter(data[field_x][mask], data[field_y][mask], \
                         color=z, alpha=alpha, marker=m)
                    
                    timex = l1 if field_x[2] == '1' else l0
                    timey = l1 if field_y[2] == '1' else l0
            
                ax.set_xlim(ranges[field_x])
                ax.set_xlabel(labels[field_x]%timex)
        
                ax.set_ylim(ranges[field_y])
                ax.set_ylabel(labels[field_y]%timey)
                    
                if field_x[0:2] == field_y[0:2]:
                    ylim = ax.get_ylim()
                    ax.plot(ylim, ylim, color = "grey", linestyle = '--', alpha=alpha)

                annotate_x_intensities = True if field_x[0] == "i" else False
                annotate_y_intensities = True if field_y[0] == "i" else False

                annotate_aeration(ax, 
                      xlines = annotate_x_intensities and annotate_intensities,
                      ylines = annotate_y_intensities and annotate_intensities)

            
                plt.subplots_adjust(top=b_top, bottom=b_bottom, left=b_left, 
                              right=b_right, hspace=b_hspace, wspace=b_wspace)
                
                if save: plt.savefig(dst_dir+root%(field_x,field_y,fmt), format=fmt)
                
                a +=1
                
    if verbose: print(8)


    if correlation:
        correlation_matrix = {}
        
        from scipy.stats import spearmanr
        for field_x in list(data.keys()):
            for field_y in list(data.keys()):
                correlation_matrix.update({(field_x,field_y):spearmanr(data[field_x],data[field_y]).correlation})
        return correlation_matrix



# =============================================================================
#                 m   a   i   n             c   o   d   e
# =============================================================================

# Load the data and get it ready for some compuations.

if __name__ == "__main2__":

    timesets = {'start':{2:'00h', 3:'03h', 4:'00h', 5:'00h', 6:'03h', 7:'00h', 
                         8:'03h', 9:'00h'}, 
                'fire_start':{2:'12h', 3:'12h', 5:'06h', 6:'12h', 7:'09h',
                         11:'12h'},
                'NOKI-FS':{2:'12h', 3:'09h', 4:'06h', 5:'03h', 6:'09h', 
               7:'09h', 8:'06h', 9:'03h'},
                'NOKI-LF':{2:'12h', 3:'12h', 4:'12h', 5:'06h', 6:'12h', 
               7:'09h', 8:'12h', 9:'09h'}} 
    
    ts0 = 'start'
    ts1 = 'NOKI-LF'

    b = bound()
    b.add_tempset(timesets)

    ki_bins = 6
    save = False
    nrois_vd = 10
    nrois_ba = 10
    nrois_rl = 3
    
    matrixBins_to_meshIds = {}

    subjlist= list(set(timesets[ts0].keys()).intersection(set(timesets[ts1].keys())))
    series_key = "paper"
    include_ki = False
    banned_subjects = []
    #no subject 7 and include ki
    
    dst = '/home/user/Projects/paper_VILI/GRAPHS/'
    dst = '%s%s_'%(dst, series_key)
    
    fields_to_load = ['VS','Intensity_Exp','Intensity_Insp', 'Delta_Gas_Fraction']
    if include_ki: fields_to_load.append('Ki')
    
    for subject in subjlist:
        for ts in [ts0, ts1]:
            state = timesets[ts][subject]
            print("Subject %i, state %s... uploading."%(subject,state))

            f = frame('PIG', subject, state, 'wl', 
                         path_to_dir = "D:/BMA/SUBJECTS/VILI_PIGS")

            m = mesh(f)
            m.load_npz()
            m.generate_ids(nrois_vd=nrois_vd, nrois_ba = nrois_ba)
            m.generate_matrix(block_ba=False)

            # save here the relationship between independent the nodes belonging to
            # each (subject and state) meshes with the corresponding matrix elems
            # in a dictionary 'keyed' by (i,j,k) tuples.
            matrixBins_to_meshIds.update({(subject,state):m.matrix_ids})

            if subject not in b.matrices:
                M = matrix(m.frame.number, ki_bins=ki_bins,
                nrois_ba=nrois_ba,nrois_vd=nrois_vd)
            for f in fields_to_load:
                means, stds, masses, counts = m.extract_matrix_data(f)
                M.load_slice(m.frame, f, means, m.matrix_validity)
                
            
            f = 'I3_U'
            shi = []
            means, stds, masses, counts = m.extract_matrix_data(f)
            for mean, std in zip(means, stds):
                shi.append(std/mean)
            M.load_slice(m.frame, 'SHI', shi, m.matrix_validity)
            
            if ts == ts0:
                i3_0 = means
            elif ts == ts1:
                i3_1 = means
        
                spi = []
                for v0, v1 in zip(i3_0, i3_1):
                    spi.append(v1/v0)
                M.load_slice(m.frame, 'SPI', spi, m.matrix_validity)

#******************************************************************************
# APPENDED: Used to compute the aeration compartments
#******************************************************************************
            # identify the tuple subject,state
            key = (subject, state)

            # retrieve the ids associated to each roi
            matrixIds = matrixBins_to_meshIds[key]
        
            
            # load the required fields for the subject/state
            db.CTs[subject][state].loadMesh()
            db.CTs[subject][state].loadMasks()
            
            # compute the volume of the original mask
            db.CTs[subject][state].masks['NEW Exp'].computeVolume()
            V_ee = db.CTs[subject][state].masks['NEW Exp'].rawVolume
            
            # load the mesh
            msh = db.CTs[subject][state].meshes['Exp_NEW.npz']
            npz = msh.getNumpyMesh()
            
            # retrieve the mass and sum it up
            mass = npz['M']
            total_mass = np.sum(mass)
            
            # retrieve the jacobian and compute the mass amplified by the 
            # jacobian to obtain a mass indicator that has been weighted by the
            # deformation associated to the inspiration.
            jacobian = (npz['VS']/100+1)
            expanded_mass = mass*jacobian
            total_exp_mass = np.sum(expanded_mass)
            
            # compute a global jacobian; the idea is to compare it with the
            # global volumetric strain obtained from the masks in order to 
            # validate this approach
            mean_jacobian = total_exp_mass/total_mass
            
            # retrieve the intensities
            intensity_exp = npz['Intensity_Exp']
            intensity_insp = npz['Intensity_Insp']
            
            # placeholder for the fields of interest
            h_ee_nat = [] # Non-aerated tissue
            h_ee_pat = [] # Poorly-aerated tissue
            h_ee_nit = [] # Normally-Insuflated (aerated) tissue
            h_ee_hit = [] # Hyperinflated tissue
            h_ee_mass = [] # Mass
            
            h_ei_nat = [] # Non-aerated tissue
            h_ei_pat = [] # Poorly-aerated tissue
            h_ei_nit = [] # Normally-Insuflated (aerated) tissue
            h_ei_hit = [] # Hyperinflated tissue
            h_ei_mass = [] # Mass
            
            
    
            # for each bin in the left-to-right (maybe flipped?) direction
            for k in range(3):
                
                ee_nat = np.full((10,10),0.0) # Non-aerated tissue
                ee_pat = np.full((10,10),0.0) # Poorly-aerated tissue
                ee_nit = np.full((10,10),0.0) # Normally-Insuflated (aerated) tissue
                ee_hit = np.full((10,10),0.0) # Hyperinflated tissue
                ee_mass = np.full((10,10),np.nan) # Mass
                
                ei_nat = np.full((10,10),0.0) # Non-aerated tissue
                ei_pat = np.full((10,10),0.0) # Poorly-aerated tissue
                ei_nit = np.full((10,10),0.0) # Normally-Insuflated (aerated) tissue
                ei_hit = np.full((10,10),0.0) # Hyperinflated tissue
                ei_mass = np.full((10,10),np.nan) # Mass
                
                for i in range(10): #vd # TODO: CHECK THIS
                    for j in range(10): #ba  # TODO: CHECK THIS
            #       for k in range(3): # rl
                        int_key = (i,j,k)
                            
                        # copy the nodes belonging to the local ROI
                        nodes = list(matrixIds[int_key])
                            
                        # if there are nodes in the ROI (not empty)
                        if len(nodes)>0:
                            
                            # retrieve local fields of interest
                            local_ie = intensity_exp[nodes]
                            local_ii = intensity_insp[nodes]
                            local_mass = mass[nodes]
                            local_mass_exp = expanded_mass[nodes]

                            #------------ End-expiration values----------------

                            mask_ee_nat = local_ie<-900.0 # Non-aerated tissue
                            mask_ee_pat = np.logical_and(local_ie<-500.0, # Poorly-aerated tissue
                                                         local_ie>-900.0)

                            mask_ee_nit = np.logical_and(local_ie<-100.0, # Normally-insuflated tissue
                                                         local_ie>-500.0)
                            mask_ee_hit = local_ie>-100.0 # Hyperinflated tissue
                            
                            # local mass
                            ee_mass[i,j] = np.sum(local_mass)

                            # if there are NAT nodes, add the corresponding 
                            # mass to the holder
                            if np.count_nonzero(mask_ee_nat):
                                ee_nat[i,j] = np.sum(local_mass[mask_ee_nat])/ee_mass[i,j]
                            
                            # Poorly aerated tissue at end-expiration
                            if np.count_nonzero(mask_ee_pat):
                                ee_pat[i,j] = np.sum(local_mass[mask_ee_pat])/ee_mass[i,j]
                            
                            # Normally insuflated tissue at end-expiration
                            if np.count_nonzero(mask_ee_nit):
                                ee_nit[i,j] = np.sum(local_mass[mask_ee_nit])/ee_mass[i,j]
                            
                            # Hyperinflated tissue
                            if np.count_nonzero(mask_ee_hit):
                                ee_hit[i,j] = np.sum(local_mass[mask_ee_hit])/ee_mass[i,j]
                                                        
                            #------------ End-inspiration values---------------
                            
                            mask_ei_nat = local_ii<-900.0 # Non-aerated tissue
                            mask_ei_pat = np.logical_and(local_ii<-500.0, # Poorly-aerated tissue
                                                         local_ii>-900.0)

                            mask_ei_nit = np.logical_and(local_ii<-100.0, # Normally-insuflated tissue
                                                         local_ii>-500.0)
                            mask_ei_hit = local_ii>-100.0 # Hyperinflated tissue
                            
                            ei_mass[i,j] = np.sum(local_mass_exp)

                            # if there are NAT nodes, add the corresponding 
                            # mass to the holder
                            if np.count_nonzero(mask_ei_nat):
                                ei_nat[i,j] = np.sum(local_mass[mask_ei_nat])/ei_mass[i,j]
                            
                            # Poorly aerated tissue at end-expiration
                            if np.count_nonzero(mask_ei_pat):
                                ei_pat[i,j] = np.sum(local_mass[mask_ei_pat])/ei_mass[i,j]
                            
                            # Normally insuflated tissue at end-expiration
                            if np.count_nonzero(mask_ei_nit):
                                ei_nit[i,j] = np.sum(local_mass[mask_ei_nit])/ei_mass[i,j]
                            
                            # Hyperinflated tissue
                            if np.count_nonzero(mask_ei_hit):
                                ei_hit[i,j] = np.sum(local_mass[mask_ei_hit])/ei_mass[i,j]
                            
                # within k-loop
                h_ee_nat += [ee_nat]
                h_ee_pat += [ee_pat]
                h_ee_nit += [ee_nit]
                h_ee_hit += [ee_hit]
                h_ee_mass += [ee_mass]
                h_ei_nat += [ei_nat]
                h_ei_pat += [ei_pat]
                h_ei_nit += [ei_nit]
                h_ei_hit += [ei_hit]
                h_ei_mass += [ei_mass]
            
                
            # k-loop end
            
            # Load every slice
            M.load_slice(m.frame, 'ee_NAT', h_ee_nat, m.matrix_validity)
            M.load_slice(m.frame, 'ee_PAT', h_ee_pat, m.matrix_validity)
            M.load_slice(m.frame, 'ee_NIT', h_ee_nit, m.matrix_validity)
            M.load_slice(m.frame, 'ee_HIT', h_ee_hit, m.matrix_validity)
            M.load_slice(m.frame, 'ee_MASS', h_ee_mass, m.matrix_validity)

            M.load_slice(m.frame, 'ei_NAT', h_ei_nat, m.matrix_validity)
            M.load_slice(m.frame, 'ei_PAT', h_ei_pat, m.matrix_validity)
            M.load_slice(m.frame, 'ei_NIT', h_ei_nit, m.matrix_validity)
            M.load_slice(m.frame, 'ei_HIT', h_ei_hit, m.matrix_validity)
            M.load_slice(m.frame, 'ei_MASS', h_ei_mass, m.matrix_validity)
#******************************************************************************

            shi = m.matrix_compute_shi()
            M.load_slice(m.frame, 'SHI', shi, m.matrix_validity)
            b.load(M)
            print("done")

    b.kmeans_clusters = 3
    
    d = b.defineClusters(include_ki=include_ki, read_ki = include_ki,
                      banned_subjects = banned_subjects, set0=ts0, set1=ts1)
    b.cross_validity(banned_subjects=banned_subjects, set0=ts0, set1=ts1)
    
    for subject in subjlist:
        if subject in timesets[ts0]:
            
            state = timesets[ts0][subject]
            M = b.matrices[subject]
            d, v = M.returnSortedData('label', state)
            M.plot_label(state, use_global_validity = True, 
            colorlist = b.color_list.copy(), save = save,
            dst = dst + "SUBJ%i_LABELS"%subject)
            
    #        M.plot('VS',state, save = save, dst = dst+"SUBJ%i_VS_E"%subject, 
    #      use_global_validity = True)
            
    #        M.plot('Delta_Gas_Fraction',state, save = save, 
    #      vmin = 0.0, vmax = 0.5,
    #      dst = dst+"SUBJ%i_DGF_E"%subject, use_global_validity = True)
            
    #        M.plot('Intensity_Exp',state, save = save,
    #       dst = dst+"SUBJ%i_IEE_E"%subject, use_global_validity = True)
            
    #        M.plot('Intensity_Insp',state, save = save,
    #       dst = dst+"SUBJ%i_IIE_E"%subject, use_global_validity = True)
            
            
        if subject in timesets[ts1]:
            state = timesets[ts1][subject]
    #        M.plot('VS',state, save = save, dst = dst+"SUBJ%i_VS_L"%subject,
    #       use_global_validity = True)
            
    #        M.plot('Delta_Gas_Fraction',state, vmin = 0.0, vmax = 0.5,
    #      dst = dst+"SUBJ%i_DGF_L"%subject, use_global_validity = True,
    #      save = save)
            
    #        M.plot('Intensity_Exp',state, save = save, 
    #      dst = dst+"SUBJ%i_IEE_L"%subject, use_global_validity = True)
            
    #        M.plot('Intensity_Insp',state, save = save,
    #      dst = dst+"SUBJ%i_IIE_L"%subject, use_global_validity = True)
            


