# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:29:58 2023

@author: angus
"""

import numpy as np
import matplotlib.pyplot as plt

# Define a suitable function to test the algorithm
def f(x,y):
    return 0.5*(x-3)**2 + 4*(y-3)**2 + - 4**2



# Generate a mesh and sample the function
xs = np.arange(-20.0, 30.0, 0.5)
ys = xs.copy()
X,Y = np.meshgrid(xs,ys)
Z = f(X,Y)

# %%

def visualize(X,Y,Z,vs,dtxt=0.5, history=None):
    
    # Generate a figure to scout the space
    fig,ax = plt.subplots(figsize=(8,8))
    CS = ax.contour(X,Y,Z, levels=10, cmap='plasma')
    ax.clabel(CS,inline=True,fontsize=10)
    ax.set_title("Nelder-Mead algorithm playground")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    for e,v in enumerate(vs):
        ax.scatter(v[0],v[1], c="k") # dots
        #ax.text(v[0]+dtxt,v[1]+dtxt,"%i"%(n)) # numbers
        l0 = e;  # id0 
        l1 = e+1;# id1
        if l1==len(vs): l1 = 0 # id correction
        xl = [vs[l0][0],vs[l1][0]] # line x-coord
        yl = [vs[l0][1],vs[l1][1]] # line y-coord
        ax.plot(xl,yl,color="k",alpha=0.5) # line
        
    if not history is None:
        for vs in history["simplex"]:
            for e,v in enumerate(vs):
                ax.scatter(v[0],v[1], c="b",alpha=0.2) # dots
                #ax.text(v[0]+dtxt,v[1]+dtxt,"%i"%(n)) # numbers
                l0 = e;  # id0 
                l1 = e+1;# id1
                if l1==len(vs): l1 = 0 # id correction
                xl = [vs[l0][0],vs[l1][0]] # line x-coord
                yl = [vs[l0][1],vs[l1][1]] # line y-coord
                ax.plot(xl,yl,color="b",alpha=0.20) # line

def reflect(vs,wid):
    # Compute the reflected vertex of the worst point
    nv = np.zeros(2) # new vertex
    for e,v in enumerate(vs):
        if e == wid:
            nv -= v
        else:
            nv += v
    return nv

def extend(wv,rv,alpha=1.0):
    # Define the extension 'ext' as half the vector between the worst and the reflected points
    ext = (rv-wv)/2
    # Note that alpha allows to modify the length of the extension
    return rv + alpha*ext

def inner_contraction(wv, rv, beta=0.5):
    # Define the extension 'ext' as half the vector between the worst and the reflected points
    ext = (rv-wv)/2
    # Note that beta sends the new point the original simplex
    return rv - beta*ext

def outer_contraction(wv, rv, gamma=0.5):
    # Define the extension 'ext' as half the vector between the worst and the reflected points
    ext = (rv-wv)/2
    # Note that beta sends the new point the original simplex
    return rv + gamma*ext

def shrink(vs, sorter, delta=0.5):
    
    mid_ext = vs[sorter[0]] - vs[sorter[1]]
    new_mid = vs[sorter[1]] + delta*mid_ext
    
    wst_ext = vs[sorter[0]] - vs[sorter[2]]
    new_wst = vs[sorter[2]] + delta*wst_ext
    
    return new_mid, new_wst

def iterate(vs,zs):
    
    # Sort the points depending on they performance
    sorter = np.argsort(zs)
    
    # Retrieve the sorted ids for each vector
    bid = sorter[0] # best
    mid = sorter[1] # mid
    wid = sorter[2] # worst
    
    # Determine reflected vector  
    rv = reflect(vs,wid)
    
    # Compute the z-value of the new vertex
    rz = f(*rv)
    
    if rz < zs[bid]: # Extend
    
        # if the reflected point is better than the best point, extend
        # Determine extended point
        ev = extend(vs[wid],rv)
        # Compute the z-value for the extended point
        ez = f(*ev)
        
        if ez < rz:
            return (ev, ez), "Extend"
        else:
            return (rv, rz), "Reflect"
        
    elif rz < zs[mid]: 
        return (rv,rz), "Reflect"
        
    elif rz < zs[wid]:
        # if the reflected point is better than the worst point, but not better than the other
        # two, then contract
        icv = inner_contraction(vs[wid], rv)
        ocv = outer_contraction(vs[wid], rv)
        
        icz = f(*icv)
        ocz = f(*ocv)
        
        if icz < ocz: 
            return (icv,icz), "Inner-contraction"
        else:
            return (ocv,ocz), "Outer-contraction"
        
    else:
        # Shrinkage
        nmv, nwv = shrink(vs,sorter)
        nmz = f(*nmv); nwz = f(*nwv)
        return (nmv,nmz,nwv,nwz), "Shrink"


def process(vs,zs,out):
    
    if len(out) == 4:
        # implies shrinkage
        # Sort the points depending on they performance
        sorter = np.argsort(zs)
        bid = sorter[0] # best
        
        nvs = np.array([vs[bid], out[0],out[2]])
        nzs = [zs[bid],out[1],out[3]]
    
    elif len(out) == 2:
        
        sorter = np.argsort(zs)
        bid = sorter[0] # best
        mid = sorter[1]
        
        nvs = np.array([vs[bid], vs[mid],out[0]])
        nzs = [zs[bid],zs[mid],out[1]]
    else:
        raise Exception("Error at process routine")
        return None

    # sort the new points
    bid, mid, wid = np.argsort(nzs)
    
    area_err = np.abs(np.linalg.det(np.hstack([nvs,np.ones((3,1))])))
    funct_err = np.abs((nzs[bid]-nzs[wid])/(np.abs(nzs[bid])+1e-9))
    
    err = [area_err, funct_err]
    return nvs, nzs, err

# %% Initial scenario plot

# CONFIG
nmaxit = 100
tol = 1e-9

# Propose three vertex to start the algorithm
x0 = [-10.,-18.]
x1 = [2., -8.]
x2 = [-5., -7]

# Reshape as numpy array
vs = np.array([x0,x1,x2]) # dynamic array

# Evaluated vertex
zs = [f(*v) for v in vs]

minz = np.min(zs)
error = None

history = {"simplex":[vs.copy()],
           "actions":["Start"],
           "area_err":[],
           "func_err":[],
           }
it = 0
err = [1,1]

# %%

while (it<nmaxit) and (err[0]>tol and err[1]>tol):
    
    out, action = iterate(vs,zs)
    vs, zs, err = process(vs,zs,out)
    history["simplex"] += [vs.copy()]
    history["actions"] += [action]    
    history["area_err"] += [err[0]]    
    history["func_err"] += [err[1]]    
    
# %%        
visualize(X,Y,Z,vs,history=history)

# %%


fig,ax= plt.subplots(figsize=(8,4))
ax.plot(np.log10(history["area_err"]), label="Simplex Area Error")
ax.plot(np.log10(history["func_err"]), label="Evaluation Error")
ax.set_ylabel("Log10(Error)")
ax.set_xlabel("Iterations")
ax.set_title("Error tracking")
ax.legend()



# %%
