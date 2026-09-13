# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:20:41 2022

@author: angus
"""

import os
from dolfin import *    
import dolfin
import numpy as np
import modelfunctions as md
import supportfunctions as sf
import time
import AirwayManager as awm
from numpy.linalg import norm


    # %% Unpack arguments

def determineAveragePressures(p,u,dx,Nsd,verbose=False,alpha=1.0):
    
    '''
    Function to determine the average pressure in every reference subdomain.
    '''
    
    # Create a data structure to hold the pressure values
    subdomain_averaged_pressure = np.empty(Nsd)

    # Determine the average pressure in every subdomain
    for j in range(Nsd):
        # Arrange the operator being omputed
        pressure_kernel=p*md.JJ(u,p,alpha=alpha)*dx(j)
        volume_kernel= md.JJ(u,p,alpha=alpha)*dx(j)
        avg_pressure = dolfin.assemble(pressure_kernel)
        subdomain_deformed_volume = dolfin.assemble(volume_kernel)
        
        # Assign average value to the data structure
        subdomain_averaged_pressure[j] = avg_pressure/subdomain_deformed_volume
        
    return subdomain_averaged_pressure

def determineAveragePressures_legacy(p,dx,subdomain_volume, verbose=False):
    
    '''
    Function to determine the average pressure in every reference subdomain.
    '''
    
    # Create a data structure to hold the pressure values
    Nsd = len(subdomain_volume)
    subdomain_averaged_pressure = np.empty(Nsd)

    # Determine the average pressure in every subdomain
    for j in range(Nsd):
        # Arrange the operator being omputed
        pressure=p*dx(j)
        
        # Determine the value
        subdomain_pressure=dolfin.assemble(pressure)/subdomain_volume[j]
        if verbose: 
            print(" > Subdomain ID%i Averaged Pressure: %.3f kPa"%(j,subdomain_pressure)) 
            
        # Assign average value to the data structure
        subdomain_averaged_pressure[j] = subdomain_pressure
    return subdomain_averaged_pressure


        
# Solver
def execute_pcv_simulation(args, verbose=True):
    
    print(os.path.abspath("."))
    
    # Initialize timer
    tic = time.time()
    
    restart_from_last_checkpoint = args["restart_from_last_checkpoint"]
    save_checkpoints = args["save_checkpoints"]
    path_to_mesh = args["mesh_dir"]
    output_to = args["output_to"]
    ncheckpoints = args["ncheckpoints"]
    ninternaldivs = args["ninternaldivs"]
    K_stiffness = args["K_stiffness"]
    KK_exp = args["KK_exp"]
    KK_factor = args["KK_factor"]
    solver_type = args["solver_type"]
    solver_dict = args["solver_dict"]
    pressure_dict = args["pressure_dict"] # mm3 ; ml*10**6=ml
    cm = args["constitutive_model"] # 'ber','ma','yoshi','bir2019','rausch'
    c,beta = args["constitutive_parameters"]
    path_to_airway = args["path_to_airway"]
    tol_p_it = args["tol_p_it"]
    tol_v_it = args["tol_v_it"]
    max_nit = args["max_nit"]    
    time_config = args['time_config']
    save_vtk = args['save_vtk']

    # Temporal configuration for the code

    ncycles = time_config['ncycles']
    Tup = time_config['Tup']
    Tplat = time_config['Tplat']
    Tdown1 = time_config['Tdown1']
    Tdown2 = time_config['Tdown2']
    Tbuffer = time_config['Tbuffer']    
    
    # Pressure dict
    pmin = pressure_dict["pmin"]
    pmed = pressure_dict["pmed"]
    pmax = pressure_dict["pmax"]
    activate_pressure_function = pressure_dict["activate_pfunction"]
    pfunction = pressure_dict["pressure_function"]
    
    # Pedley config
    additional_resistances = args['additional_resistances']

    pedley_config = additional_resistances['pedley_config']
    pedley_activate = pedley_config['activate']
    pedley_tolerance = pedley_config['tolerance']
    pedley_gamma = pedley_config['gamma']
    pedley_nitmax = pedley_config['nitmax']

    # Isotropic prestrain
    alpha = args['isotropic_prestrain']

    # Flow regimes
    # A: Flow moves from zero to prescribed value. Lasts 0.001 s
    # B: Steady inflation. Lasts 0.999 s
    # C: Transition. Changes steady flow to zero flow. Lasts 0.001 s
    # D: Zero flow. Achieve plateau pressure. 0.25 s
    # E: Expiration begins. Rapid changes. 0.25 s * 0.2 = 0.05 s
    # F: Pseudo-steady expiration. Lasts long but is kind of regular. 1.75 s.

    snes_solver_parameters = sf.solver_configuration(solver_type, solver_dict)
    
    # Make sure variables are initialized and paths exist
    if save_checkpoints:
        chk = 0
    
    # %%
    print("Executing this simulation with:")
    print("Permeability = %.1f*10**%.1f"%(KK_factor, KK_exp))
    
    # %%
    
    # Create the structure associated to the airway tree
    tree = awm.Pipeline(path_to_airway, gamma=pedley_gamma)
    
    # Additional resistances management
    if not additional_resistances["downstream"] is None:
        dr = additional_resistances["downstream"]
        temp_tree = awm.initialize_tree_from_vtu(path_to_airway)
        terminals = temp_tree.distal_points
        reference_mesh = path_to_mesh+"mesh000000.vtu"
        _,_,volumes = awm.distribute_subdomains(reference_mesh, terminals)
        total_volume = np.sum(volumes)
        weights = volumes/total_volume
        downstream_resistances = {e:dr*weights[i] for i,e in enumerate(tree.distal_elem_ids)}
    else:
        downstream_resistances = None
    
    if not additional_resistances["upstream"] is None:
        upstream_resistance = additional_resistances["upstream"]
    else:
        upstream_resistance = None
    
    # Determine the number of subdomains
    Nsd = np.count_nonzero(tree.distal_elem_ids)

    #Load the file with mesh and the boundaries
    mesh = dolfin.Mesh()
    
    # Load tetraedral mesh
    hdf = dolfin.HDF5File(mesh.mpi_comm(), '%stetrahedral_mesh.h5'%path_to_mesh, "r")
    hdf.read(mesh, "/mesh", False)
    boundary_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    hdf.read(boundary_markers, "/boundary_markers")
    hdf.close()
    
    # Read Phi0, the initial porosity, and Omega_i, the subdomain.
    subdomain = MeshFunction("size_t", mesh, path_to_mesh+"/Omega.xml.gz")
    KK=dolfin.Constant(KK_factor*pow( 10., KK_exp))
    
    # %% FEniCS data structures and problem formulation
    
    #Define some dolfin parameters
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
    dolfin.parameters["allow_extrapolation"] = True 
    dolfin.parameters["form_compiler"]["optimize"] = True 

    # Get mesh dimension and prepare boundary measure
    gdim = mesh.geometry().dim()
    dx = dolfin.Measure("dx", domain=mesh, subdomain_data=subdomain)
    ds = dolfin.Measure("ds", domain=mesh, subdomain_data=boundary_markers)
    
    # Limit quadrature degree
    dx = dx(degree=3)
    ds = ds(degree=3)
    
    # Build function space. We use Taylor-Hood element
    V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)  ##
    P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dolfin.FiniteElement("CG", mesh.ufl_cell(), 1)
    TH = P2 * P1  #Taylor 
    W = dolfin.FunctionSpace(mesh, TH)  #creo el nuevo espacio de funciones (mixto) DESPLAZAMIENTO Y PRESION
    dolfin.info("Num DOFs {}".format(W.dim()))
    
    # Function space for
    G = FunctionSpace(mesh,"CG",1)
    g0= Constant(0.50)
    phi0 =interpolate(g0,G)
    
    #Define an expression to time step
    dt = dolfin.Expression(("beta"), beta=0., degree=2, domain=mesh)
        
    # Unknowns, values at previous step and test functions
    w = dolfin.Function(W)
    u, p = dolfin.split(w)
    
    w0 = dolfin.Function(W)
    u0, p0 = dolfin.split(w0)
    
    _u, _p = dolfin.TestFunctions(W)
    du = dolfin.TrialFunction(W)            # Incremental displacement , sirve para Jacobian solamente
    I = dolfin.Identity(W.mesh().geometry().dim())
        
    if cm=='ber':
        P=md.stress_ber(u,p)
    elif cm=='ma':
        P=md.stress_ma(u,p)
    elif cm=='bir2019':
        P=md.stress_bir2019(u,p,c=c,beta=beta,alpha=alpha)   
    elif cm=='yoshi':
        P=md.stress_yoshi(u,p) 
    elif cm=='rausch':
        P=md.stress_rausch(u,p) 
            
    # Eqn (37)'s first term; Gravity term omitted.
    # P: First Piola Kirchoff stress tensor
    # _u: Trial function for displacement field
    # dx: Volume measure
    F1=dolfin.inner(P, dolfin.grad(_u) )*dolfin.dx 
        
    # *** Where does this term stem from? *** dPhi/dt
    # Jacobian(u) * Trace(grad(u)-grad(u0))*Finv(u)
    # JJ computes jacobian from displacement field u, p is not used.
    # u0: Obtained from a split applied to w0, a function from the W mixed 
    #     function space.
    # Finv: The inverse from the deformation gradient tensor computed from
    #       'u', p is not used.
    F2aux1=md.JJ(u,p,alpha=alpha)*tr((grad(u)-grad(u0))*md.Finv(u,p,alpha=alpha)) 
        
    # KK: Permeability as a constant
    # This computes a term similar to 'Q' as defined in eqn. (21), while
    # disregarding gravity effects
    F2aux2=md.F2aux_mass(u,p,KK,alpha=alpha)
    
    class src_obj(UserExpression):
        
        def __init__(self, subdomains,sources, **kwargs):
            super().__init__(**kwargs)
            self.subdomains = subdomains
            self.sources = sources
        
        def eval_cell(self, values, x, cell):
            values[0] = self.sources[self.subdomains[cell.index]]
                
        def update(self, sources):
            self.sources=sources
            
        def values_shape(self):
            return (1,)
    
    # Initialize source objects
    sources = {i:0.0 for i in np.arange(Nsd)}
    src = src_obj(subdomain, sources) 
    
    F2aux3 = src*(md.JJ(u,p,alpha=alpha)-1+phi0)# ORIGINAL

    F2= (dolfin.inner(F2aux1, _p))*dolfin.dx + \
        dt*(dolfin.inner(grad(_p),F2aux2))*dolfin.dx + \
        dt*dolfin.inner(_p,F2aux3)*dolfin.dx    
    
    # Compute spring force acting upon the lung's surface
    F3=dolfin.inner(u,_u)*K_stiffness*dolfin.ds(subdomain_data=boundary_markers,
                                             subdomain_id=2)
        
    # Functional to be optimized
    R=F1+F2+F3
        
    # Derivative to the functional
    Jac = dolfin.derivative(R, w, du)
    
    # Initialize solver
    problem = dolfin.NonlinearVariationalProblem(R, w, J=Jac,
                                          form_compiler_parameters={'optimize':True})
    
    solver = dolfin.NonlinearVariationalSolver(problem)
    
    # First solver parameter definition
    solver.parameters.update(snes_solver_parameters)
    solver.solve()

    # Extract solution components
    u, p = w.split()
    u.rename("u", "displacement")
    p.rename("p", "pressure")

    #Creamos listas con los instantes de tiempo, pasos de tiempo y flujos. 
    times,ps,dts, checkpoints = md.times_and_pressures(ncycles,pmed,pmax,Tup,Tplat,
                                                       Tdown1,Tdown2,Tbuffer=Tbuffer,
                                                       pmin=pmin,                                                       
                                                       ncheckpoints=ncheckpoints, 
                                                       ninternaldivs=ninternaldivs)  
    
    Q = 0.0 # dummy value

    # Time-stepping loop
    t = 0
    iterativetime=[]
    
    # Branched initialization depending on whether is a new simulation or a restart
    if restart_from_last_checkpoint:
        last_checkpoint_id = sf.retrieve_last_checkpoint(output_to)
    else:
        Jacob=[]
        fluxes=[]
        presionestodas=[]
        effectivetimes=[]

    # Time loop
    for i in np.arange(len(times)):
        
        # Read prescribed fluxes and current simulation time
        aw_pressure = ps[i]
        t=times[i]
        dt.beta=dts[i]
        
        print(" > Pressure: %.1f (cmH2O)"%(aw_pressure*10.19))
        
        if activate_pressure_function:
            aw_pressure = pfunction(t)

# %% Start of the restart mechanism

        # Restarting mechanism
        # Note that we are not entering the iterative cycles while checking this.
        if restart_from_last_checkpoint:
            
            # Find a 't' really close to the t from the checkpoint
            # TODO: Check if this is numerical precision gets us in trouble
            time_difference = np.abs(t-checkpoints[last_checkpoint_id])
            
            if not time_difference<1e-8:
                # Skip those t's not within tolerance
                continue
            
            elif t>checkpoints[last_checkpoint_id]+0.01: # arbitrary delta
                raise Exception("Exceeded last checkpoint t. Stopping Code")
                break
            else:
                
                print("Found a matching time for resuming: t=%.6f"%t)
                
                # Initialize a function space corresponding to the splitted funct.
                Uchk = u.function_space().collapse()
                Pchk = p.function_space().collapse()
                
                # Initialize new functions 
                uchk = Function(Uchk)
                pchk = Function(Pchk)
                
                # Turn off the restart flag
                restart_from_last_checkpoint = False
                
                # Generate
                checkname = "%3.3i.xdmf"%(last_checkpoint_id+1)
                print("Loading data from checkpoint id: %i"%(last_checkpoint_id+1))
                
                with XDMFFile(MPI.comm_world, output_to+"Checkpoints/u/"+checkname) as infile:
                    infile.read_checkpoint(uchk, "u",0)
                    infile.close()
                    
                with XDMFFile(MPI.comm_world, output_to+"Checkpoints/p/"+checkname) as infile:
                    infile.read_checkpoint(pchk, "p",0)
                    infile.close()
                
                # Update checkpoint counter
                chk = last_checkpoint_id+1 #### TODO: modified!!!! borrar el +1 
                # Pass the loaded information into the relevant structures
                assigner = FunctionAssigner(W,[Uchk,Pchk])
                assigner.assign(w0,[uchk,pchk])
                
                # Load the corresponding signals
                signals_path = output_to+"Checkpoints/Signals/%i/"%chk 
                if not os.path.isdir(signals_path):
                    raise Exception("Missing the signals corresponding to checkpoint %i"%chk)
                    break
                else:
                    Jacob = list(np.load(signals_path+"volumenes.npy"))
                    presionestodas = list(np.load(signals_path+"presionestodas.npy"))
                    fluxes = list(np.load(signals_path+"fluxes.npy"))
                    effectivetimes = list(np.load(signals_path+"effectivetimes.npy"))
                    
                # Initialize arrays that should be created from a previous step in the iterative cycle
                # Determine average pressures from initial conditions (0)
                current_it_subdomain_pressure = determineAveragePressures(p0,u0,dx,Nsd,alpha=alpha)
                            
                # Change format for compatibility with the tree system
                Pdistal = {tree.translate["s2p"][s]:current_it_subdomain_pressure[s] for s in range(Nsd)}
                            
                # Assemble linear system using the average pressures
                tree.assemble_linear_system(Pdistal=Pdistal, Q0=Q,
                                            upstream_resistance=upstream_resistance, 
                                            downstream_resistances=downstream_resistances)
                        
                # Initialize the subdomain deformed volume array using the reference volumes
                current_it_subdomain_defomed_volume = np.empty(Nsd,dtype=float)
                for j in range(Nsd):
                    dummy = (md.JJ(u0,p0,alpha=alpha)-1+phi0)*dx(j)
                    current_it_subdomain_defomed_volume[j] = dolfin.assemble(dummy)
                    
                continue        
        
# %% End of the restarting mechanism

        # Save the time when this cycle started
        iterativetime += [time.time()]
        
        if i==0: 
            # Initialize the arrays
            
            # Determine average pressures from initial conditions (0)
            current_it_subdomain_pressure = determineAveragePressures(p,u,dx,Nsd,alpha=alpha)
                        
            # Change format for compatibility with the tree system
            Pdistal = {tree.translate["s2p"][s]:current_it_subdomain_pressure[s] for s in range(Nsd)}
                        
            # Assemble linear system using the average pressures
            tree.assemble_linear_system(Pdistal=Pdistal, Q0=Q,
                                        upstream_resistance=upstream_resistance, 
                                        downstream_resistances=downstream_resistances)
                    
            # Initialize the subdomain deformed volume array using the reference volumes
            current_it_subdomain_defomed_volume = np.empty(Nsd,dtype=float)
            for j in range(Nsd):
                dummy = (md.JJ(u,p,alpha=alpha)-1+phi0)*dx(j)
                current_it_subdomain_defomed_volume[j] = dolfin.assemble(dummy)
                            
        else:
            # Report the time employed in solving the previous iterative cycle
            if len(iterativetime)>2:
                print("time/it: %i (s)"%((iterativetime[-1]-iterativetime[-2])))
        
        print("Time: %.3f [s]"%t)
        print("Step volume: %.3f [mL]"%(-Q*dts[i]*1e-3))

        # Iterative loop for convergence in airway tree and lung information
        for ii in range(max_nit):
            
            if ii == 0 and verbose:
                print("\n"*4)
                print("***********************************")
                print("***** Starting iterative loop *****")
                print("***********************************")
            
                            
            # Backup the previously converged subdomain averaged pressure and deformed volume to be used later when 
            # computing a norm assocciated to the convergence between iterative cycles of this value.
            prev_it_subdomain_pressure = current_it_subdomain_pressure.copy()
            prev_it_subdomain_defomed_volume = current_it_subdomain_defomed_volume.copy()
            
            # Update the linear system with the Pdistal which is defined at the end of an iterative loop with the 
            # previous step subdomain-averaged values.
            tree.update_linear_system(Pdistal, Q0=None, P0=aw_pressure)
            
            # Solve the linear system associated to the flow in the airway tree
            tree.solve_linear_system()
            
            if pedley_activate:
                ped_cnt = 0 # Initialize counter
                x_old = tree.x.copy()
                err = 999 # Initialize error in a high value
                while err > pedley_tolerance:
                    # Update the Pedley resistances
                    tree.update_pedley_resistances()
                    # Solve the linear system
                    tree.solve_linear_system()
                    # Determine a the error
                    err = norm(x_old-tree.x)/norm(tree.x)
                    # Update counter and save current error as x_old
                    ped_cnt += 1 
                    x_old = tree.x.copy()
                    
                    if ped_cnt>pedley_nitmax:
                        raise Exception("Divergence at Pedley's iterations")

            # Retrieve the new fluxes that are going to become input for the poromechanical system
            subdomain_fluxes = tree.retrieve_qs()
            
            # Update the source values according to subdomain air volume and the recently computed fluxes.
            sources = {j:subdomain_fluxes[j]/prev_it_subdomain_defomed_volume[j] for j in range(Nsd)}
            src.update(sources)
            
            # Solve the poromechanical model
            solver.solve()
             
            # Determine the new subdomain averaged pressures and store them
            # in current_subd_pressure.
            current_it_subdomain_pressure = determineAveragePressures(p,u,dx,Nsd,alpha=alpha)

            # Determine the current deformed volume of the lung
            current_it_subdomain_defomed_volume = np.empty(Nsd,dtype=float)
            for j in range(Nsd):
                dummy = (md.JJ(u,p,alpha=alpha)-1+phi0)*dx(j)
                current_it_subdomain_defomed_volume[j] = dolfin.assemble(dummy)
                            
            # Compute a norm for the difference between the previous iteration 
            # and the current subdomain pressures.
            p_diff = np.linalg.norm(prev_it_subdomain_pressure - current_it_subdomain_pressure)
            
            # Determine the difference between the difference between iterations
            v_diff = np.linalg.norm(current_it_subdomain_defomed_volume - prev_it_subdomain_defomed_volume)

            # Change format for compatibility with the tree system (to be used in the next iteration). 
            Pdistal = {tree.translate["s2p"][s]:current_it_subdomain_pressure[s] for s in range(Nsd)}            
            
            
            # Report the difference
            print("Pressure difference: ",p_diff)
            print("Volume difference: ", v_diff)
            
            print("\n*************** ITERATIVE STEP %i OVER ***************\n"%ii)
            
            
            convergence_criteria = p_diff < tol_p_it
            
            # If the difference is below the tolerance, save the results and
            # move into the new step
          #  if p_diff < tol_p_it and v_diff < tol_v_it: ORIGINAL
            if convergence_criteria:
                
                # Compute mean Jacobian    
                vol_=dolfin.assemble((det(I+grad(u)))*dx)
                volL=vol_/(10**6)
                
                deltavol=dolfin.assemble(((md.JJ(u,p,alpha=alpha)-md.JJ(u0,p0,alpha=alpha))/dt)*dx)
                
                # Append the new quantities
                presionestodas.append(tree.x[tree.N])
                Jacob.append(volL)
                fluxes.append(deltavol/10**6)
                effectivetimes.append(t)    
                
                tree.export_solution(output_to+"Airways/tree%6.6i.vtu"%i,compute_re=True)

                
                md.write_signals(output_to+"Signals",times,fluxes,
                                    presionestodas, Jacob,effectivetimes,i,
                                    iterativetime)
                                
                if save_vtk: md.write_vtk(mesh,output_to+"VTK","bir2019",I,u,p,dx,i,t,c,beta,alpha=alpha)
                
                # Assign the converged current result to the 'previous' step 
                # variable so that the next round of computations is OK.
                w0.assign(w)

                # Write checkpoint
                if save_checkpoints:
                    # Check if "t" activates any checkpoint
                    if np.any(np.abs(checkpoints-t)<1e-10):
                        
                        # If so, update the checkpoint tracker
                        chk += 1
                        print("Checkpoint at t = %f"%t)
                        print("Chk value: ", chk)
                        
                        # Split the variables in a deepcopy mode
                        u_, p_ = w.split(deepcopy=True)

                        # Save into an unique XDMF file
                        # TODO: Should we implement a single file checkpoint?
                        with XDMFFile(MPI.comm_world, output_to+"Checkpoints/u/%3.3i.xdmf"%chk) as outfile:
                            outfile.parameters["flush_output"] = True 
                            outfile.parameters['rewrite_function_mesh'] = True
                            outfile.write_checkpoint(u_,"u",0,XDMFFile.Encoding.HDF5,False)
                            outfile.close()
                            
                        with XDMFFile(MPI.comm_world, output_to+"Checkpoints/p/%3.3i.xdmf"%chk) as outfile:
                            outfile.parameters["flush_output"] = True 
                            outfile.parameters['rewrite_function_mesh'] = True
                            outfile.write_checkpoint(p_,"p",0,XDMFFile.Encoding.HDF5,False)
                            outfile.close()
                        
                        md.write_signals(output_to+"Checkpoints/Signals/%i"%chk,times,fluxes,
                                            presionestodas, Jacob,effectivetimes,i,
                                            iterativetime)

                
                break
            
            elif not convergence_criteria and ii == (max_nit-1):
                raise Exception("Reached maximum number of iterations without convergence.")
    
    
    toc = time.time()
        
    overall_time = toc-tic
    
    print("Overall time: %.0f"%(overall_time))
    

if __name__ =="__main__":
    

    # Flow regimes
    # A: Flow moves from zero to prescribed value. Lasts 0.001 s
    # B: Steady inflation. Lasts 0.999 s
    # C: Transition. Changes steady flow to zero flow. Lasts 0.001 s
    # D: Zero flow. Achieve plateau pressure. 0.25 s
    # E: Expiration begins. Rapid changes. 0.25 s
    # F: Pseudo-steady expiration. Lasts long but is kind of regular. 1.75 s.
    
    # Simulation Codename
    codename = "PIG5-APRV-quick" # This is a name for the folder where the output is directed
    mesh_name = "" #  This should change for different states/subjects; While in development, just keep 'stable'
    case ="FEniCS" # This is the specific name for the mesh in use
    
    # Declare the path to the folder
    path_to_mesh = "/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG5/ARDSnet/MESH/"
    path_to_airway = path_to_mesh+"skel.vtu"
    
    # Direct the output of this execution towards this folder
    output_to = "/mnt/c/Users/angus/OneDrive - Universidad Católica de Chile/Documentos/ards-lung-simulator/"
    # Path to signals
    signal_path = "/mnt/c/Users/angus/Downloads/CORNELL-NEWGEO/PIG6/APRV/PIG6-APRV.mat"

    # Checkpoint parameters
    restart_from_last_checkpoint = False
    save_checkpoints = True
    save_vtk=False
    
    # Processing the paths
    path_to_mesh =  sf.manage_mesh_directory(path_to_mesh,mesh_name,case)
    output_to = sf.manage_output_directory(output_to,codename,restart_from_last_checkpoint)
      
    # Quick parameter setting
    K_stiffness = 0.0146
    
    # Isotropic prestrain
    isotropic_prestrain = 1.00
    
    # Temporal management
    ncheckpoints = [2,4,4,2,2]    # These will be used to save some checkpoints
    ninternaldivs= [5,10,30,10,10] # Intervals between checkpoints divisions
    
    #### only relevant if block_variable_permeability == True
    KK_exp = 5
    KK_factor = 1.0

    # Tolerances in the iterative cycles
    max_nit = 50
    tol_p_it = 1e-3
    tol_v_it = 1e-2

    # Selecting solver
    solver_type = "dict"
    solver_dict = {"nonlinear_solver":"snes",
                   "snes_solver":{"linear_solver":"mumps",
                                  "relative_tolerance":1e-6,
                                  "absolute_tolerance":1e-8,
                                  "maximum_iterations":60,
                                  "line_search":"bt",
                                  "report":True,
                                  "error_on_nonconvergence":True,
                                  "preconditioner":"default"}
                                  }
    
    # Time configuration for the pressure-controlled ventilation
    ncycles = 2
    Tup = 0.08 # Stepping into Phigh
    Tplat = 1.41 # At Phigh
    Tdown1 = 0.11 # From Phigh to Pmed
    Tdown2 = 0.36 # From Pmed to Plow
    Tbuffer = 0.10 # First steps in Plateau with a different time step
    
    time_config = {"ncycles":ncycles,
                   "Tup":Tup,
                   "Tplat":Tplat,
                   "Tdown1":Tdown1,
                   "Tdown2":Tdown2,
                   "Tbuffer":Tbuffer,}

    
    # Pressure config
    peep = 1.46
    pmin = 0.0+peep; pmax = 17.73+peep; # in cmH2O
    pmed = 1.24+peep
    pressure_dict = {"pmin":pmin/10.1972,
                     "pmed":pmed/10.1972,
                     "pmax":pmax/10.1972,
                     "activate_pfunction":False,
                     "pressure_function":None}

    # Constitutive model; 'ber','ma','yoshi','bir2019','rausch'
    cm = "bir2019"
    c = 2.45 # [factor is 3.25 for VT30]
    beta = 1.075
    
    additional_resistances = {"upstream":None,
                              "downstream":None,
                              "pedley_config":{"activate":True,
                                               "gamma":0.327*0.5,
                                               "tolerance":1e-8,
                                               "nitmax":100}}
    


    args = {"restart_from_last_checkpoint":restart_from_last_checkpoint,
            "save_checkpoints":save_checkpoints,
            "mesh_dir":path_to_mesh,
            "output_to":output_to,
            "ncheckpoints":ncheckpoints,
            "ninternaldivs":ninternaldivs,
            "K_stiffness":K_stiffness,
            "isotropic_prestrain":isotropic_prestrain,
            "KK_exp":KK_exp,
            "KK_factor":KK_factor,
            "solver_type":solver_type,
            "solver_dict":solver_dict,
            "pressure_dict":pressure_dict, # conversion from L to mm3
            "time_config":time_config,
            "constitutive_model":cm,
            "constitutive_parameters":(c,beta),
            "path_to_airway":path_to_airway,
            "additional_resistances":additional_resistances,
            "tol_p_it":tol_p_it,
            "tol_v_it":tol_v_it,
            "max_nit":max_nit,
            "save_vtk":save_vtk,

            }
    
    execute_pcv_simulation(args)
    