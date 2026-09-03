# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:20:41 2022

@author: angus

Last updated: 10-11-2025
"""

import os
from dolfin import *    
import dolfin
import numpy as np
import time
import AirwayManager as awm
import modelfunctions as md
import supportfunctions as sf
from numpy.linalg import norm

def determineAveragePressures(p,u,dx,Nsd,verbose=False,Fprev=None):
    
    '''
    Function to determine the average pressure in every reference subdomain.
    '''
    
    # Create a data structure to hold the pressure values
    subdomain_averaged_pressure = np.empty(Nsd)
    # Jacobian
    J = md.JJ(u,p,Fprev=Fprev)

    # Determine the average pressure in every subdomain
    for j in range(Nsd):
        pressure_kernel=p*J*dx(j) # Pressure in the current domain operator
        volume_kernel= J *dx(j) # Volume in the current domain operator
        avg_pressure = dolfin.assemble(pressure_kernel) # Volume-averaged pressure
        subdomain_deformed_volume = dolfin.assemble(volume_kernel) # Current domain volume
        
        # Assign average value to the data structure
        subdomain_averaged_pressure[j] = avg_pressure/subdomain_deformed_volume # Averaged pressure
        
    return subdomain_averaged_pressure # Volume-averaged pressure

def determineTreeInletConditions(t, Q, Tinsp, duration_step, presfn=None, 
                                 flowfn=None, t_tol=1e-4, pplat=None,peep=0.0,
                                 block="NULL"):
    '''
    Set the values for Q0 and P0, flow and pressure at the inlet of the airway
    tree according to the current time in the simulation.
    
    '''
    # Set the inlet variables for the airway tree
    if t<=Tinsp: # 1st Inspiration + Inspiratory pause 
        if flowfn is None:
            Q0 = Q; P0 = None
        else:
            Q0 = flowfn(t); P0=None
        
        block="A"
    elif Tinsp<t<=duration_step: # 1st Expiration
        if presfn is None:    
            Q0 = None; P0 = peep
        else:
            Q0 = None;
            try:
                P0 = min([presfn(t,pplat),pplat])
            except:
                P0 = presfn(t,pplat)
                print("Failed to determine a mininum pressure")
        block="B"
                
    elif duration_step<t<duration_step+Tinsp: # 2nd Inspiration
        if flowfn is None:
            Q0 = Q; P0 = None
        else:
            Q0 = flowfn(t); P0 = None
        block="C"
            
    else: # 2nd Expiration
        if presfn is None:    
            Q0 = None; P0 = peep
        else:
            Q0 = None
            try:
                P0 = min([presfn(t,pplat),pplat])
            except:
                P0 = presfn(t,pplat)
                print("Failed to determine a mininum pressure")
        block="D"

    if (Q0 is None) and (P0 is None):
        print("Warning: Both Q0 and P0 are None and this will cause a crash")
        print("Block where the condition was prescribed: Block '%s'"%block)        
                
    return Q0, P0
        
# Solver
def execute_vcv_simulation(args, verbose=True):
    
    print(os.path.abspath("."))
    
    # Initialize timer
    tic = time.time()
    
    restart_from_last_checkpoint = args["restart_from_last_checkpoint"]
    save_checkpoints = args["save_checkpoints"]
    path_to_mesh = args["mesh_dir"]
    output_to = args["output_to"]
    ncheckpoints = args["ncheckpoints"]
    ninternaldivs = args["ninternaldivs"]
    
    K_cw_stiffness = args["K_cw_stiffness"]
    K_d_stiffness = args["K_d_stiffness"]
    K_m_stiffness = args["K_m_stiffness"]
    
    inspiratory_pause_stop = args["inspiratory_pause_stop"]

    solver_type = args["solver_type"]
    solver_dict = args["solver_dict"]
    tidal_volume = args["tidal_volume"] # mm3 ; ml*10**6=ml
    
    # Stiffness related values
    cm_config = args['cm_config']
    cm = cm_config["model_name"] # 'ma','bir2019'
    c_tissue = cm_config['c_tissue']
    c3 = cm_config['c3']
    k = cm_config['k']
    phi_t = cm_config['phi_transition']    
    stiffness_function = cm_config['stiffness_function']
    variable_stiffness = cm_config['variable_stiffness']
    stiffness_file = cm_config['stiffness_file']
    
    path_to_airway = args["path_to_airway"]
    tol_p_it = args["tol_p_it"]
    tol_v_it = args["tol_v_it"]
    max_nit = args["max_nit"]    
    time_config = args['time_config']
    porosity_config = args['porosity_config']
    save_vtk = args['save_vtk']
    additional_resistances = args['additional_resistances']
    
    # Boundary condition functions
    bc_dict = args['bc_dict']
    use_bc_functions = bc_dict['activate']
    pplat = None
    if use_bc_functions:
        flowfn = bc_dict['flow_func']
        presfn = bc_dict['pres_func']
    else:
        flowfn=None; presfn=None
        
    # Alveolar pressure enters in cmH2O and is transformed here to kPa
    p_alv = args['alveolar_pressure'] # Enters in cmH2O
    if not (p_alv is None):
        p_alv = p_alv/10.1972

    # Permeability configuration
    perm_config = args['permeability_dict']
    variable_permeability = perm_config['variable_permeability']
    perm_file = perm_config['permeability_file']
    KK_exp = perm_config['KK_exp']
    KK_factor = perm_config['KK_factor']

    # Temporal configuration for the code
    ncycles = time_config['ncycles']
    Tsyr = time_config['Tsyr']
    Tpausa = time_config['Tpausa']
    Texp = time_config['Texp']
    Tinsp = Tsyr+Tpausa
    duration_step=Tinsp+Texp
    
    # Pedley config
    pedley_config = additional_resistances['pedley_config']
    pedley_activate = pedley_config['activate']
    pedley_tolerance = pedley_config['tolerance']
    pedley_gamma_exp = pedley_config['expiratory_gamma']
    pedley_gamma_insp = pedley_config['inspiratory_gamma']
    pedley_nitmax = pedley_config['nitmax']

    # Porosity config
    variable_porosity = porosity_config['activate']
    mean_porosity = porosity_config['mean']
    porosity_file = porosity_config['file']
    
    # Gravity
    gravity_config = args['gravity_config']
    gravity_activate = gravity_config['activate']
    g = gravity_config['g'] # mm/s^2 #Gravitational acceleration
    gvec = Constant((0.0,g,0.0))
    rho_t = gravity_config['rho_t'] # kg/mm3  Water density
    rho_a = gravity_config['rho_a'] #  kg/mm3  Air density
    K_inf = gravity_config['K_inf'] # Very stiff spring
    
    # Inverse displacement analysis configuration
    ida_config = args['ida_config']
    
    # PEEP variable initialization (not used)
    peep = 0.0

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

    # Create the structure associated to the airway tree
    tree = awm.Pipeline(path_to_airway, gamma=pedley_gamma_insp)
    
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
    
    # Load subdomains
    subdomain = MeshFunction("size_t", mesh, path_to_mesh+"/Omega.xml.gz")
    
    # Define permeability
    print("Executing this simulation with:")
    
    # Manage variables that may vary in space, defined in .xml.gz files elsewhere. First
    # generate dictionaries and then pass them info the support function.
    stiffness_config = {'read_from_file':variable_stiffness,
                        'path_to_file':output_to+"/"+stiffness_file,
                        'property_name':"Tissue stiffness",
                        'property_value':c_tissue,
                       # 'stiffness_function':stiffness_function,
                        'k_value':k,
                        'phi_t':phi_t}
    
    porosity_config = {'read_from_file':variable_porosity,
                       'path_to_file':path_to_mesh+"/"+porosity_file,
                       'property_name':"End-expiratory porosity",
                       'property_value':mean_porosity}
    
    permeability_config = {'read_from_file':variable_permeability,
                           'path_to_file':path_to_mesh+"/k0.xml.gz",
                           'property_name':"Permeability",
                           'property_value':KK_factor*10**KK_exp}

    if variable_stiffness:
        sf.generate_stiffness_file(mesh, stiffness_config, porosity_config)
        
    c_tissue = sf.define_property(mesh, stiffness_config)
    KK = sf.define_property(mesh, permeability_config)
    phi = sf.define_property(mesh, porosity_config)
        
    
    # %% FEniCS data structures and problem formulation
    
    # Sort the material properties
    material_properties = {"constitutive_model":cm,
                           "tissue_stiffness":c_tissue,
                           "tissue_alt_parameter":c3,
                           "permeability":KK,
                           "phi":phi,
                           'rho_a':rho_a,
                           'rho_t':rho_t,
                           'gravity_vector':gvec,
                           'K_inf':K_inf,
                           'p_alv':p_alv}
    
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
     
    #Define an expression to time step
    dt = dolfin.Expression(("beta"), beta=0., degree=2, domain=mesh)
    
# %% Newly introduced code for inverse problem

    Fprev = sf.inverse_deformation_analysis(mesh, boundary_markers, material_properties, ida_config, output_to)

    # %% Regular code for time integration

    # Initialize sources with zero values
    sources = {i:0.0 for i in np.arange(Nsd)}   
    # Build object
    src = sf.build_source_term(subdomain, sources)

    # Unknowns, values at previous step and test functions
    w = dolfin.Function(W)
    u, p = dolfin.split(w)
    
    w0 = dolfin.Function(W)
    u0, p0 = dolfin.split(w0)
    
    _u, _p = dolfin.TestFunctions(W)
    du = dolfin.TrialFunction(W)    

    I = dolfin.Identity(W.mesh().geometry().dim())
        
    if cm=='ma':
        P=md.stress_ma(u,p,c=c_tissue,Fprev=Fprev)
    elif cm=='bir2019':
        P=md.stress_bir2019(u,p,c=c_tissue,c3=c3,Fprev=Fprev)   
    elif cm=='perez':
        P=md.stress_perez(u,p,c=c_tissue,c3=c3,Fprev=Fprev)   
        
    # Phi0 is Phi at end-expiration
    phi0 = phi
            
    # Eqn (37)'s first term; Gravity term omitted.
    # P: First Piola Kirchoff stress tensor
    # _u: Trial function for displacement field
    # dx: Volume measure
    F1_base=dolfin.inner(P, dolfin.grad(_u) )*dolfin.dx 
    
    if gravity_activate:
        F1_grav = dolfin.inner((md.JJ(u,p,Fprev=None)*rho_a + (1 - phi0)*(rho_t-rho_a))*gvec,_u)*dolfin.dx
        F1 = F1_base-F1_grav
    else:
        F1 = F1_base
        
    # *** Where does this term stem from? *** dPhi/dt
    # Jacobian(u) * Trace(grad(u)-grad(u0))*Finv(u)
    # JJ computes jacobian from displacement field u, p is not used.
    # u0: Obtained from a split applied to w0, a function from the W mixed 
    #     function space.
    # Finv: The inverse from the deformation gradient tensor computed from
    #       'u', p is not used.
    F2aux1=md.JJ(u,p,Fprev=None)*tr((grad(u)-grad(u0))*md.Finv(u,p,Fprev=None)) 
        
    # KK: Permeability as a constant
    # This computes a term similar to 'Q' as defined in eqn. (21), while
    # disregarding gravity effects
    F2aux2=md.F2aux_mass(u,p,KK,gravity_activate=gravity_activate,rho_a=rho_a,gvec=gvec,Fprev=None)

    F2aux3 = src*(md.JJ(u,p,Fprev=None)-1+phi0) # Right term is the Lagrangian porosity

    F2= (dolfin.inner(F2aux1, _p))*dolfin.dx + \
        dt*(dolfin.inner(grad(_p),F2aux2))*dolfin.dx + \
        dt*dolfin.inner(_p,F2aux3)*dolfin.dx    
    
    # Compute spring force acting upon the lung's surface
    F3=dolfin.inner(u,_u)*K_cw_stiffness*dolfin.ds(subdomain_data=boundary_markers,
                                             subdomain_id=2) # Chest wall
    
    # Make the diaphragm much less stiffer than the thoracic cage
    F4=dolfin.inner(u,_u)*K_d_stiffness*dolfin.ds(subdomain_data=boundary_markers,
                                             subdomain_id=1) # Diaphragm

    if gravity_activate:
        # Activate the gravitational effects
        F5=dolfin.inner(u,_u)*K_inf*dolfin.ds(subdomain_data=boundary_markers,
                                              subdomain_id=3) # Dorsal  
    else:            
        # Instead of a stiff dorsal region, use the conventional 
        F5=dolfin.inner(u,_u)*K_cw_stiffness*dolfin.ds(subdomain_data=boundary_markers,
                                              subdomain_id=3) # Dorsal  

    # Make the diaphragm much less stiffer than the thoracic cage
    F6=dolfin.inner(u,_u)*K_m_stiffness*dolfin.ds(subdomain_data=boundary_markers,
                                             subdomain_id=4) # Mediastinum

    R=F1+F2+F3+F4+F5+F6

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
    times,qs,dts, checkpoints = md.times_and_fluxes(ncycles,tidal_volume,1.0,
                                                    Tsyr,Texp,Tpausa,Tinsp, 
                                                    ncheckpoints=ncheckpoints, 
                                                    ninternaldivs=ninternaldivs)  

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
        prescribedfluxes=[]

    # Time loop
    for i in np.arange(len(times)):
        
        # Read prescribed fluxes and current simulation time
        t=times[i]
        dt.beta=dts[i]
        # Overrided if use_bc_functions is True (see) determineTreeInletConditions
        Q = qs[i]
        
        # We determine a local t
        loc_t = t-np.floor(t/duration_step)*duration_step
        # If we are just in the change between expiration and inspiration
        if np.abs(loc_t-Tinsp)<1e-8:
            # Start of expiration
            tree.update_gamma(pedley_gamma_exp)
        elif np.abs(loc_t)<1e-8:
            # Start  of cycle
            tree.update_gamma(pedley_gamma_insp)
        
        if inspiratory_pause_stop:
            if (loc_t-Tinsp) > 1e-4:
                print("Stopping the simulation after the inspiratory pause by design.")
                return True
        
# %% Start of the restart mechanism

        # Restarting mechanism
        # Note that we are not entering the iterative cycles while checking this.
        if restart_from_last_checkpoint:
            
            # Find a 't' really close to the t from the checkpoint
            # TODO: Check if this is numerical precision gets us in trouble
            time_difference = np.abs(t-checkpoints[last_checkpoint_id])
            
            plateau_time_difference = np.abs(t-Tinsp)
            
            if plateau_time_difference <1e-8:
                # Load pressure data
                chk = np.sum([ncheckpoints[i] for i in range(4)])
                signals_path = output_to+"Checkpoints/Signals/%i/"%chk 
                presionestodas = list(np.load(signals_path+"presionestodas.npy"))
                pplat = presionestodas[np.sum([ncheckpoints[jj]*ninternaldivs[jj] for jj in range(4)])-1]
                print("Loading plateau pressure: %.2f"%pplat)
            
            
            if not time_difference<1e-8:
                # Skip those t's not within tolerance
                continue
            
            elif t>checkpoints[last_checkpoint_id]+0.01: # arbitrary delta
                print("Exceeded last checkpoint t. Stopping Code")
                return False
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
                    print("Missing the signals corresponding to checkpoint %i"%chk)
                    return False
                else:
                    Jacob = list(np.load(signals_path+"volumenes.npy"))
                    presionestodas = list(np.load(signals_path+"presionestodas.npy"))
                    fluxes = list(np.load(signals_path+"fluxes.npy"))
                    prescribedfluxes = list(np.load(signals_path+"prescribedfluxes.npy"))
                    effectivetimes = list(np.load(signals_path+"effectivetimes.npy"))
                    peep = presionestodas[0] # Assume PEEP is the original pressure value

                # Initialize arrays that should be created from a previous step in the iterative cycle
                # Determine average pressures from initial conditions (0)
                current_it_subdomain_pressure = determineAveragePressures(p0,u0,dx,Nsd,Fprev=None)
                            
                # Change format for compatibility with the tree system
                Pdistal = {tree.translate["s2p"][s]:current_it_subdomain_pressure[s] for s in range(Nsd)}
                            
                # Assemble linear system using the average pressures
                tree.assemble_linear_system(Pdistal=Pdistal, Q0=Q)
                        
                # Initialize the subdomain deformed volume array using the reference volumes
                current_it_subdomain_defomed_volume = np.empty(Nsd,dtype=float)
                for j in range(Nsd):
                    dummy = (md.JJ(u0,p0,Fprev=None)-1+phi0)*dx(j)
                    current_it_subdomain_defomed_volume[j] = dolfin.assemble(dummy)
                    
                continue        
        
# %% End of the restarting mechanism

        # Save the time when this cycle started
        iterativetime += [time.time()]
        
        if i==0: 
            # Initialize the arrays
            
            # Determine average pressures from initial conditions (0)
            current_it_subdomain_pressure = determineAveragePressures(p,u,dx,Nsd,Fprev=None)
                        
            # Change format for compatibility with the tree system
            Pdistal = {tree.translate["s2p"][s]:current_it_subdomain_pressure[s] for s in range(Nsd)}
                        
            # Assemble linear system using the average pressures
            tree.assemble_linear_system(Pdistal=Pdistal, Q0=Q)
            # Initialize the subdomain deformed volume array using the reference volumes
            current_it_subdomain_defomed_volume = np.empty(Nsd,dtype=float)
            J = md.JJ(u,p,Fprev=None)
            for j in range(Nsd):
                dummy = (J-1+phi0)*dx(j)
                current_it_subdomain_defomed_volume[j] = dolfin.assemble(dummy)
                            
        else:
            # Report the time employed in solving the previous iterative cycle
            if len(iterativetime)>2:
                print("time/it: %i (s)"%((iterativetime[-1]-iterativetime[-2])))
        
        print("Time: %.3f [s]"%t)

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
            
            # Sample a plateau pressure before
            if use_bc_functions:
                if np.abs(Tinsp-t)<1e-4: pplat = tree.x[tree.N]
            
            # Set the inlet variables for the airway tree
            Q0, P0 = determineTreeInletConditions(t, Q, Tinsp, duration_step,
                                                  presfn=presfn, flowfn=flowfn, 
                                                  pplat=pplat, peep=peep)
            
            # Update the linear system with the Pdistal which is defined at the end of an iterative loop with the 
            # previous step subdomain-averaged values.
            tree.update_linear_system(Pdistal, Q0=Q0, P0=P0)
            
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
                        print("Divergence at Pedley's iterations")
                        return False
                    
            # Retrieve the new fluxes that are going to become input for the poromechanical system
            subdomain_fluxes = tree.retrieve_qs()
            
            # Update the source values according to subdomain air volume and the recently computed fluxes.
            sources = {j:subdomain_fluxes[j]/prev_it_subdomain_defomed_volume[j] for j in range(Nsd)}
            src.update(sources)
            
            # Solve the poromechanical model
            solver.solve()
             
            # Determine the new subdomain averaged pressures and store them
            # in current_subd_pressure.
            current_it_subdomain_pressure = determineAveragePressures(p,u,dx,Nsd)

            # Determine the current deformed volume of the lung
            current_it_subdomain_defomed_volume = np.empty(Nsd,dtype=float)
            J = md.JJ(u,p,Fprev=None)
            for j in range(Nsd):
                # dummy = (J+phi0)*dx(j) # THIS IS WEIRD; Kept for reviewing later
                dummy = (J+phi0-1)*dx(j) # This is the integral of the current fluid volume
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
            
            
            # Convergence criteria is simplified to only assess the pressure stability, 
            # as keeping volume on check tends to be an issue. Also, the difference term in volume
            # never reaches something like 1e-8 but it is always < 1mm3 which is small enough for a 
            # lung that is a couple of liters big. 
            
            convergence_criteria = p_diff < tol_p_it
            # Alternative : convergence_criteria =  (p_diff < tol_p_it) and (v_diff < tol_v_it)
          
            if convergence_criteria:
                
                # Jacobian
                J = md.JJ(u,p,Fprev=None)
                # Compute mean Jacobian    
                vol_=dolfin.assemble(J*dx)
                volL=vol_/(10**6)
                deltavol=dolfin.assemble(((md.JJ(u,p,Fprev=None)-md.JJ(u0,p0,Fprev=None))/dt)*dx)
                
                # Append the new quantities
                presionestodas.append(tree.x[tree.N])
                Jacob.append(volL)

                # If we are in the starting time, keep this pressure value as a PEEP approximation
                if i == 0: peep = presionestodas[0]
                
                if use_bc_functions:
                    prescribedfluxes.append(flowfn(t)*1e-6)
                else:
                    prescribedfluxes.append(Q)
                
                fluxes.append(deltavol*1e-6)
                effectivetimes.append(t)    
                
                if save_vtk and False: tree.export_solution(output_to+"Airways/tree%6.6i.vtu"%i,compute_re=True)

                
                md.write_signals(output_to+"Signals",times,fluxes,
                                    presionestodas, Jacob,effectivetimes,i,
                                    iterativetime,prescribedfluxes)
                
                if np.any(np.abs(checkpoints-t)<1e-8) and save_vtk:
                    md.write_vtk(mesh,output_to+"VTK",cm,u,p,KK,dx,i,t,c_tissue,c3,Fprev=Fprev)
                
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
                                            iterativetime,prescribedfluxes)

                break
            
            elif not convergence_criteria and ii == (max_nit-1):
                print("Reached maximum number of iterations without convergence.")
                return False
    
    toc = time.time()
        
    overall_time = toc-tic
    
    print("Overall time: %.0f"%(overall_time))
    
