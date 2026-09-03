# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:24:11 2023

@author: angus
"""

import os 
import numpy as np
import dolfin
from dolfin import *
import modelfunctions as md


def solver_configuration(solver_type, solver_dict=None):
    
    '''
    store here different configurations for the non-linear solvers 
    '''
    
    if solver_type == "iterative":
        snes_solver_parameters = {"nonlinear_solver":"snes",
                                  "snes_solver"    :{"linear_solver"   : "gmres", # lu mumps gmres
                                                     "maximum_iterations": 20,
                                                     "report": True,
                                                     "error_on_nonconvergence": False,
                                                     "absolute_tolerance":1e-10,
                                                     "relative_tolerance":1e-8,
                                                     "line_search":"basic", # "basic", "bt", "l2", "cp", "nleqerr"
                                                     "preconditioner":"default",
                                                     },
                               "newton_solver":{"krylov_solver":{"nonzero_initial_guess":True}
                                }}
    
    elif solver_type == "direct":
        snes_solver_parameters = {"nonlinear_solver":"snes",
                                  "snes_solver":{"linear_solver":"mumps",
                                                 "relative_tolerance":1e-8,
                                                 "absolute_tolerance":1e-10,
                                                 "maximum_iterations":60,
                                                 "line_search":"basic",
                                                 "report":True,
                                                 "error_on_nonconvergence":True,
                                                 "preconditioner":"default"}
                              }
    elif solver_type == "dict" and not solver_dict is None:
        snes_solver_parameters = solver_dict
    else:
        raise Exception("Wrong solver type: '%s'"%solver_type)
        
    return snes_solver_parameters

def manage_output_directory(output_to,codename, restart_from_last_checkpoint):
    
    '''
    Make sure a valid directory address exists and create a new folder
    to handle this execution
    '''
    
    # Check if the folder 'output_to' exists
    if not os.path.isdir(output_to):
        # Check if that output should be written in windows WSL format
        if not os.path.isdir("/mnt/c/"+output_to[3:]):
            # If not, raise an error
            raise ValueError("The path provided for output does not exist")
        else:
            # If exists, just rename this output
            output_to = "/mnt/c/"+output_to[3:]
    
    # If we are not restarting from the last checkpoint, then we just create a new folder following typical convention
    if not restart_from_last_checkpoint:
        
        provisory_output = output_to+codename+"/"
        cnt=0
        while os.path.isdir(provisory_output):
            # Folder already exists, create a new one using extended filename
            cnt+=1
            provisory_output = output_to+codename+"-%i/"%cnt
        output_to = provisory_output
        print("Directing outputs to: %s"%output_to)
        # If it doesn't exist, create it
        os.mkdir(output_to)
        
        for folder in ["Checkpoints", "Signals", "VTK","Airways"]:
            os.mkdir(output_to+folder)
            if folder == "Checkpoints":
                for subfolder in ["Signals","u","p"]:
                    os.mkdir(output_to+folder+"/"+subfolder)
            
                
    
        return output_to
    
    else:
        
        # See all the files existing in the main output folder
        candidates = os.listdir(output_to)
        # Generate a filter to discard every simulation not involved in the current codename
        filt = lambda x: codename in x
        # Filter candidates
        candidates = list(filter(filt,candidates))
        # Request last modification time for all the candidates
        ftime = lambda x: os.path.getmtime(output_to+x)
        modtimes = list(map(ftime,candidates))
        selected = candidates[np.argmax(modtimes)]
        # Return the path to the selected candidate
        return output_to+selected+"/"
        
        
def manage_mesh_directory(path_to_mesh,mesh_name,case):
    '''
    Make sure the mesh folder exists and has everything necessary for the 
    execution
    '''
    # Written in windows format
    path_to_mesh_a = path_to_mesh
    # Written in WSL format
    path_to_mesh_b = "/mnt/c/"+path_to_mesh_a[3:]

    # Check if folder exists
    if os.path.isdir(path_to_mesh_a):
        print("Base folder '%s' found"%path_to_mesh_a.split("/")[-2])
        path_to_mesh = path_to_mesh_a
        print("Executing code in 'spyder' format")
    elif os.path.isdir(path_to_mesh_b):
        print("Base folder '%s' found"%path_to_mesh_b.split("/")[-2])
        print("Executing code in a 'terminal' format")
        path_to_mesh = path_to_mesh_b
    else:
        print("Base folder '%s' not found"%path_to_mesh_a.split("/")[-2])
        print("Targeted folder did not exist. Check your paths!")
        raise ValueError()
  
    path_to_mesh = path_to_mesh+"%s/%s/"%(mesh_name,case)
    return path_to_mesh

def retrieve_last_checkpoint(path):
    
    '''
    open checkpoint-related folders, see the last checkpoint and retrieve
    its number.
    '''
    
    for folder in ["Checkpoints","Checkpoints/u","Checkpoints/p",
                   "Checkpoints/Signals"]:
        if not os.path.isdir(path+folder):
            raise Exception("There are no checkpoints to restart from!")
    
    chk_list = os.listdir(path+"Checkpoints/u")
    # Split xdmf from h5 and other possible objects
    xdmf_filter = lambda x: x.split(".")[1] == "xdmf"
    ids = list(filter(xdmf_filter,chk_list))
    # Retrieve last checkpoint
    last_checkpoint_name=sorted(ids)[-1]
    # Retrieve last checkpoint id. This value is diminished in one to account
    # for the array "checkpoints" whose initial value is in the 0th loc.
    last_checkpoint_id = int(last_checkpoint_name.split(".xdmf")[0])-1
    
    print("Last checkpoint found: %s"%last_checkpoint_name)
    print("Loading...")
    
    return last_checkpoint_id

def resume_from_checkpoint(comm, path, u, p, last_checkpoint_id, save_checkpoints):

    last_checkpoint_name = "%3.3i.xdmf"%(last_checkpoint_id+1)
    
    try:
        # Read data from the folder and assign it to our relevant variables
        with XDMFFile(comm, path+"Checkpoints/u/"+last_checkpoint_name) as in_u:
            in_u.read_checkpoint(u,"u_check",last_checkpoint_id+1)
                
        with XDMFFile(comm, path+"Checkpoints/p/"+last_checkpoint_name) as in_p:
            in_p.read_checkpoint(p,"p_check",last_checkpoint_id+1)
        
        # Return the fields and a flag that indicates success
        return u, p, True
    
    except:
        # Return the fields and a flag that reports failure
        return u, p, False
    
    
def define_property(mesh, property_config):
    
    read_from_file = property_config['read_from_file']
    path_to_property = property_config['path_to_file']
    property_name = property_config['property_name']
    
    
    if read_from_file:

        print("Variable %s simulation:"%property_name)
        print(" > File path: %s"%path_to_property)
        
        # Define tissue stiffness as MeshFunction
        property_value = MeshFunction("double", mesh, path_to_property)
    
        # Code for C++ evaluation of permeability (conductivity)
        code_outline = """
        
        #include <pybind11/pybind11.h>
        #include <pybind11/eigen.h>
        namespace py = pybind11;
        
        #include <dolfin/function/Expression.h>
        #include <dolfin/mesh/MeshFunction.h>
        
        class Value : public dolfin::Expression
        {
        public:
        
          // Create expression with 1 component
          Value() : dolfin::Expression(1) {}
        
          // Function for evaluating expression on each cell
          void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
          {
            const uint cell_index = cell.index;
            values[0] = (*property_value)[cell_index];
          }
        
          // The data stored in mesh functions
          std::shared_ptr<dolfin::MeshFunction<double>> property_value;
        
        };
        
        PYBIND11_MODULE(SIGNATURE, m)
        {
          py::class_<Value, std::shared_ptr<Value>, dolfin::Expression>
            (m, "Value")
            .def(py::init<>())
            .def_readwrite("property_value", &Value::property_value);
        }
        
        """
        
        processed_values = dolfin.CompiledExpression(compile_cpp_code(code_outline).Value(),
                                          property_value=property_value, degree=0)
    
        property_value = processed_values[0]
        
    else:    
        
        property_value = property_config['property_value']
        
        print("Constant %s simulation:"%property_name)
        print(" > Mean value = %.2f"%(property_value))
        
        H = FunctionSpace(mesh,"CG",1)
        h0 = Constant(property_value)
        property_value = interpolate(h0,H)
        
    return property_value


def generate_stiffness_file(mesh, stiffness_config, porosity_config):
    
    phi0 = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
    File(porosity_config['path_to_file']) >> phi0
    
    c_tissue = MeshFunction("double", mesh, mesh.topology().dim())
    
    # Read values associated to the default stiffness function
    c_tissue_min = stiffness_config['property_value']
    c_tissue_max = 5.0 * c_tissue_min
    
    # This value makes the exponential function softer (k~50) or stepper (k~100)
    if 'k' in stiffness_config:
        k = stiffness_config['k_value']
    else:
        k = 80 # Default value
    #
    if 'phi_transition' in stiffness_config:
        phi_t = stiffness_config['phi_transition']
    else:
        phi_t = 0.2 # Default value
    
    # If there is no stiffness function in the config dictionary
    if not ('stiffness_function' in stiffness_config):
        # Use a default function
        def stiffness_function(phi):
            # Sigmoid transition centered at 0.2, steepness controlled by k
            return c_tissue_max - (c_tissue_max - c_tissue_min) * (1 - 1 / (1 + np.exp(-k * (phi_t - phi))))

    # Otherwise, use a function that is created beforehand
    else:
        stiffness_function = stiffness_config['stiffness_function']
    
    # Whatever the source for the function was, evaluate for each cell    
    for cell in cells(mesh):
        phi_local = phi0[cell]
        c_tissue[cell] = stiffness_function(phi_local)
    
    # Export as an xml.gz file at the defined destination
    File(stiffness_config['path_to_file']) << c_tissue
    
    # Export pvd visualization
    V0 = FunctionSpace(mesh, "DG", 0)
    c_tissue_func = Function(V0)
    
    vals = c_tissue_func.vector().get_local()
    for cell in cells(mesh):
        vals[cell.index()] = c_tissue[cell]
    c_tissue_func.vector().set_local(vals)
    c_tissue_func.vector().apply("insert")
    
    c_tissue_func.rename("c_tissue", "")
    
    output_file = stiffness_config['path_to_file'].split('.xml.gz')[0]+'.pvd'

    File(output_file) << (c_tissue_func, 0.0)
        
    return None
    
    
def inverse_deformation_analysis(mesh, boundary_markers, material_properties, 
                                 ida_config, output_to, export_vtu=False):
    
    # Unpack material properties
    cm = material_properties['constitutive_model']
    c_tissue = material_properties['tissue_stiffness']
    c3 = material_properties['tissue_alt_parameter']
    kappa = material_properties['permeability']
    phi = material_properties['phi']
    rho_a = material_properties['rho_a']
    rho_t = material_properties['rho_t']
    gvec = material_properties['gravity_vector']
    K_inf = material_properties['K_inf']
    p_alv = material_properties['p_alv']
    
    #Define some dolfin parameters
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    dolfin.parameters["form_compiler"]["representation"] = "uflacs"
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
    dolfin.parameters["allow_extrapolation"] = True 
    dolfin.parameters["form_compiler"]["optimize"] = True 

    # Get mesh dimension and prepare boundary measure
    dx = dolfin.Measure("dx", domain=mesh)
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
             
    # Unknowns, values at previous step and test functions
    w = dolfin.Function(W)
    u, p = dolfin.split(w)
    
    _u, _p = dolfin.TestFunctions(W)
    du = dolfin.TrialFunction(W)
    
    # Unpack configuration for the inverse deformation analysis
    ida_folder = ida_config['folder']
    ida_activate_pressure_gradient = ida_config['activate_pressure_gradient']
    ida_pressure_reference = ida_config['pressure_reference']
    ida_delta_pressure = ida_config['delta_pressure']
    ida_activate_gravity = ida_config['activate_gravity']
    ida_solver_parameters = ida_config['solver_parameters']

    # Manage the output files directioning
    ida_output_to = output_to + ida_folder
    if not os.path.isdir(ida_output_to):
        os.mkdir(ida_output_to)
    
    if ida_activate_pressure_gradient:
        
        # Retrieve geometry bounds
        _, y_min, _ = mesh.coordinates().min(axis=0) 
        _, y_max, _ = mesh.coordinates().max(axis=0) # 
        
        # Define the pressures being applied
        p_max = ida_pressure_reference # transform cmH2O into kPa
        p_min = ida_pressure_reference + ida_delta_pressure / 10.1972 # Forced to be zero
        
        # See Pelosi et al. (2021) for a source where they declare that 
        # Delta Ppl is 10 cmH2O on average. 
        
        # Note that the lung is 'upside down' and the ventral region occurs
        # at y_min and and the dorsal at y_max.
        p_expr = Expression(
            "p_min + (p_max - p_min)*(y_max-x[1])/(y_max - y_min)",
            degree=1, p_max=p_max,p_min=p_min,y_min=y_min,y_max=y_max)     
        
        # Generate a function space for exporting this data
        Qp = FunctionSpace(mesh,"CG",1)
        p_pl = interpolate(p_expr, Qp)
        
        if export_vtu:
            # Export as a pvd file
            pleural_pressure_file = File(ida_output_to+"pleural-pressure.pvd")
            pleural_pressure_file << (p_pl, 0.0)

    # Definition of the deformation gradient based on the inverse displacement
    I = Identity(3)
    F = inv(I + grad(u))
    J = det(F)
    
    if cm=='ma':
        sigma=md.stress_ma_inv(u,p,c=c_tissue,p_alv=p_alv)
    elif cm=='bir2019':
        sigma=md.stress_bir2019_inv(u,p,c=c_tissue,p_alv=p_alv,c3=c3)
    elif cm=='perez':
        sigma=md.stress_perez_inv(u,p,c=c_tissue,p_alv=p_alv,c3=c3)
    else:
        print("CM ERROR")
            
    # Exception for whether p_alv is None
    if not (p_alv is None):
        q = -kappa*grad((p+p_alv))
    else:
        q = -kappa*grad((p))
        
    # Add gravity term to Darcy's equation
    if ida_activate_gravity:
        q = q - kappa*rho_a*gvec
    
    # Dorsal support with an infinitely stiff spring; More stable than locking displacement
    F3 = dolfin.inner(u,_u)*K_inf*dolfin.ds(subdomain_data=boundary_markers,
                                          subdomain_id=3) # Dorsal support  
    
    # Residual: mechanical + body force + incompressibility (pressure) equation
    Res = inner(sigma, grad(_u))*dx  + inner(q, grad(_p))*dx + F3
    
    if ida_activate_gravity:
        # Composite density term in deformed configuration
        R = (J-phi)*rho_t + phi*rho_a
        
        Res -= R*dot(gvec, _u)*dx
        
    if ida_activate_pressure_gradient:
        
        # Define a normal function
        n = FacetNormal(mesh)
        
        # Imposing the pleural pressure on the chest-wall surface
        F2 = dolfin.inner(p_pl*n,_u)* dolfin.ds(subdomain_data=boundary_markers,
                                                  subdomain_id=2) # Chest wall    
        
        # Adding this functional to the residual
        Res -= F2
    
    # Jacobian (derivative of residual w.r.t the mixed unknown w in direction du)
    Jac = derivative(Res, w, du)
    
    # Build the problem
    problem = NonlinearVariationalProblem(Res, w, J=Jac)
    solver = NonlinearVariationalSolver(problem)
    
    # First solver parameter definition
    solver.parameters.update(ida_solver_parameters)
    # Solve 
    solver.solve()

    # Extract solution components
    u, p = w.split()
    u.rename("Displacement", "")
    p.rename("Pressure", "")

    # Kinematics
    I = Identity(len(u))
    Fprev = inv(I + grad(u))
    
    t = 0.0 # Dummy

    if export_vtu:

        Jprev = det(Fprev)
        phi0 = phi + 1 - Jprev
        
        # Define output spaces
        T_ = TensorFunctionSpace(mesh, "CG", 1) 
        Q_ = FunctionSpace(mesh, 'CG', 1)
        V_ = VectorFunctionSpace(mesh, "Lagrange", 1)  ##
        
        # Determine hydrostatic pressure
        P_hyd = -1./3.*tr(sigma)*10.1972 # in cmH2O
        P_hyd_ = project(P_hyd, Q_)
        P_hyd_.rename("Hydrostatic pressure (cmH2O)","")
        
        # Deformation gradient
        F_out = project(Fprev, T_)
        F_out.rename("F","")
        
        # Original (?) porosity
        phi0 = phi + 1 - J
        phi0_out = project(phi0, Q_)
        phi0_out.rename("Phi_0","")
        
        # Eulerian porosity
        phi_out = project(phi, Q_)
        phi_out.rename("Phi","")
        
        # Delta Gas fraction
        dgf = phi - phi0 # Tiene sentido? Quizás no
        dgf_out = project(dgf, Q_)
        dgf_out.rename("Delta Gas Fraction","")
        
        # Jacobian
        Jproj = project(Jprev, Q_)
        Jproj.rename("Jacobian", "")
            
        # Export to XDMF        
        # Create one .pvd file per variable
        F_file = File(ida_output_to + "F.pvd")
        P_hyd_file = File(ida_output_to + "HydrostaticPressure.pvd")
        u_file = File(ida_output_to + "Displacement.pvd")
        p_file = File(ida_output_to + "Pressure.pvd")
        phi0_file = File(ida_output_to + "Phi0.pvd")
        phi_file = File(ida_output_to + "Phi.pvd")
        J_file = File(ida_output_to + "Jacobian.pvd")
        dgf_file = File(ida_output_to + "DeltaGasFraction.pvd")
        
        # Write each variable to its file
        F_file << (F_out, t)
        P_hyd_file << (P_hyd_, t)
        u_file << (u, t)
        
        if not (p_alv is None):
            p_file << (p+p_alv, t)
        else:
            p_file << (p, t)
            
        phi0_file << (phi0_out, t)
        phi_file << (phi_out, t)
        J_file << (Jproj, t)
        dgf_file << (dgf_out, t)       
    else:
        # Define output spaces
        Q_ = FunctionSpace(mesh, 'CG', 1)
        T_ = TensorFunctionSpace(mesh, "CG", 1) 
        # Determine and project Jacobian
        Jprev = det(Fprev)
        Jproj = project(Jprev, Q_)
        Jproj.rename("Jacobian", "")
        # Project F tensor
        F_out = project(Fprev, T_)
        F_out.rename("F","")
        # Export minimal required files for postprocessing purposes
        F_file = File(ida_output_to + "F.pvd")
        F_file << (F_out, t)
        J_file = File(ida_output_to + "Jacobian.pvd")
        J_file << (Jproj, t)

         
    return Fprev


def build_source_term(subdomain, sources):
    
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
    
    return src_obj(subdomain, sources) 