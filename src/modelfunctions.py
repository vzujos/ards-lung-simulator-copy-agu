# -*- coding: utf-8 -*-
"""
@author: Nibaldo Avilés-Rojas
"""

from dolfin import *    
import dolfin
import os
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import shutil

#%%
I = Identity(3)

def stress_bir2019(u, p,p_alv=None,c=0.3567,c3=5.766e-3, beta=1.075,Fprev=None):
    I = Identity(3)
    F = (I + grad(u))
    if not Fprev is None:        
        F = F*Fprev        
    C=variable(F.T*F)
    I1C=variable(tr(C))
    I2C=0.5*((tr(C)**2)-tr(C*C))
    I3C=variable(det(C))
    J=variable(sqrt(I3C))
    I1raya=I1C*pow(I3C,-1/3)
    psi=c*(I1C-3)  +  (c/beta)*(pow(I3C,-beta)-1)+(278.2/1000)*pow((pow(I3C,-1/3)*I1C-3),3)+c3*pow((pow(I3C,1/3)-1),6)
    S=2*diff(psi,C)
    
    PK=F*S-p*J*inv(F).T 
    
    return PK             
 
def stress_bir2019_inv(u, p,p_alv=None, c=0.3567,beta=1.075,c3=5.766e-3):
    I = Identity(3)
    F = inv(I + grad(u))
    C=variable(F.T*F)
    I1C=variable(tr(C))
    I2C=0.5*((tr(C)**2)-tr(C*C))
    I3C=variable(det(C))
    J=variable(sqrt(I3C))
    I1raya=I1C*pow(I3C,-1/3)
    psi=c*(I1C-3)  +  (c/beta)*(pow(I3C,-beta)-1)+(278.2/1000)*pow((pow(I3C,-1/3)*I1C-3),3)+c3*pow((pow(I3C,1/3)-1),6)
    S=2*diff(psi,C)    
    
    if p_alv is None:
        sigma = 1./J*F*S*F.T - p*I # Akin to what we had for P
    else:
        sigma = 1./J*F*S*F.T - (p+p_alv)*I # Akin to what we had for P
        
    return sigma     
 
def stress_perez(u, p,p_alv=None,c=0.3567,c3=5.766e-3, beta=1.075,Fprev=None):
    I = Identity(3)
    F = (I + grad(u))
    if not Fprev is None:        
        F = F*Fprev        
    C=variable(F.T*F)
    I1C=variable(tr(C))
    I2C=0.5*((tr(C)**2)-tr(C*C))
    I3C=variable(det(C))
    J=variable(sqrt(I3C))
    I1raya=I1C*pow(I3C,-1/3)
    psi=c*(I1C-3)  +  (c/beta)*(pow(I3C,-beta)-1)+c3*pow((pow(I3C,1/3)-1),6)
    S=2*diff(psi,C)
    
    PK=F*S-p*J*inv(F).T 
    
    return PK             
 
def stress_perez_inv(u, p,p_alv=None, c=0.3567,beta=1.075,c3=5.766e-3):
    I = Identity(3)
    F = inv(I + grad(u))
    C=variable(F.T*F)
    I1C=variable(tr(C))
    I2C=0.5*((tr(C)**2)-tr(C*C))
    I3C=variable(det(C))
    J=variable(sqrt(I3C))
    I1raya=I1C*pow(I3C,-1/3)
    psi=c*(I1C-3)  +  (c/beta)*(pow(I3C,-beta)-1)+c3*pow((pow(I3C,1/3)-1),6)
    S=2*diff(psi,C)    
    
    if p_alv is None:
        sigma = 1./J*F*S*F.T - p*I # Akin to what we had for P
    else:
        sigma = 1./J*F*S*F.T - (p+p_alv)*I # Akin to what we had for P
        
    return sigma     
 
def stress_ma(u, p, c=2.0, p_alv=None, a = 0.43, b=-0.6, Fprev = None):
    I = Identity(3)
    F = I + grad(u)
    
    if not Fprev is None:
        F = F*Fprev        
    J = det(F)
    C=variable(F.T*F)
    E=0.5*(C-I)
    I1E=tr(E)
    I2E=0.5*((tr(E)**2)-tr(E*E))
    I3E=det(E)
    aa=Constant(a)
    bb=Constant(b)
    psi=c*exp(aa*I1E*I1E+bb*I2E)
    S=2*diff(psi,C)
    PK=F*S-p*J*inv(F).T 

    return PK          


def stress_ma_inv(u, p, p_alv=None, c=2.0, a = 0.43, b=-0.6):
    I = Identity(3)
    F = inv(I + grad(u))
    J = det(F)
    C=variable(F.T*F)
    E=0.5*(C-I)
    I1E=tr(E)
    I2E=0.5*((tr(E)**2)-tr(E*E))
    I3E=det(E)
    aa=Constant(a)
    bb=Constant(b)
    psi=c*exp(aa*I1E*I1E+bb*I2E)
    S=2*diff(psi,C)
    if p_alv is None:
        sigma = 1./J*F*S*F.T - p*I # Akin to what we had for P
    else:
        sigma = 1./J*F*S*F.T - (p+p_alv)*I # Akin to what we had for P
    return sigma       

def write_vtk(mesh,path,cm,u,p,KK,dx,i,t,c,c3,tol=1e-8, Fprev=None):
    
    # Ademas, nosotros construimos espacios para proyectar funciones y tensores outputs
    T_DG0_out = dolfin.TensorFunctionSpace(mesh, "CG", 1) 
    V0_out = dolfin.FunctionSpace(mesh, 'CG', 1)
    V1_out = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)  ##

    I = Identity(3)
    FF = I + dolfin.grad(u)             # Deformation gradient 
    if not Fprev is None:
       FF = FF*Fprev

    vol=dolfin.assemble((det(I+grad(u)))*dx)
    J_pp = dolfin.det(FF)   
    JJ_proj=dolfin.project(J_pp,V0_out,solver_type='cg',preconditioner_type='hypre_amg')
        
    dolfin.File(path+"/Jacobian_%.6f.pvd"%t)<< JJ_proj;   
    dolfin.File(path+"/Pressure_%.6f.pvd"%t)<< (p,t);  
    dolfin.File(path+"/Displacement_%.6f.pvd"%t)<< (u,t);  

    FFinv_t = dolfin.inv(FF).T
    FFinv=dolfin.inv(FF)
    
    if cm == "bir2019":
        PK_pp=stress_bir2019(u,p,c=c,c3=c3,Fprev=Fprev)   
    elif cm == "ma":
        PK_pp=stress_ma(u,p,c=c,Fprev=Fprev)
    elif cm == "perez":
        PK_pp=stress_perez(u,p,c=c,c3=c3,Fprev=Fprev)   
    
    P_PK_eff_pp = PK_pp # ya incluye el termino del fluido #- J_pp*p*FFinv_t
    sigma_pp = (J_pp**(-1))*(P_PK_eff_pp )*FF.T
    hyd_stress=(1/3)*tr(sigma_pp)
    sigma_dev=sigma_pp-hyd_stress*I
    vm_stress=sqrt((3/2)*inner(sigma_dev,sigma_dev))
    vm_proj=dolfin.project(vm_stress,V0_out,solver_type='cg',preconditioner_type='hypre_amg')
    hyd_proj=dolfin.project(hyd_stress,V0_out,solver_type='cg',preconditioner_type='hypre_amg')
       
    dolfin.File(path+"/VM_%.6f.pvd"%t)<< vm_proj;   
    dolfin.File(path+"/HYD_%.6f.pvd"%t)<< hyd_proj;  
    
    # Internal flow within the lung
    qq = J_pp*FFinv*(KK*I)*FFinv_t*grad(p)
    qq_proj = dolfin.project(qq,V1_out,solver_type='cg',preconditioner_type='hypre_amg')
    qq_proj.rename("qq","qq")
    dolfin.File(path+"/QQint_%.6f.pvd"%t) << qq_proj
    
    return 

def write_signals(folder,times,fluxes,presionestodas,Jacob,
                     effectivetimes,i, iterativetime,prescribedfluxes):
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    np.save(folder+"/tiempos.npy",times[0:i+1])
    np.save(folder+"/fluxes.npy",fluxes)
    np.save(folder+"/presionestodas.npy",presionestodas)
    np.save(folder+"/volumenes.npy",Jacob)
    np.save(folder+"/effectivetimes.npy",effectivetimes)
    np.save(folder+"/iterativetime.npy",iterativetime)
    np.save(folder+"/prescribedfluxes.npy",prescribedfluxes)
    
    return

def F(u,p,Fprev=None):
    I = Identity(3)
    F = I + grad(u) 
    if not Fprev is None:
        F = F*Fprev
    return F

def Finv(u,p,Fprev=None):
    I = Identity(3)
    F = I + grad(u)
    if not Fprev is None:
        F = F*Fprev
    Finverse=inv(F)
    return Finverse

def Finv_t(u,p,Fprev=None):
    I = Identity(3)
    F = I + grad(u)
    if not Fprev is None:
        F = F*Fprev
    Finverse=inv(F)        
    Finvt=Finverse.T
    return Finvt

def JJ(u,p,Fprev=None):
    I = Identity(3)
    F = I + grad(u)
    if not Fprev is None:
        F = F*Fprev
    J=det(F)
    return J


def lnJJ(u,p,Fprev=None): ###OBSE
    I = Identity(3)
    F = I + grad(u)
    if not Fprev is None:
        F = F*Fprev
    J=det(F)
    lnJJ=ln(J)
    return J

def phi(u,p,Fprev=None):
    I = Identity(3)
    F = I + grad(u)
    if not Fprev is None:
        F = F*Fprev  
    J=det(F)
    phi0=0.99
    phi=J-1+phi0
    return phi
   
def F2aux_mass(u,p,KK,gravity_activate=False,rho_a=None,gvec=None,Fprev=None):
    I = Identity(3)
    F = I + grad(u)
    
    if not Fprev is None:
        F = F*Fprev
        
    Finverse=inv(F)
    Ft = F.T
    Finvt=Finverse.T
    J=det(F)
    K=KK*I

    aux = J*Finverse*K*Finvt*(grad(p))
    
    if gravity_activate:
        aux += J*Finverse*K*Finvt*rho_a*Ft*gvec
        
    return aux

def equation_of_motion_regression(fileflujos,filepresiones,filetiempos,filevolumenes,name):
    from sklearn import linear_model
    flujos=np.asarray((fileflujos))*1/60
    presiones=10.2*np.asarray((filepresiones))
    tiempos=np.asarray((filetiempos))
    volumenes=np.asarray((filevolumenes)) 
    volumenes=volumenes
    
    FV=np.zeros((flujos.shape[0],2))
    FV[:,0]=flujos
    FV[:,1]=volumenes
    reg = linear_model.LinearRegression()
    reg.fit(FV,presiones)
    R,E=reg.coef_
    print(name)
    print('Resistence(R)=',(R,2), 'cm H2O L/S, ')
    print('Compliance (C)=',(1000*1/E), 'ml/cm H2O')

    print('---------------')
    P=E*volumenes+R*flujos
    Compliance=((1000*1/E))
    Resistance=R
    return Compliance, Resistance


def times_with_checkpoints(tstart,tend,ncheckpoints,ninternaldivisions, 
                           endpoint = False):
    '''
    Auxiliar function to create the timesteps for the execution of the code and
    assign checkpoints.
    '''
    
    # Number of intervals created when checkpoints are established
    ndivisions=ncheckpoints
    # If endpoint==True, we need an additional checkpoint to keep the same dt
    if endpoint: ncheckpoints+=1
    # Generate checkpoints
    checkpoints = np.linspace(tstart, tend, ncheckpoints, endpoint=endpoint)
    # Time list manager
    times = []
    
    # Generate subintervals within checkpoints
    for i in range(ndivisions):
        
        # Print checkpoint id
        t0 = checkpoints[i]
        
        # Adjust endpoints
        if i != (ndivisions-1):
            t1 = checkpoints[i+1]
        else:
            t1 = tend
        
        # Compute times within subinterval
        if i != (ndivisions-1): # Not final intervals
            subtimes = np.linspace(t0,t1,ninternaldivisions,endpoint=False)
        elif (i == (ndivisions-1)) and endpoint: # Final overall interval
            subtimes = np.linspace(t0,t1,ninternaldivisions+1,endpoint=True)
        else: # Final but not last interval (of the complete simulation)
            subtimes = np.linspace(t0,t1,ninternaldivisions,endpoint=False)
            
        # Add the subtimes into the time list
        times += list(subtimes)
    
    return checkpoints, np.array(times)


def give_times_pressures_deltas(Nciclos, p_min,p_max):
    duration_step=3
    for ciclo in np.arange(1,Nciclos+1):
        
        if ciclo==Nciclos:    
            end=True
        else:
            end=False
            
        p0=p_min
        p1=p_max
        p2=p_max
        p3=p_min
        p34=p_min
        p4=p_min
           
        t0=duration_step*(ciclo-1)
        t1=0.1+t0
        t2=1+t0
        t3=1.025+t0
        t4=duration_step+t0
        t34=2.0+t0 #1.5 + to
        #Discretizacion temporal
        n0=10 #subida
        n1=20 #15 # flujo >0
        n2=10 #bajada
        n23=10 #15
        n3=10 #15 #nulo
        
        times0=np.linspace(t0,t1,n0,endpoint=False)
        times1=np.linspace(t1,t2,n1,endpoint=False)
        times2=np.linspace(t2,t3,n2,endpoint=False)
        times3=np.linspace(t3,t34,n23,endpoint=False)
        times4=np.linspace(t34,t4,n3,endpoint=end)
        timesaux=np.concatenate((times0,times1,times2,times3,times4))
        
        ps0=np.linspace(p0,p1,n0,endpoint=False)
        ps1=np.linspace(p1,p2,n1,endpoint=False)
        ps2=np.linspace(p2,p3,n2,endpoint=False)
        ps3=np.linspace(p3,p34,n23,endpoint=False)
        ps4=np.linspace(p34,p4,n3,endpoint=end)
        psaux=np.concatenate((ps0,ps1,ps2,ps3,ps4))
        
        if ciclo==1:
            times=timesaux
            ps=psaux
        else:
            times=np.concatenate((times,timesaux))
            ps=np.concatenate((ps,psaux))
    times=times[1:]
    ps=ps[1:]
    dts=[]
    for i in np.arange(len(times)):
        if i==0:
            dts.append(times[i])
        else:
            dts.append(times[i]-times[i-1])  
    return times,ps,dts

def times_and_pressures(Nciclos,pmed,pmax,Tup,Tplat,Tdown1,Tdown2,Tbuffer=0.10,
                        pmin=0.0,
                        ncheckpoints=[2,4,2,2,2], 
                        ninternaldivs=[2,10,2,5,10],):
    
    '''
    Based on the function 'give_times_fluxes_deltas_VCV', includes an 
    additional output called checkpoints whose values are used to tell the 
    program to save its current state so long runs can be started from there.
    
    Also, it includes the lists ncheckpoints and ninternaldivs to callibrate
    the requested checkpoints and internal divisions within checkpoints which
    determinate the timestep.
    '''
    
    duration_step=Tup+Tplat+Tdown1+Tdown2 #duracion ciclo inspiracion-expiracion

    for ciclo in np.arange(1,Nciclos+1):
        
        end = True if ciclo==Nciclos else False
                
        t0=duration_step*(ciclo-1) # Inicio del ciclo, P=Pmin
        t1=t0+Tup # Insuflación, P alcanza P=Pmax
        t2=t0+Tup+Tbuffer # Suavizacion del dT para acomodar grandes deformaciones
        t3=t0+Tup+Tplat  # Fin del periodo donde P=Pmax
        t4=t3+Tdown1  # Descenso a una P=Pmed con una determinada pendiente
        t5=t4+Tdown2 # Descenso final a un P=Pmin, Tdown2>Tdown1
        
        # Corresponding fluxes
        p0 = pmin
        p1 = pmax
        p2 = pmax
        p3 = pmax
        p4 = pmed
        p5 = pmin
        
        # Extract number of checkpoints and number of internal divisions per segment
        nchk1,nchk2,nchk3,nchk4,nchk5 = ncheckpoints
        nintdiv1,nintdiv2,nintdiv3,nintdiv4,nintdiv5 = ninternaldivs
        
        # Determine times and checkpoints per segment
        checkpoints1,times1=times_with_checkpoints(t0,t1,nchk1,nintdiv1,endpoint=False)
        checkpoints2,times2=times_with_checkpoints(t1,t2,nchk2,nintdiv2,endpoint=False)
        checkpoints3,times3=times_with_checkpoints(t2,t3,nchk3,nintdiv3,endpoint=False)
        checkpoints4,times4=times_with_checkpoints(t3,t4,nchk4,nintdiv4,endpoint=False)
        checkpoints5,times5=times_with_checkpoints(t4,t5,nchk5,nintdiv5,endpoint=end)
        
        # Determine fluxes corresponding to each timestep
        ps1 = np.linspace(p0,p1,nchk1*nintdiv1,endpoint=False)
        ps2 = np.linspace(p1,p2,nchk2*nintdiv2,endpoint=False)
        ps3 = np.linspace(p2,p3,nchk3*nintdiv3,endpoint=False)
        ps4 = np.linspace(p3,p4,nchk4*nintdiv4,endpoint=False)
        if not end:
            ps5 = np.linspace(p4,p5,(nchk5*nintdiv5),endpoint=end)
        else:
            ps5 = np.linspace(p4,p5,(nchk5*nintdiv5)+1,endpoint=end)
            
        # Join times, checkpoints and fluxes of all segments
        checkpoints_ = np.concatenate([checkpoints1,checkpoints2,checkpoints3,checkpoints4,checkpoints5])
        times_ = np.concatenate([times1,times2,times3,times4,times5])
        ps_ = np.concatenate([ps1,ps2,ps3,ps4,ps5])
        
        # Merge different cycles in big lists
        if ciclo==1:
            times = times_
            checkpoints = checkpoints_
            ps = ps_
        else:
            times = np.concatenate([times,times_])
            checkpoints = np.concatenate([checkpoints,checkpoints_])
            ps = np.concatenate([ps,ps_])
        
    times=times[1:]
    checkpoints=np.array(checkpoints[1:])
    ps=ps[1:]
        
    # Determine timesteps
    dts=[]
    for i in np.arange(len(times)):
        if i == 0:
            dts.append(times[i])
        else:
            dts.append(times[i]-times[i-1])
    dts=np.array(dts)
    
    # Return values
    return times, ps, dts, checkpoints

def vc_supersyringe_times_and_fluxes(Ncycles,vol_step,Tsyr,Tpause,ninternaldivs, 
                                     dt_buffer = 0.05,Tpause_split=0.3,
                                     inverse_cycles=True):
    
    '''
    
    '''
    # Determine flux for the syringe stage
    q_step = vol_step/Tsyr
    
    # Duration of a whole cycle
    duration_step=Tsyr+Tpause 
    
    # Extract number of checkpoints and number of internal divisions per segment
    nintdiv1,nintdiv2,nintdiv3,nintdiv4,nintdiv5 = ninternaldivs
    
    if inverse_cycles:
        Ncycles *= 2
        
    
    # Initiate loop
    for ciclo in np.arange(1,Ncycles+1):
        
        # Adjust number of divisions for the last cycle
        if ciclo==Ncycles:
            end = True 
            corrector = 1
        else:
            end = False
            corrector = 0
        
        if inverse_cycles and ciclo==(int(Ncycles/2)+1):
            q_step *= -1
        
        # Relevant events declaration
        t0=duration_step*(ciclo-1) # Event 0: Initiate
        t1=t0+dt_buffer # Event 1: Ramp from 0 to flux
        t2=t0+Tsyr # Event 2: Ramp form flux to 0
        t3=t0+Tsyr+dt_buffer # Event 3: Fine timestepping
        t4=t0+Tsyr+Tpause*Tpause_split # Event 4: Coarse time-stepping
        t5=t0+duration_step # Event 5: New cycle
        
        # Determine times and checkpoints per segment
        times1=np.linspace(t0,t1,nintdiv1,endpoint=False)
        times2=np.linspace(t1,t2,nintdiv2,endpoint=False)
        times3=np.linspace(t2,t3,nintdiv3,endpoint=False)
        times4=np.linspace(t3,t4,nintdiv4,endpoint=False)
        times5=np.linspace(t4,t5,nintdiv5+corrector,endpoint=end)
        
        # Determine fluxes corresponding to each timestep
        qs1 = np.linspace(0.0,-q_step,nintdiv1,endpoint=False)
        qs2 = np.linspace(-q_step,-q_step,nintdiv2,endpoint=False)
        qs3 = np.linspace(-q_step,0.0,nintdiv3,endpoint=False)
        qs4 = np.linspace(0.0,0.0,nintdiv4,endpoint=False)
        qs5 = np.linspace(0.0,0.0,nintdiv5+corrector,endpoint=end)
        
        # Join times, checkpoints and fluxes of all segments
        times_ = np.concatenate([times1,times2,times3,times4,times5])
        qs_ = np.concatenate([qs1,qs2,qs3,qs4,qs5])
        
        # Merge different cycles in big lists
        if ciclo==1:
            times = times_
            qs = qs_
        else:
            times = np.concatenate([times,times_])
            qs = np.concatenate([qs,qs_])
        
    times=times[1:]
    qs=qs[1:]
        
    # Determine timesteps
    dts=[]
    for i in np.arange(len(times)):
        if i == 0:
            dts.append(times[i])
        else:
            dts.append(times[i]-times[i-1])
    dts=np.array(dts)
    
    # Return values
    return times, qs, dts
    


def times_and_fluxes(Nciclos, vol_step,area,Tsyr,Texp,Tpausa,Tinsp,
                     ncheckpoints=[2,10,2,2,5,5], 
                     ninternaldivs=[3,10,5,5,5,10],):
    
    '''
    Based on the function 'give_times_fluxes_deltas_VCV', includes an 
    additional output called checkpoints whose values are used to tell the 
    program to save its current state so long runs can be started from there.
    
    Also, it includes the lists ncheckpoints and ninternaldivs to callibrate
    the requested checkpoints and internal divisions within checkpoints which
    determinate the timestep.
    '''
    
    q_step=vol_step/(area*Tsyr)
    
    for ciclo in np.arange(1,Nciclos+1):
        
        end = True if ciclo==Nciclos else False
        
        # Relevant events declaration
        duration_step=Tsyr+Tpausa+Texp #duracion ciclo inspiracion-expiracion
        t0=duration_step*(ciclo-1)
        t1=0.001+t0
        t2=Tsyr+t0
        t3=Tsyr+t0+0.001  #tiempo cuando finaliza la insuflacion, por lo que comienza la pausa
        t4=Tsyr+Tpausa+t0  #tiempo que finaliza la pausa y comienza presion =0 , cambio de BC
        t5=t4+Tpausa*0.2 #tiempo auxiliar para detertar peak negativo
        t6=duration_step+t0 #fin de ciclo 
        
        # Corresponding fluxes
        q0=0.0
        q1=-q_step
        q2=-q_step
        q3=0.0
        q4=0.0
        q5=0.0
        q6=0.0
        
        # Extract number of checkpoints and number of internal divisions per segment
        nchk1,nchk2,nchk3,nchk4,nchk5,nchk6 = ncheckpoints
        nintdiv1,nintdiv2,nintdiv3,nintdiv4,nintdiv5,nintdiv6 = ninternaldivs
        
        # Determine times and checkpoints per segment
        checkpoints1,times1=times_with_checkpoints(t0,t1,nchk1,nintdiv1,endpoint=False)
        checkpoints2,times2=times_with_checkpoints(t1,t2,nchk2,nintdiv2,endpoint=False)
        checkpoints3,times3=times_with_checkpoints(t2,t3,nchk3,nintdiv3,endpoint=False)
        checkpoints4,times4=times_with_checkpoints(t3,t4,nchk4,nintdiv4,endpoint=False)
        checkpoints5,times5=times_with_checkpoints(t4,t5,nchk5,nintdiv5,endpoint=False)
        checkpoints6,times6=times_with_checkpoints(t5,t6,nchk6,nintdiv6,endpoint=end)
        
        # Determine fluxes corresponding to each timestep
        qs1 = np.linspace(q0,q1,nchk1*nintdiv1,endpoint=False)
        qs2 = np.linspace(q1,q2,nchk2*nintdiv2,endpoint=False)
        qs3 = np.linspace(q2,q3,nchk3*nintdiv3,endpoint=False)
        qs4 = np.linspace(q3,q4,nchk4*nintdiv4,endpoint=False)
        qs5 = np.linspace(q4,q5,nchk5*nintdiv5,endpoint=False)
        if not end:
            qs6 = np.linspace(q5,q6,(nchk6*nintdiv6),endpoint=end)
        else:
            qs6 = np.linspace(q6,q6,(nchk6*nintdiv6)+1,endpoint=end)


        # Join times, checkpoints and fluxes of all segments
        checkpoints_ = np.concatenate([checkpoints1,checkpoints2,checkpoints3,checkpoints4,checkpoints5,checkpoints6])
        times_ = np.concatenate([times1,times2,times3,times4,times5,times6])
        qs_ = np.concatenate([qs1,qs2,qs3,qs4,qs5,qs6])
        
        # Merge different cycles in big lists
        if ciclo==1:
            times = times_
            checkpoints = checkpoints_
            qs = qs_
        else:
            times = np.concatenate([times,times_])
            checkpoints = np.concatenate([checkpoints,checkpoints_])
            qs = np.concatenate([qs,qs_])
        
    times=times[1:]
    checkpoints=np.array(checkpoints[1:])
    qs=qs[1:]
        
    # Determine timesteps
    dts=[]
    for i in np.arange(len(times)):
        if i == 0:
            dts.append(times[i])
        else:
            dts.append(times[i]-times[i-1])
    dts=np.array(dts)
    
    # Return values
    return times, qs, dts, checkpoints


def vc_include_peep(times, qs, dts, checkpoints, 
                    peep_initial_value, 
                    peep_target_value,
                    peep_times, peep_nintdivs):
    
    # Specify times and divisions for both regions
    peep_time_ramp, peep_time_plateau = peep_times
    ndivs_ramp, ndivs_plateau = peep_nintdivs
    
    # Generate linear steps of pressure for the pressure cycles
    ps_ramp = np.linspace(peep_initial_value,peep_target_value,ndivs_ramp,endpoint=False)
    ps_plateau = np.linspace(peep_target_value,peep_target_value,ndivs_plateau+1,endpoint=True)
    null_ps = np.full_like(qs,peep_target_value)
    new_ps = np.concatenate([ps_ramp,ps_plateau,null_ps])
    
    # Create a new list specific for peep-asssociated times, then update the
    # old times list to account for the time dedicated to pressure-controllled
    # inflation, and then concatenate both arrays
    peep_times_ramp = np.linspace(0,peep_time_ramp, ndivs_ramp,endpoint=False)
    peep_times_plateau = np.linspace(peep_time_ramp,peep_time_plateau+peep_time_ramp,
                                     ndivs_plateau+1,endpoint=True)
    times += peep_time_ramp+peep_time_plateau
    new_times = np.concatenate([peep_times_ramp,peep_times_plateau,times])
    
    # Update the dts using the new times
    pdt_ramp = peep_time_ramp/ndivs_ramp
    pdt_plateau = peep_time_plateau/ndivs_plateau
    peep_dts_ramp = np.linspace(pdt_ramp,pdt_ramp, ndivs_ramp,endpoint=False)
    peep_dts_plateau = np.linspace(pdt_plateau,pdt_plateau, ndivs_plateau+1,endpoint=True)
    new_dts = np.concatenate([peep_dts_ramp,peep_dts_plateau,dts])
    
    # Displace the time associated with the checkpoints using peep_time
    new_checkpoints = checkpoints + peep_time_ramp + peep_time_plateau
    # Add a new checkpoint at the end of the ramp
    new_checkpoints = np.insert(new_checkpoints, 0, peep_time_ramp, axis=0)
    
    # append empty qs for the peep region
    peep_qs_ramp = np.linspace(0.0,0.0,ndivs_ramp,endpoint=False)
    peep_qs_plateau = np.linspace(0.0,0.0,ndivs_plateau+1,endpoint=True)
    new_qs = np.concatenate([peep_qs_ramp,peep_qs_plateau,qs])
    
    # return the new variables 
    return new_times, new_qs, new_ps, new_dts, new_checkpoints,
    
    
