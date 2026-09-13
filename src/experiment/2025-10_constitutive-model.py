# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:35:38 2025

@author: angus
"""

import numpy as np
import matplotlib.pyplot as plt

def material_Birzle(C,c=0.3567,beta=1.075,c1=0.2782,c3=5.766e-3,d1=3,d3=6):
    
    I1=np.trace(C)
    I = np.eye(3)
    I3=np.linalg.det(C)
    C_inv = np.linalg.inv(C)
        
    S=  c*I - c*(I3**(-beta)) * C_inv.T + \
        c1 * d1 * (I1 * I3**(-1/3) - 3)**(d1 - 1) * (I * I3**(-1/3) - (1/3) * I1 * I3**(-4/3) * C_inv.T) +  \
        c3 * d3 * (I3**(1/3) - 1)**(d3 - 1) * (1/3) * I3**(1/3) * C_inv.T
    
    return S



def material_MaExponential(C, a=0.43, b=-0.6, c=2.0):

    I = np.eye(3)
    E = 0.5*(C - I)    
    J1 = np.trace(E)
    J2 = 0.5*( (np.trace(E)**2 - np.trace(E@E) ) )
    
  #  I1=np.trace(C)
   # I2=0.5*( (I1*I1-np.trace(C@C)))
    psi=c*np.exp(a*(J1**2)+b*J2)
    
    #S= psi *(  (I1*a-3*a-b )*np.eye(3)+0.5*b*(I1*np.eye(3)-C))
    S = psi * (a*2*J1*I + b*(J1*I - E ))
    return S



# %%

alpha = 1.00
lam = 1.0
lam_max = 3.1**(1/3)
lam_step = 0.01

Js = []
Sxxs_bir = []
Sxxs_ma = []

while lam < lam_max:
    
    # Build the deformation gradient tensor
    F=np.array([[ lam, 0., 0],
                [   0., lam, 0],
                [   0,    0,  lam]]) 
    
    # Pre-strain factor    
    F *= alpha 
    
    # Jacobian
    J=np.linalg.det(F)
    C=F.T*F
    
    # Evaluate models
    Se_bir=material_Birzle(C)
    Se_ma =material_MaExponential(C)
    
    # Extract relevant properties
    Sxxs_bir.append(Se_bir[0,0])
    Sxxs_ma.append(Se_ma[0,0])
    Js.append(J)
    
    # Update lambda value
    lam += lam_step

plt.figure(dpi=300)
plt.title("Comparing Ma's and Birzle's models ("+"$\\alpha=$"+"%.2f)"%alpha)
plt.plot(Js, Sxxs_bir, label="Birzle")
plt.plot(Js ,Sxxs_ma, label="Ma")
plt.xlabel("Jacobian (-)")
plt.ylim(0,4)
plt.xlim(1.0,3.5)
plt.ylabel("Stress (kPa)")
plt.grid(True)
plt.legend()
plt.show()

# %%

factors = [0.80, 0.90,1.0,1.10,1.20]
colors = ["darkred","lightcoral","black","lightsteelblue","darkblue"]
titles = ["'$a$' parameter",
          "'$b$' parameter",
          "'$c$' parameter"]

fig, axes = plt.subplots(ncols=3,figsize=(12,4),dpi=200)

for e,(ax,title) in enumerate(zip(axes,titles)):
    
    for nf, (ff,color) in enumerate(zip(factors,colors)):
        
        if nf == 2:
            ls = '--'
        else:
            ls = '-'
        
        Sxxs_ma = []
        Js = []
        lam = 1.0
        
        while lam < lam_max:
            
            # Build the deformation gradient tensor
            F=np.array([[ lam, 0., 0],
                        [   0., lam, 0],
                        [   0,    0,  lam]]) 
            
            # Pre-strain factor    
            F *= alpha 
            
            # Jacobian
            J=np.linalg.det(F)
            C=F.T*F
            
            # Evaluate models
            if e == 0:
                Se_ma =material_MaExponential(C,a=0.43*ff)
            elif e == 1:
                Se_ma =material_MaExponential(C,b=-0.6*ff)
            elif e == 2:
                Se_ma =material_MaExponential(C,c=2.0*ff)
            else:
                raise Exception("Invalid")

            # Extract relevant properties
            Sxxs_ma.append(Se_ma[0,0])
            Js.append(J)
            
            # Update lambda value
            lam += lam_step
        
        ax.plot(Js, Sxxs_ma, label="$\mu$=%.2f"%ff, ls=ls, color=color)
        ax.set_ylim((0,8))
        ax.set_title(title)
        ax.set_xlabel("Jacobian (-)")
        
        if e !=0:
            ax.set_yticks([])
        else:
            ax.set_ylabel("Stress (kPa)")
        
        ax.legend()

        
        
    plt.tight_layout()

# %%

alpha = 1.00
lam = 1.0
lam_max = 10**(1/3)
lam_step = 0.01

Js = []
Sxxs_bir = []
Sxxs_bir0 = []
Sxxs_bir1 = []
Sxxs_bir2 = []
Sxxs_bir3 = []
Sxxs_bir4 = []

f0 = 0.5
f1 = 2
f2 = 4
f3 = 8
f4 = 16 


while lam < lam_max:
    
    # Build the deformation gradient tensor
    F=np.array([[ lam, 0., 0],
                [   0., lam, 0],
                [   0,    0,  lam]]) 
    
    # Pre-strain factor    
    F *= alpha 
    
    # Jacobian
    J=np.linalg.det(F)
    C=F.T*F
    
    # Evaluate models
    Se_bir = material_Birzle(C,c=1.5)
    Se_mod = material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*15,d1=3,d3=6)
    Se_mod = material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*15,d1=3,d3=6)
    Se_mod = material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*15,d1=3,d3=6)
    Se_mod = material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*15,d1=3,d3=6)
    
    # Extract relevant properties
    Sxxs_bir.append(Se_bir[0,0])
    Sxxs_bir0.append(material_Birzle(C,c=1.5*0.5,beta=1.075,c1=0.2782,c3=5.766e-3*0.5,d1=3,d3=6)[0,0])
    Sxxs_bir1.append(material_Birzle(C,c=1.5*0.5,beta=1.075,c1=0.2782,c3=5.766e-3*2.0,d1=3,d3=6)[0,0])
    Sxxs_bir2.append(material_Birzle(C,c=1.5*2.0,beta=1.075,c1=0.2782,c3=5.766e-3*0.5,d1=3,d3=6)[0,0])
    Sxxs_bir3.append(material_Birzle(C,c=1.5*2.0,beta=1.075,c1=0.2782,c3=5.766e-3*2.0,d1=3,d3=6)[0,0])
    
    
    Js.append(J)
    
    # Update lambda value
    lam += lam_step

plt.figure(dpi=300)
plt.title("Employing a scaling factor $f$  to \n modify the constitutive function behavior")
plt.plot(Js, Sxxs_bir, label="f = 1",ls='--',color='k')
plt.plot(Js ,Sxxs_bir0, label="$f_{1} = 0.5 \,| \, f_{2}=0.5$")
plt.plot(Js ,Sxxs_bir1, label="$f_{1} = 0.5 \,| \, f_{2}=2.0$")
plt.plot(Js ,Sxxs_bir2, label="$f_{1} = 2.0 \,| \, f_{2}=0.5$")
plt.plot(Js ,Sxxs_bir3, label="$f_{1} = 2.0 \,| \, f_{2}=2.0$")

plt.text(1.5,7.0,'$W(\mathbf{C})=c(I_{1}-3)+c\slash\\beta(I_{3}^{-\\beta}-1)+$\n$ \qquad  \qquad c_{1}(I_{1} I_{3}^{-1\slash 3}-3)^{d_{1}} + $\n$\qquad \qquad  c_{3}(I_{3}^{1\slash 3}-1)^{d_{3}}$')
plt.text(1.5,6.25,'$c =f_{1} \cdot c$')
plt.text(3.5,6.25,'$c_{3}=f_{2} \cdot c_{3}$')


plt.xlabel("Jacobian (-)")
plt.ylim(0,10)
plt.xlim(1.0,10)
plt.ylabel("Stress (kPa)")
plt.grid(True,alpha=0.25)
plt.legend(bbox_to_anchor=(0.0,0.0,1.4,0.4))

plt.show()

# %% 

alpha = 1.00
lam = 1.0
lam_max = 10**(1/3)
lam_step = 0.01

Js = []
Sxxs_bir = []
Sxxs_bir0 = []
Sxxs_bir1 = []
Sxxs_bir2 = []
Sxxs_bir3 = []
Sxxs_bir4 = []

f0 = 0.5
f1 = 2
f2 = 4
f3 = 8
f4 = 16 


while lam < lam_max:
    
    # Build the deformation gradient tensor
    F=np.array([[ lam, 0., 0],
                [   0., lam, 0],
                [   0,    0,  lam]]) 
    
    # Pre-strain factor    
    F *= alpha 
    
    # Jacobian
    J=np.linalg.det(F)
    C=F.T*F
    
    # Evaluate models
    Se_bir = material_Birzle(C,c=1.5)
    Se_mod = material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*15,d1=3,d3=6)
    Se_mod = material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*15,d1=3,d3=6)
    Se_mod = material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*15,d1=3,d3=6)
    Se_mod = material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*15,d1=3,d3=6)
    
    # Extract relevant properties
    Sxxs_bir.append(Se_bir[0,0])
    Sxxs_bir0.append(material_Birzle(C,c=1.5,beta=1.075*f0,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    Sxxs_bir1.append(material_Birzle(C,c=1.5,beta=1.075*f1,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    Sxxs_bir2.append(material_Birzle(C,c=1.5,beta=1.075*f2,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    Sxxs_bir3.append(material_Birzle(C,c=1.5,beta=1.075*f3,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    Sxxs_bir4.append(material_Birzle(C,c=1.5,beta=1.075*f4,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    
    
    Js.append(J)
    
    # Update lambda value
    lam += lam_step

plt.figure(dpi=300)
plt.title("Employing a scaling factor $f$  to \n modify the constitutive function behavior")
plt.plot(Js, Sxxs_bir, label="f = 1",ls='--',color='k')
plt.plot(Js ,Sxxs_bir0, label="f = $1\slash2$")
plt.plot(Js ,Sxxs_bir1, label="f = 2")
plt.plot(Js ,Sxxs_bir2, label="f = 4")
plt.plot(Js ,Sxxs_bir3, label="f = 8")
plt.plot(Js ,Sxxs_bir3, label="f = 16")
plt.text(1.5,7.0,'$W(\mathbf{C})=c(I_{1}-3)+c\slash\\beta(I_{3}^{-\\beta}-1)+$\n$ \qquad  \qquad c_{1}(I_{1} I_{3}^{-1\slash 3}-3)^{d_{1}} + $\n$\qquad \qquad  c_{3}(I_{3}^{1\slash 3}-1)^{d_{3}}$')

plt.text(1.5,6.25,'$\\beta=f \cdot \\beta$')



plt.xlabel("Jacobian (-)")
plt.ylim(0,10)
plt.xlim(1.0,10)
plt.ylabel("Stress (kPa)")
plt.grid(True,alpha=0.25)
plt.legend(bbox_to_anchor=(0.0,0.0,1.23,0.5))

plt.show()

# %% 

alpha = 1.00
lam = 1.0
lam_max = 10**(1/3)
lam_step = 0.01

Js = []
Sxxs_bir = []
Sxxs_bir0 = []
Sxxs_bir1 = []
Sxxs_bir2 = []
Sxxs_bir3 = []
Sxxs_bir4 = []

f0 = 0.5
f1 = 2
f2 = 4
f3 = 8
f4 = 16 


while lam < lam_max:
    
    # Build the deformation gradient tensor
    F=np.array([[ lam, 0., 0],
                [   0., lam, 0],
                [   0,    0,  lam]]) 
    
    # Pre-strain factor    
    F *= alpha 
    
    # Jacobian
    J=np.linalg.det(F)
    C=F.T*F
    
    # Evaluate models
    Se_bir = material_Birzle(C,c=1.5)
    
    # Extract relevant properties
    Sxxs_bir.append(Se_bir[0,0])
    Sxxs_bir0.append(material_Birzle(C,c=1.5*f0,beta=1.075,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    Sxxs_bir1.append(material_Birzle(C,c=1.5*f1,beta=1.075,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    Sxxs_bir2.append(material_Birzle(C,c=1.5*f2,beta=1.075,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    Sxxs_bir3.append(material_Birzle(C,c=1.5*f3,beta=1.075,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    Sxxs_bir4.append(material_Birzle(C,c=1.5*f4,beta=1.075,c1=0.2782,c3=5.766e-3,d1=3,d3=6)[0,0])
    
    
    Js.append(J)
    
    # Update lambda value
    lam += lam_step

plt.figure(dpi=300)
plt.title("Employing a scaling factor $f$  to \n modify the constitutive function behavior")
plt.plot(Js, Sxxs_bir, label="f = 1",ls='--',color='k')
plt.plot(Js ,Sxxs_bir0, label="f = $1\slash2$")
plt.plot(Js ,Sxxs_bir1, label="f = 2")
plt.plot(Js ,Sxxs_bir2, label="f = 4")
plt.plot(Js ,Sxxs_bir3, label="f = 8")
plt.plot(Js ,Sxxs_bir3, label="f = 16")
plt.text(2.5,7.0,'$W(\mathbf{C})=c(I_{1}-3)+c\slash\\beta(I_{3}^{-\\beta}-1)+$\n$ \qquad  \qquad c_{1}(I_{1} I_{3}^{-1\slash 3}-3)^{d_{1}} + $\n$\qquad \qquad  c_{3}(I_{3}^{1\slash 3}-1)^{d_{3}}$')

plt.text(2.5,6.25,'$c=f \cdot c$')



plt.xlabel("Jacobian (-)")
plt.ylim(0,10)
plt.xlim(1.0,10)
plt.ylabel("Stress (kPa)")
plt.grid(True,alpha=0.25)
plt.legend(bbox_to_anchor=(0.0,0.0,1.23,0.5))

plt.show()

# %%

alpha = 1.00
lam = 1.0
lam_max = 10**(1/3)
lam_step = 0.01

Js = []
Sxxs_bir = []
Sxxs_bir0 = []
Sxxs_bir1 = []
f0 = 10.0
f1 = 100.0

while lam < lam_max:
    
    # Build the deformation gradient tensor
    F=np.array([[ lam, 0., 0],
                [   0., lam, 0],
                [   0,    0,  lam]]) 
    
    # Pre-strain factor    
    F *= alpha 
    
    # Jacobian
    J=np.linalg.det(F)
    C=F.T*F
    
    # Evaluate models
    Se_bir = material_Birzle(C,c=1.5)
    
    # Extract relevant properties
    Sxxs_bir.append(Se_bir[0,0])
    Sxxs_bir0.append(material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*f0,d1=3,d3=6)[0,0])
    Sxxs_bir1.append(material_Birzle(C,c=1.5,beta=1.075,c1=0.2782,c3=5.766e-3*f1,d1=3,d3=6)[0,0])

    Js.append(J)
    
    # Update lambda value
    lam += lam_step

plt.figure(dpi=300)
plt.title("Employing a scaling factor $f$  to \n modify the constitutive function behavior")
plt.plot(Js, Sxxs_bir, label="Original",ls='--',color='k')
plt.plot(Js ,Sxxs_bir0, label="Modified ($f=10^{1})$",color='tab:blue')
plt.plot(Js ,Sxxs_bir1, label="Modified ($f=10^{2})$",color='tab:orange')
plt.text(5.5,7.5,'$W(\mathbf{C})=c(I_{1}-3)+c\slash\\beta(I_{3}^{-\\beta}-1)+$\n$ \qquad  \qquad c_{1}(I_{1} I_{3}^{-1\slash 3}-3)^{d_{1}} + $\n$\qquad \qquad  f \cdot c_{3}(I_{3}^{1\slash 3}-1)^{d_{3}}$')

#plt.text(2.5,6.25,'$c=f \cdot c$')



plt.xlabel("Jacobian (-)")
plt.ylim(0,10)
plt.xlim(1.0,10)
plt.ylabel("Stress (kPa)")
plt.grid(True,alpha=0.25)
plt.legend(bbox_to_anchor=(0.0,0.0,1.40,0.5))

plt.show()
