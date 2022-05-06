# -*- coding: utf-8 -*-
"""
Simple Wall & Radiator with Thermostatic Radiator Valve

Produced on Fri Feb  4 10:40:05 2022

@author: Hakan İbrahim Tol, PhD
"""

""" Libraries """
import numpy as np
import math
import matplotlib.pyplot as plt

delta_t=10          # [s]       time step for all
Nt=7500            # [-]       number of time steps
T_init=15           # [°C]      initial temperature 

""" Input Parameters - Building Envelope """
# Wall
height_w = 3        # [m]       height of the wall
wide_w = 5          # [m]       wide of the wall
thick_w = 0.3       # [m]       thickness of the wall
volume_w = height_w * wide_w * thick_w  # [m3] volume of the wall
area_w= height_w * wide_w               # [m2] wall area

# Room
deep_r = 4          # [m]       deepness of room
volume_r = height_w * wide_w * deep_r # [m3] volume of the room air

# Initial Temperature
T_in = 20           # [°C]      indoor temperature
T_out = 2           # [°C]      outdoor temperature

# Thermal
h_in = 7.7          # [W/m2K]   convective heat transfer coefficient (indoor)
h_out = 25          # [W/m2K]   convective heat transfer coefficient (outdoor)

# see: Balaji NC et al. Thermal performance of the building walls
k_w = 0.811         # [W/mK]    thermal conductivity (brick)
rho_w = 1820        # [kg/m3]   density (brick)
Cp_w = 880          # [J/kgK]   specific heat capacity (brick)

# Air
rho_a = 1.204       # [kg/m3]   density (air at 20 °C)
Cp_a = 1006         # [J/kgK]   specific heat capacity (air at 20 °C)

""" Input Parameters - Radiator """
# Input for Operational Parameters
mF=0.1          # [kg/s]    mass flow rate
Ts=90           # [°C]      radiator inlet temperature
Ta=20           # [°C]      air temperature - assumed to be constant

Ti=20           # [°C]      initial temperature of the radiator water mass

# Radiator Properties (Panel Radiator Model: Lenhovda MP 25 500)
h_rad=0.5       # [m]       radiator height
l_rad=1         # [m]       radiator length
n_r=1.286       # [-]       emprical radiator exponent

# Nominal 
Qn=276          # [W]       nominal heat output rate
LMTDn=30        # [°C]      nominal LMTD

# Mass properties
Mw=3.23         # [kg]      total water mass in the radiator
Mm=10.71        # [kg]      total metal mass of the radiator unit

# Physical Properties
Cw=4180         # [J/kgK]   specific heat capacity of water  
Cm=897          # [J/kgK]   specific heat capacity of metal

# Numerical Parameters
n=5             # [-]       number of elements
dx=l_rad/n      # [m]       uniform grid spacing (mesh step)

Qi=Qn/n         # [W]       nominal heat output rate per element

""" Input Parameters - TRV """
K_sen=500
tau_s=1/K_sen
Xk=-9.8027
Yk=125.49

""" Preliminary Calculations - Building Envelope """

C_a = rho_a * Cp_a * volume_r   # [J/K]     heat capacity of indoor air
C_w = rho_w * Cp_w * volume_w   # [J/K]     heat capacity of wall

R_in = 1 / (h_in * area_w)      # [K/W]     thermal resistance (indoor air)
R_out = 1 / (h_out * area_w)    # [K/W]     thermal resistance (outdoor air)

R_w = thick_w / (k_w * area_w)  # [K/W]     overall thermal resistance (wall)

""" Preliminary Calculations - Radiator """
Crad=(Mw*Cw+Mm*Cm)/n # [J/K] Heat capacity per element (water-metal)

D_t=Crad/delta_t
D_c=Cw*mF
D_q=Qi

Q_rad=10

""" Numerical Calculation (T_new = A x T_pre + B) """
# Solution Matrix - A
A = np.zeros((5, 5))

A[0][0]=1-delta_t/(C_a*R_in)
A[0][1]=delta_t/(C_a*R_in)

A[1][0]=4*delta_t/(C_w*R_in)
A[1][1]=1-(4*delta_t/C_w)*(1/R_in+2/R_w)
A[1][2]=8*delta_t/(C_w*R_w)

A[2][1]=4*delta_t/(C_w*R_w)
A[2][2]=1-(2*delta_t/C_w)*(2/R_w+2/R_w)
A[2][3]=4*delta_t/(C_w*R_w)

A[3][2]=8*delta_t/(C_w*R_w)
A[3][3]=1-(4*delta_t/C_w)*(2/R_w+1/R_out)

A[4][0]=1-math.exp(-delta_t/tau_s)
A[4][4]=math.exp(-delta_t/tau_s)

# Solution Matrix - B
B = np.zeros((5,1))
B[0][0]=(delta_t/C_a)*Q_rad
B[3][0]=4*delta_t*T_out/(C_w*R_out)

""" Fasten Your Seat Belts (Code Runs Here) """
# Initialize the solution space - Building Envelope
T_initial=np.ones((5,1))*T_init
T_w=np.ones((5,1))*T_init

# Initialize the solution space - Radiator
T_r=np.zeros((n+1,Nt+1))
T_r[:,0]=np.ones(n+1)*(T_init+2)    # initial condition
T_r[0,:]=Ts       # boundary condition - Dirichlet

Q_rN=np.zeros((n,Nt+1))

mF_d=mF

for iT in range(1,Nt):
    # building envelope
    T=np.matmul(A,T_initial)+B
    
    T_w=np.append(T_w,T,axis=1)
    
    T_initial=T
    
    # radiator
    for iS in range(1,n+1):
        
        T_r[iS,iT]=D_c/D_t*(T_r[iS-1,iT-1]-T_r[iS,iT-1])-D_q/D_t*((T_r[iS,iT-1]-T[0,0])/LMTDn)**n_r+T_r[iS,iT-1]
        
        Q_rN[iS-1,iT-1]=Qi*((T_r[iS,iT-1]-T[0,0])/LMTDn)**n_r
    
    Q_rad=np.append(Q_rad,np.sum(Q_rN[:,iT-1]))
    
    B[0][0]=(delta_t/C_a)*Q_rad[iT]
    
    # TRV
    Delta_T_s=20-T[0,0]
    
    if Delta_T_s>0:
    
        mF=min(0.1,(Xk*Delta_T_s**2+Yk*Delta_T_s)/3600) # [kg/s]
        
    else:
        
        mF=0
        
    mF_d=np.append(mF_d,mF)

    D_c=Cw*mF

""" Plotting the Results """
t_plot=np.arange(0,delta_t*Nt,delta_t) # time between 0 and 80 min

fig, (ax1, ax2, ax3) = plt.subplots(3)

# indoor temperature
ax1.plot(t_plot,T_w[0,:])

# radiator - Temperature Degrees
for i in range(n+1):
    if i==0:
        ax2.plot(t_plot,T_r[i,1:],linewidth=1,label='Tinlet')
    else:
        ax2.plot(t_plot,T_r[i,1:],linewidth=1,label='N='+str(i))

ax3.plot(t_plot,mF_d)
