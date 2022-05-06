# -*- coding: utf-8 -*-
"""
Simple Case with Multi-Layer Wall without Radiator & Thermostatic Radiator Valve
Produced on Thu Apr 21 09:31:29 2022

@author: Hakan İbrahim Tol, PhD
"""

""" Libraries """
import numpy as np
import math
import matplotlib.pyplot as plt

""" Test Case """
T_ini=20            # [°C]      indoor temperature
T_out=2             # [°C]      outdoor temperature

""" Numerical Parameters """
delta_t=10          # [s]       time step for all
Nt=7500             # [-]       number of time steps

""" Input Parameters - Building Envelope """
# Room
height_r = 3        # [m]       height of the room
wide_r = 5          # [m]       wide of the room
deep_r = 4          # [m]       deepness of room

volume_r = height_r * wide_r * deep_r   # [m3] volume of the room air
area_w = height_r * wide_r              # [m2] area of the wall

# Air Properties
rh_a = 1.204        # [kg/m3]   density (air at 20 °C)
Cp_a = 1006         # [J/kgK]   specific heat capacity (air at 20 °C)

# Thermal
h_in = 7.7          # [W/m2K]   convective heat transfer coefficient (indoor)
h_out = 25          # [W/m2K]   convective heat transfer coefficient (outdoor)

""" Wall Layers """ # See: Balaji NC et al. Thermal performance of the building walls
# Interior Cement Plastering
th_icp = 0.03      # [m]       thickness 
tc_icp = 0.721     # [W/mK]    thermal conductivity 
rh_icp = 1762      # [kg/m3]   density 
Cp_icp = 840       # [J/kgK]   specific heat capacity 

# Brick Wall
th_bwa = 0.19      # [m]       thickness 
tc_bwa = 0.811     # [W/mK]    thermal conductivity 
rh_bwa = 1820      # [kg/m3]   density 
Cp_bwa = 880       # [J/kgK]   specific heat capacity

# Insulation (EPS)
th_ins = 0.06      # [m]       thickness 
tc_ins = 0.035     # [W/mK]    thermal conductivity 
rh_ins = 24        # [kg/m3]   density 
Cp_ins = 1340      # [J/kgK]   specific heat capacity

# Exterior Cement Plastering
th_ecp = 0.02      # [m]       thickness 
tc_ecp = 0.721     # [W/mK]    thermal conductivity 
rh_ecp = 1762      # [kg/m3]   density 
Cp_ecp = 840       # [J/kgK]   specific heat capacity 

""" Preliminary Calculations """
# Volume of Layers [m3]
volume_icp = th_icp * area_w
volume_bwa = th_bwa * area_w
volume_ins = th_ins * area_w
volume_ecp = th_ecp * area_w

# Heat Capacity [J/K]
C_in = rh_a * Cp_a * volume_r            # [J/K]     heat capacity of indoor air
C_icp = rh_icp * Cp_icp * volume_icp     # [J/K]     heat capacity of interior cement plaster
C_bwa = rh_bwa * Cp_bwa * volume_bwa     # [J/K]     heat capacity of brick wall
C_ins = rh_ins * Cp_ins * volume_ins     # [J/K]     heat capacity of insulation
C_ecp = rh_ecp * Cp_ecp * volume_ecp     # [J/K]     heat capacity of exterior cement plaster

# Thermal Resistance [K/W]
R_in = 1 / (h_in * area_w)              # [K/W]     thermal resistance (indoor air)
R_out = 1 / (h_out * area_w)            # [K/W]     thermal resistance (outdoor air)

R_icp = th_icp / (tc_icp * area_w)      # [K/W]     thermal resistance for interior cement plaster
R_bwa = th_bwa / (tc_bwa * area_w)      # [K/W]     thermal resistance for brick wall
R_ins = th_ins / (tc_ins * area_w)      # [K/W]     thermal resistance for insualation
R_ecp = th_ecp / (tc_ecp * area_w)      # [K/W]     thermal resistance for exterior cement plaster

""" Numerical Calculation (T_new = A x T_pre + B) """

T_w=np.ones((7,1))*T_ini

# Solution Matrix - A
A = np.zeros( (7, 7) )

A[0][0]=1-delta_t/(C_in*(R_in+R_icp/2))
A[0][1]=delta_t/(C_in*(R_in+R_icp/2))

A[1][0]=delta_t/(C_icp*(R_in+R_icp/2))
A[1][1]=1-delta_t/C_icp*(1/(R_in+R_icp/2)+2/R_icp)
A[1][2]=2*delta_t/(C_icp*R_icp)

A[2][1]=8*delta_t/(C_bwa*R_icp)
A[2][2]=1-4*delta_t/C_bwa*(2/R_icp+2/R_bwa)
A[2][3]=8*delta_t/(C_bwa*R_bwa)

A[3][2]=4*delta_t/(C_bwa*R_bwa)
A[3][3]=1-8*delta_t/(C_bwa*R_bwa)
A[3][4]=4*delta_t/(C_bwa*R_bwa)

A[4][3]=8*delta_t/(C_bwa*R_bwa)
A[4][4]=1-8*delta_t/C_bwa*(1/R_bwa+1/R_ins)
A[4][5]=8*delta_t/(C_bwa*R_ins)

A[5][4]=2*delta_t/(C_ins*R_ins)
A[5][5]=1-delta_t/C_ins*(2/R_ins+1/(R_ins/2+R_ecp/2))
A[5][6]=delta_t/(C_ins*(R_ins/2+R_ecp/2))

A[6][5]=delta_t/(C_ecp*(R_ins/2+R_ecp/2))
A[6][6]=1-delta_t/C_ecp*(1/(R_ins/2+R_ecp/2)+1/(R_ecp/2+R_out))

# Solution Matrix - B
B = np.zeros((7,1))

B[6][0]=delta_t*T_out/C_ecp*(1/(R_ecp/2+R_out))

""" Fasten Your Seat Belts (Code Runs Here) """
T_init=np.ones((7,1))*T_ini

for i in range(1,Nt):
    
    T=np.matmul(A,T_init)+B
    
    T_init=T
    
    T_w=np.append(T_w,T,axis=1)
    
t_plot=np.arange(0,delta_t*Nt,delta_t) # time between 0 and 80 min

# indoor temperature
plt.plot(t_plot,T_w[0,:],t_plot,T_w[1,:],t_plot,T_w[2,:],t_plot,T_w[3,:],t_plot,T_w[4,:],t_plot,T_w[5,:],t_plot,T_w[6,:])

































































