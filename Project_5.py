# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 21:53:21 2022

@author: HP
"""

import numpy as np
import math
import copy
import random
import sys
import matplotlib.pyplot as plt
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import statistics
import pandas as pd

kbt = 8.6167e-5       
Na = 6.02214086e+23         

mass = 39.95 
sigma = 3.405               # equl radius

T = 94.4                      # Temperature 
Beta_val = kbt * T


L_original = 10.229*sigma

d = L_original/(864**(1/3))
print (d)
print (L_original)
Natms = 8
N_periodic = Natms**3  # No of atoms in PBC
N = N_periodic
print (N_periodic, np.int(np.rint(Natms**3)))
box = Natms*d
L = box                # Length of PBC box for the desired No. of atoms
rc = box/2             # radius cutoff
print (rc)


""" Generating coordinates in PBC """
def p_lattice(Natms):
    N_periodic = Natms**3
    box = Natms*d
    pos = np.zeros((N_periodic,3))
    count = 0
    for i1 in range(Natms):
        for i2 in range(Natms):
            for i3 in range(Natms):
                pos[count] = d*i1, d*i2, d*i3
                count += 1
    return pos

# initial pos in fcc lattice
def InitPositions(N, L):

    Pos = np.zeros((N,3), float)
    Lat = int(N**(1./3.) + 1.)     # Lattice points
    r = L * (np.arange(Lat, dtype=float)/Lat - 0.5)
    i = 0
    for x in r:
        for y in r:
            for z in r:
                Pos[i] = np.array([x,y,z], float)
                i += 1
                if i >= N:
                    return Pos
    return Pos


""" Calculating LJ Potential for the complete system"""
@njit
def LJ_pot(N_periodic,coord,L,rc, bins):
    eps = 0.0103  
    sigma = 3.405 
    #print (eps, sigma)
    N = N_periodic
    #F = np.zeros(coord.shape)
    U = 0.0

    V = L**3            # volume of the box (A^3)
    dr = L / bins     # thickness
    #rs = np.zeros(bins)
    rs = []
    rs = [dr*i for i in range(0, bins)]
    rs = np.array(rs)
    #for i in range(0, bins):
    #    rs[i] = dr * i
    h = np.zeros(bins)
    
    for i in range(0, (N-1)):
        for j in range(i+1, N):
            r_ij = coord[i,:] - coord[j,:] 
#            r_ij = coord[i] - coord[j]
            r_ij = r_ij - ( (np.rint(r_ij/L)) * L ) # PBC minimizing image
            #r = np.sqrt(np.sum(r_ij**2))
            r = np.linalg.norm(r_ij)
            idx = int(r / dr)
            h[idx] += 1            
            if (r**2 < rc**2):
                #LJ_F = 4.0 * eps * (( (12 * np.power(sigma, 12)) / (np.power(r, 14))) - 
                #                    ( (6*np.power(sigma, 6)) / (np.power(r, 8))))
                ecut = 4 * eps * ( ((sigma/rc)**12) - ((sigma/rc)**6) )
                PP = 4.0 * eps * ( ((sigma/r)**12) - ((sigma/r)**6) )
                PP = PP - ecut
                #F_new = LJ_F * (r_ij)
                U = U + PP
                #F[i,:] = F[i,:] + F_new 
                #F[j,:] = F[j,:] - F_new
                #idx = int(r / dr)
                #h[idx] += 1

    h *= 2 / N
    
    g = (V / N) * (1./ (4. * np.pi * dr * rs)) * (h/rs)
                
    return U, g, rs


""" Calculating LJ Potential for a single particle """
@njit
def LJ_pot_particle(atom, N_periodic,coord,L,rc,bins):
    eps = 0.0103  
    sigma = 3.405 
    #print (eps, sigma)
    N = N_periodic
    #F = np.zeros(coord.shape)
    U = 0.0

    V = L**3            # volume of the box (A^3)
    dr = L / bins     # thickness
    #rs = np.zeros(bins)
    rs = []
    rs = [dr*i for i in range(0, bins)]
    rs = np.array(rs)
    #for i in range(0, bins):
    #    rs[i] = dr * i
    h = np.zeros(bins)

    for i in range(0, (N-1)):
        if i != atom:                      # atom is the index of the selected random  part.
            r_ij = coord[atom,:] - coord[i,:]
#            r_ij = coord[atom] - coord[i]     
            r_ij = r_ij - ( (np.rint(r_ij/L)) * L ) # PBC minimizing image
            #r = np.sqrt(np.sum(r_ij**2))
            r = np.linalg.norm(r_ij)          
            idx = int(r / dr)
            h[idx] += 1            
            if (r**2 < rc**2):
                #LJ_F = 4.0 * eps * (( (12 * np.power(sigma, 12)) / (np.power(r, 14))) - 
                #                    ( (6*np.power(sigma, 6)) / (np.power(r, 8))))
                ecut = 4 * eps * ( ((sigma/rc)**12) - ((sigma/rc)**6) )
                PP = 4.0 * eps * ( ((sigma/r)**12) - ((sigma/r)**6) )
                PP = PP - ecut
                U = U + PP

    h *= 2 / N
    
    g = (V / N) * (1./ (4. * np.pi * dr * rs)) * (h/rs)

    return U, g, rs



""" Monte Carlo Approach """
def MC_metropolis(N, coords, L , rc, N_moves, Beta_val):
    Delta = 0.2
    move_accept = 0
    move_reject = 0
    Gs = []
    PEs = []
    step = 0
    
   
    total_energy, initial_g, rs = LJ_pot(N, coords, L, rc, bins)
    Gs.append(initial_g)
    while step <= N_moves:
        
        old_coords = copy.deepcopy(coords)
        atom = random.randint(0, N-1)
 
        PE_ini,current_g,current_rs = LJ_pot_particle(atom, N, coords, L, rc,bins)  

        coords[atom,0] += Delta * (random.uniform(0,1) - 0.5)
        coords[atom,1] += Delta * (random.uniform(0,1) - 0.5)
        coords[atom,2] += Delta * (random.uniform(0,1) - 0.5)

        coords[atom,:] %= L
        coords = coords - (np.floor(coords/L)) * L
    
 
        PE_fni,neww_g,new_rs = LJ_pot_particle(atom, N, coords, L, rc,bins)  

        Delta_U = PE_fni - PE_ini

        accept = False
        if Delta_U < 0:
            accept = True
        else:
            phi = math.exp(-(Delta_U / Beta_val ))
            if (phi >= random.uniform(0.,1.)):
                accept = True
            else:
                accept = False
        
        if accept == True:
            move_accept += 1
            total_energy += Delta_U
            PEs.append(total_energy)
            #Gs.append(neww_g)
            #coords = coords
        else:
            move_reject += 1
            coords = old_coords 
        
        if step % 50000 == 0:            
            total_energy, new_g, rs = LJ_pot(N, coords, L, rc, bins) 
            #PEs.append(total_energy)
            Gs.append(new_g)  
        
        if step % 10000 == 0:
            print (step, total_energy, move_accept, move_reject, (move_accept/(move_accept+move_reject)))

        
        step += 1
        
    PEs = np.array(PEs)
    Gs = np.array(Gs)
    print ('CHk',move_accept, len(PEs), len(Gs))
    return PEs, Gs, coords, move_accept, move_reject

# Generating coordinates in lattice
coords = InitPositions(N, L)


step = 0
Gs = []
PEs = []

bins = 100
total_moves = 30e5
PEs, Gs, coords, move_accept, move_reject = MC_metropolis(N, coords, L, rc, total_moves, Beta_val)

new_energy, new_RDF, rs = LJ_pot(N, coords, L, rc, bins)

DATA_MD = pd.DataFrame(PEs,columns= ['KE'])
Gs = np.nan_to_num(np.array(Gs))
RDF_data = pd.DataFrame(Gs)
#print (DATA_MD)
#print ()
MD_data = np.array(DATA_MD)
#print (MD_data)
#np.savetxt(r'RDF.txt', RDF_data.values, fmt='% 0.12f')
#print (RDF_data[:3])
np.savetxt('RDF.txt', Gs, fmt='% 0.12f')
np.savetxt('MD_data.txt', MD_data, fmt='% 0.12f')
np.savetxt('rs.txt', rs, fmt='% 0.12f')

plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams['figure.dpi'] = 100

x_axis = np.arange(0, len(PEs))
plt.plot(x_axis,PEs, 'g--', label = "U")
plt.xlabel('Steps')
plt.ylabel('PE (eV / $\AA$)')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.savefig('PE_MC.pdf', dpi = 300, bbox_inches="tight")
plt.show()