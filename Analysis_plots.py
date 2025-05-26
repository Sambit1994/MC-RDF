# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:05:53 2022

@author: HP
"""

import numpy as np
import math
import copy
import random
import sys
import matplotlib.pyplot as plt
#from numba import njit
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import statistics
import pandas as pd

####
## Constants and units
####

Boltzmann = 8.6167e-5       # eV
Na = 6.02214086e+23         # Avogadro's number
N = 512#864 #512 #864                     # Number of particles
mass = 39.95 #* 1.66054e-27  # Argon atomic mass unit (amu)
sigma = 3.405               # van der Waals radius (A)
L = 10.229 * sigma          # Lenght of the cubic box side
T = 94.4                      # Temperature of the exp. (K)
rc = L * 0.5                # cut-off radius
beta = Boltzmann * T
max_it = 30000              # Set the number of Monte Carlo iterations to perform

###
## Initialization
####


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

N_av=6.023e23
#mass = (mass/N_av)*N_periodic

#steps = 20000

Gs = np.loadtxt('RDF.txt')
rs = np.loadtxt('rs.txt')
MD_data = np.loadtxt('MD_data.txt')
print (len(MD_data), MD_data)
#print (MD_data[:,1])
MD_data = MD_data[0:(int((3*(len(MD_data))/5)-50000) )]
#KEs = np.array(MD_data[:,0])
#PEs = np.array(MD_data[:,0])
PEs = np.array(MD_data)
#TEs = np.array(MD_data[:,2])
#Temps_arr = np.array(MD_data[:,3])

plt.rcParams['figure.figsize'] = [6,6]
fig, (ax1) = plt.subplots()

ax1.set_title('Potential Energy Monte Carlo')
ax1.plot(PEs, label = "PE")
#ax1.plot(PEs, label = "PE")
#ax1.plot(TEs, label = "TE")
ax1.set_ylabel('Calculated PEs (eV)')
ax1.legend(loc='center right')
#ax1.xaxis.set_visible(False)

ax1.set_xlabel('Steps')
#ax2.set_ylabel('Calculated Temperature (K)')
ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

fig.tight_layout()
plt.savefig('PE_MC.svg', dpi=500)
plt.savefig('PE_MC.pdf', dpi=500)
#plt.savefig('Eng_Temp_vs_time_n4_dt.01.svg', dpi=500)
#plt.savefig('Eng_Temp_vs_time_n4_dt.01.pdf', dpi=500)
plt.show()

N_points = int(len(MD_data))
equl_range = int(N_points*(4/5))
print (equl_range)

print ('no. of atoms',N, Boltzmann, T)
PE_equl = (PEs[equl_range:-1])/N
#PE_equl = (PEs)/N
PE_equl_2 = (PE_equl - np.mean(PE_equl))**2
sigma2_PE = np.mean(PE_equl_2)

ev_to_J = 1.602176565e-19
N_av = 6.023e23

Cv = (1.5*Boltzmann) + ((N*sigma2_PE)/(Boltzmann*(T**2)))

print ('new Cv', Cv, (Cv*ev_to_J), ((Cv*ev_to_J)*N_av))


#N = N_periodic
#gs_equl_range= int(len())
Gs_data_equl = Gs[30:-1]

rdf_mean = np.mean((Gs_data_equl), axis=0)

plt.rcParams['figure.figsize'] = [6,6]
fig, (bx) = plt.subplots()

bx.set_title('RDF Monte Carlo')
bx.plot((rs/sigma), rdf_mean, color='red')
bx.set_xlabel('r/$\sigma$ ($\AA$)')
bx.set_ylabel('g(r)')
bx.legend(loc='center right')

bx.grid()
#plt.xlim(0,(0.5*(L/sigma)))
fig.tight_layout()
plt.savefig('RDF_MC.svg', dpi=500)
plt.savefig('RDF_MC.pdf', dpi=500)
plt.show()

def Block_Average(Kps, sample_data):

    K = np.zeros((len(Kps)))
    #print (K)
    for i in range(len(Kps)):
        K[i] = int(2**(Kps[i]))
        #K[i] = int((Kps[i])+1)

    #mean_range = int((len(sample_data)/2)-1) 
    mean_range = int((len(sample_data)*(3/5))-1) 
    #mean_range = int(0)#int((len(sample_data)))
    #data_range = int((len(sample_data)*(1/5))-1) 
    data_range = int(0)#int((len(sample_data)))

    #print (mean_range, data_range)

    A_av = np.mean(sample_data[mean_range:-1])
    
    print (mean_range, data_range, A_av)
    
    sample_data = sample_data[data_range:-1]
        
    Block_min = K[0]
    Block_max = K[-1]
    
    N_data_pts = len(sample_data)   # No. of data points considered for statistics
    
    print (N_data_pts, Block_min, Block_max, A_av)
    
    N_Blocks_arr = len(K)

    Mean_arr   = np.zeros(N_Blocks_arr) #Mean (Array elements for each block size)
    Var_arr    = np.zeros(N_Blocks_arr) #Variance (Array elements for each block size)
    Std_arr    = np.zeros(N_Blocks_arr) #Standard error (Array elements for each block size)
    U_err_bar  = np.zeros(N_Blocks_arr) #Upper error (Array elements for each block size)
    L_err_bar  = np.zeros(N_Blocks_arr) #Lower error (Array elements for each block size)

    nblk_size  = 0
    
    for block_size in range(len(K)):
        
        N_blk_elemets    = int(np.floor(N_data_pts/K[block_size])) #nb as per compendium
        Blk_data   = np.zeros(N_blk_elemets)
        print (Kps[block_size], K[block_size], len(Blk_data), (K[block_size]*len(Blk_data)))
        
        sample_i = int(0)
        summ = 0
        for i in range(N_blk_elemets):
            sample_f =  int(sample_i + K[block_size])    
            #print (sample_i, sample_f)
            data_arr = sample_data[sample_i:sample_f]  
            #if i == int((N_blk_elemets)-1) and i > 0 :
            #    data_arr = sample_data[sample_i:-1]
            
            summ = summ + len(data_arr)
            #print (len(data_arr))    
            Blk_data[i] = np.mean(data_arr)            
            sample_i = int(sample_f)

        print ('total',summ)
        Mean_arr[nblk_size] = np.mean(Blk_data)   
        
        #Variance with method 1
        val1 = (Blk_data - A_av)**2
        var = np.mean(val1)#np.var(Blk_data)
        
        #Variance with method 2
        #var = np.var(Blk_data)
        Var_arr[nblk_size]   = var/(N_blk_elemets-1)   
        Std_arr[nblk_size]   = np.sqrt(Var_arr[nblk_size])
        U_err_bar[nblk_size] = (np.sqrt(Var_arr[nblk_size]))*(1+(1/(np.sqrt(2*(N_blk_elemets-1)))))
        L_err_bar[nblk_size] = (np.sqrt(Var_arr[nblk_size]))*(1-(1/(np.sqrt(2*(N_blk_elemets-1)))))
        nblk_size += 1

    #print (L_err_bar)    
    #print ()    
    #print (U_err_bar)
    #asymm_err_bar = [L_err_bar, U_err_bar]    
    asymm_err_bar = np.array(list(zip(L_err_bar, U_err_bar))).T    
    #print (asymm_err_bar[0])
    print (Mean_arr)
    print (Std_arr)
    print (asymm_err_bar)
 
    return Mean_arr, Var_arr, Std_arr, asymm_err_bar

Kps = np.arange(0,20)

Mean_arr, Var_arr, Std_arr, asymm_err_bar = Block_Average(Kps, PEs)
#print (asymm_err_bar)

plt.rcParams['figure.figsize'] = [6,3]
fig, (ax) = plt.subplots()

ax.set_title('Standard deviation in determination of PE from MC')
ax.plot(Kps, asymm_err_bar[0], '.', color = 'red')
ax.plot(Kps, asymm_err_bar[1], '.', color = 'red')
ax.errorbar(Kps, Std_arr, yerr=np.sqrt(Var_arr), color = 'orange')
ax.plot(Kps, Std_arr, 'o', color = 'blue')
ax.set_ylabel('$\sigma$' r'($\bar{A}$)')
ax.set_xlabel('No. of applied Block Transformations')
ax.ticklabel_format(axis="x")

fig.tight_layout()
plt.savefig('Std_dev_PE.svg', dpi=500)
plt.savefig('Std_dev_PE.pdf', dpi=500)
plt.show()