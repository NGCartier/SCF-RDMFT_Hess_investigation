#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:16:29 2022

@author: nicolascartier
"""
import numpy as np
import pyscf
from  pyscf import gto, scf, cc, ci, mp


import Compute_1RDM
from Interface import rdm_guess_CISD, rdm_guess, compute_1RDM

#Definition of a list of molecules for PySCF
H2 = gto.M(atom = [['H', (-1.5,0,0)], ['H',(1.5,0,0)]], basis ='sto3g', spin = 0)
H2_pvdz = gto.M(atom = [['H', (-1.5,0,0)], ['H',(1.5,0,0)]], basis ='cc-pvdz', spin = 0)
H2_pvtz = gto.M(atom = [['H', (-1.5,0,0)], ['H',(1.5,0,0)]], basis ='cc-pvtz', spin = 0)
H4 = gto.M(atom = [['H', (-1.5,0,0)], ['H',(1.5,0,0)],
                   ['H', (0,-1.5,0)], ['H',(0,1.5,0)]], basis ='sto3g', spin = 0)
H2.unit = 'Bohr' ; 
He = gto.M(atom = [['He',(0,0,0)]], basis ='sto3g')

Be = gto.M(atom=[['Be',(0,0,0)]], basis = 'sto3g')
Be_631gs = gto.M(atom=[['Be',(0,0,0)]], basis = '631g*')
Be_pvdz = gto.M(atom=[['Be',(0,0,0)]], basis = 'cc-pvdz')
Be_pvtz = gto.M(atom=[['Be',(0,0,0)]], basis = 'cc-pvtz')
Be_pv5z = gto.M(atom=[['Be',(0,0,0)]], basis = 'cc-pv5z')

alpha = 52.22*np.pi/180 ; a=1.811
H2O = gto.M(atom = [['O', (0,0,0)],['H',(a*np.cos(alpha),a*np.sin(alpha),0)],
                    ['H',(-a*np.cos(alpha),a*np.sin(alpha),0)]], basis = 'sto3g')
H2O_631gs = gto.M(atom = [['O', (0,0,0)],['H',(a*np.cos(alpha),a*np.sin(alpha),0)],
                    ['H',(-a*np.cos(alpha),a*np.sin(alpha),0)]], basis = '6-31gs',
                      unit='Bohr')
H2O_pvdz = gto.M(atom = [['O', (0,0,0)],['H',(a*np.cos(alpha),np.sin(alpha),0)],
                    ['H',(-a*np.cos(alpha),np.sin(alpha),0)]], basis = 'cc-pvdz',
                     unit='Bohr')
H2O_pvtz = gto.M(atom = [['O', (0,0,0)],['H',(a*np.cos(alpha),np.sin(alpha),0)],
                    ['H',(-a*np.cos(alpha),a*np.sin(alpha),0)]], basis = 'cc-pvtz',
                     unit='Bohr')

H_He = gto.M(atom = [['H', (-1/2,0,0)], ['He', (1/2,0,0)]], basis = 'sto3g', spin=1)

CH4 = gto.M(atom  = [['H', (1,0,-1/np.sqrt(2))],['H',(-1,0,-1/np.sqrt(2))],
                     ['H',(0,1,1/np.sqrt(2))], ['H',(0,-1,1/np.sqrt(2))],
                     ['C',(0,0, 0)]], basis= 'sto3g')

CH4_pvdz = gto.M(atom  = [['H', (1,0,-1/np.sqrt(2))],['H',(-1,0,-1/np.sqrt(2))],
                     ['H',(0,1,1/np.sqrt(2))], ['H',(0,-1,1/np.sqrt(2))],
                     ['C',(0,0, 0)]], basis= 'cc-pvdz')

O2 = gto.M(atom = [['O', (-1,0,0)], ['O', (1,0,0)]], basis = 'sto3g')

N2 = gto.M(atom = [['N', (-0.55,0,0)], ['N',(0.54,0,0)]], spin = 0)
HF = gto.M(atom = [['H', (-0.46,0,0)], ['F',(0.45,0,0)]], spin = 0)

HF_pvtz = gto.M(atom = [['H',(0,0,1.73)],['F',(0,0,0)]],basis = 'cc-pvtz',
                     unit='Bohr')

N2_pvtz = gto.M(atom = [['N',(0,0,-1.037)],['N',(0,0,1.037)]],basis = 'cc-pvtz',
                     unit='Bohr')

CH3OH_pvdz = gto.M(atom = [['C',(-0.0461,0.6544,0)],     ['O',(-0.0461,-0.7454,0)],
                          ['H',(-1.0790,0.9733,0)],   ['H',(.4391,1.0621,0.8836)],
                          ['H',(0.4391,1.0621,-0.8836)],['H',(0.8462,-1.0606,0)]],
                     basis='cc-pvdz');

a = np.cos(111.2*np.pi/180) ; d = 0.76
C2H6 = gto.M(atom  = [['H', (-d+a,a,0)],['H',(-d,a,a)],
                     ['H',(-d+a,0,a)], ['C',(-d,0,0)], 
                     ['H', (d+a,a,0)],['H',(d,a,a)],
                     ['H',(d+a,0,a)], ['C', (d,0,0)]], basis= 'sto3g')
C2H6_pvdz = gto.M(atom  = [['H', (-d+a,a,0)],['H',(-d,a,a)],
                     ['H',(-d+a,0,a)], ['C',(-d,0,0)], 
                     ['H', (d+a,a,0)],['H',(d,a,a)],
                     ['H',(d+a,0,a)], ['C', (d,0,0)]], basis= 'cc-pvdz')

C2H6O = gto.M(atom =     [['C' ,     ( 0.01247000,       0.02254000,       1.08262000)]
                         ,['C',      ( -0.00894000,     -0.01624000,      -0.43421000)]
                         ,['H',      (-0.49334000,       0.93505000,       1.44716000)]
                         ,['H',      (1.05522000,        0.04512000,       1.44808000)]
                         ,['H',      (-0.64695000,      -1.12346000,       2.54219000)]
                         ,['H',      ( 0.50112000,      -0.91640000,      -0.80440000)]
                         ,['H',      ( 0.49999000,       0.86726000,      -0.84481000)]
                         ,['H',      (-1.04310000,      -0.02739000,      -0.80544000)]
                         ,['O',      (-0.66442000,      -1.15471000,       1.56909000)]]
                         , basis = 'sto3g')
C2H6O_pvdz = gto.M(atom =[['C' ,     ( 0.01247000,       0.02254000,       1.08262000)]
                         ,['C',      ( -0.00894000,     -0.01624000,      -0.43421000)]
                         ,['H',      (-0.49334000,       0.93505000,       1.44716000)]
                         ,['H',      (1.05522000,        0.04512000,       1.44808000)]
                         ,['H',      (-0.64695000,      -1.12346000,       2.54219000)]
                         ,['H',      ( 0.50112000,      -0.91640000,      -0.80440000)]
                         ,['H',      ( 0.49999000,       0.86726000,      -0.84481000)]
                         ,['H',      (-1.04310000,      -0.02739000,      -0.80544000)]
                         ,['O',      (-0.66442000,      -1.15471000,       1.56909000)]]
                         , basis = 'cc-pvdz')

C2H6O_pvtz = gto.M(atom =[['C' ,     ( 0.01247000,       0.02254000,       1.08262000)]
                         ,['C',      ( -0.00894000,     -0.01624000,      -0.43421000)]
                         ,['H',      (-0.49334000,       0.93505000,       1.44716000)]
                         ,['H',      (1.05522000,        0.04512000,       1.44808000)]
                         ,['H',      (-0.64695000,      -1.12346000,       2.54219000)]
                         ,['H',      ( 0.50112000,      -0.91640000,      -0.80440000)]
                         ,['H',      ( 0.49999000,       0.86726000,      -0.84481000)]
                         ,['H',      (-1.04310000,      -0.02739000,      -0.80544000)]
                         ,['O',      (-0.66442000,      -1.15471000,       1.56909000)]]
                         , basis = 'cc-pvtz')

C3H8_pvdz = gto.M(atom= [['C',(0,0,0.5859)],['C',(0,1.2729,-0.2599)],['C',(0,0.-1.2729,-0.2599)],
                         ['H',(0.8700,0,1.2380)],['H',(-0.8700,0,1.2380)],['H',(0,2.1618,0.3635)],
                         ['H',(0,-2.1618,0.3635)],['H',(0.8770,1.3170,-0.8999)],['H',(-8.770,1.3170,-0.8999)],
                         ['H',(-0.8770,-1.3170,-0.8999)],['H',(0.8770,-1.3170,-0.8999)]], 
                          basis = 'cc-pvdz')

C3H8O_pvdz = gto.M(atom= [['C',(-1.4475,1.2270,0.0000)],['C',(0.0000,0.7379,0.0000)]
                          ,['C',(0.1037,-0.7769,0.0000)],['O',(1.4641,-1.1242,0.0000)]
                          ,['H',(-0.3967,-1.1837,-0.8780)],['H',(1.5524,-2.0666,0.0000)]
                          ,['H',(0.5249,1.1174,-0.8714)],['H',(-0.3967,-1.1837,0.8780)]
                          ,['H',(0.5249	,1.1174,0.8714)],['H',(-1.9842,0.8770,-0.8775)]
                          ,['H',(-1.9842,0.8770,0.8775)],['H',(-1.4903,2.3110,0.0000)]]
                          , basis = 'cc-pvdz')

C4H10_pvdz = gto.M(atom= [ ['C',(-0.4215,0.6382,0.0000)],['C',(0.4215,-0.6382,0.0000)]
                          ,['C',(0.4215	,1.9129,0.0000)], ['C',(-0.4215,-1.9129,0.0000)]
                          ,['H',(-1.0746,0.6369	,0.8704)],['H',(-1.0746,0.6369,-0.8704)]
                          ,['H',(1.0746	,-0.6369,0.8704)],['H',(1.0746,-0.6369,-0.8704)]
                          ,['H',(-0.2048,2.7997	,0.0000)],['H',(1.0611,1.9598,0.8770)]
                          ,['H',(1.0611,1.9598,-0.8770)],['H',(0.2048,-2.7997,0.0000)]
                          ,['H',(-1.0611,-1.9598,0.8770)],['H',(-1.0611,-1.9598,-0.8770)]]
                          , basis = 'cc-pvdz')

C4H10O_pvdz = gto.M(atom= [['C',(1.3594	,-0.3333,0.0000)],['C',(0.0000,0.3432,0.0000)]
                          ,['C',(-1.1610,-0.6522,0.0000)],['C',(-2.5268,0.0333,0.0000)]
                          ,['H',(1.4586	,-0.9704,0.8779)],['H',(1.4586,-0.9704,-0.8779)]
                          ,['H',(-0.0617,0.9891,0.8717)],['H',(-0.0617,0.9891,-0.8717)]
                          ,['H',(-1.0850,-1.3006,0.8706)],['H',(-1.0850,-1.3006,-0.8706)]
                          ,['H',(-3.3316,-0.6951,0.0000)],['H',(-2.6493,0.6625,0.8769)]
                          ,['H',(2.6493,0.6625,-0.8769)],['H',(3.2042,0.2648,0.0000)]
                          ,['O',(2.3466	,0.6654,0.0000)]], basis = 'cc-pvdz')

mol_list = [('H2O',H2O_pvdz), ('CH4',CH4_pvdz)
           ,('CH3OH',CH3OH_pvdz), ('C2H6',C2H6_pvdz),
           ('C3H8',C3H8_pvdz), ('HF',HF_pvtz), ('N2',N2_pvtz)]

#to run to evaluate the energy for all the molecules of mol_list
def auto_test(mol_list):
    for mol in mol_list:
        print('----------------------------------------------')
        print('Computation of '+mol[0])
        mol[1].verbose = 0
        n, no = compute_1RDM(mol[1], epsi=1e-8)
    print('Computation terminated')

#check that the ground state energy converges with stronger termination criterion epsi 
def conv_test(mol,disp=0):
    for i in range(9):
        epsi= 10**(-i)
        print('Convergence for epsi='+str(epsi))
        n,no = compute_1RDM(mol,epsi=epsi,disp=disp)
        print('-------------------------------------------')
    











