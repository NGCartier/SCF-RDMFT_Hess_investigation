#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:40:30 2023

@author: nicolascartier
"""


import re
import numpy  as np 
from matplotlib import pyplot as plt


#The path used to generate the output files of compute_1RDM
path = "/Users/nicolascartier/Documents/Programmes/Tests/Conv data/"
          
E_H2O = -76.1545657
E_h2o = -75.0711483
E_C2H6 = -73.41954
E_CH3OH = -115.776267      

#Bellow a list of plots I made, full_ refers to an Hessian in x space close_ to an Hessian in gamma space,
#the files were generated with compute_1RDM, their names correspond to the name used for the argument file.
exa_hess_h2o       = [path+'full_hess_exa_h2o.txt','exa','black',E_h2o]
close_hess_SR1_h2o = [path+'close_hess_SR1_h2o.txt','SR1 aux','red',E_h2o]
close_hess_BFGS_h2o= [path+'close_hess_BFGS_h2o.txt','BFGS aux','green',E_h2o]
full_hess_SR1_h2o  = [path+'full_hess_SR1_h2o.txt','SR1 full','lightcoral',E_h2o]
full_hess_BFGS_h2o = [path+'full_hess_BFGS_h2o.txt','BFGS full','limegreen',E_h2o]

hess_approx_h2o = [exa_hess_h2o,close_hess_BFGS_h2o,close_hess_SR1_h2o,
                   full_hess_BFGS_h2o,full_hess_SR1_h2o]

exa_hess_H2O       = [path+'full_hess_exa_H2Opvdz.txt','exa','black',E_H2O]
close_hess_SR1_H2O = [path+'close_hess_SR1_H2Opvdz.txt','SR1 aux','red',E_H2O] 
close_hess_BFGS_H2O= [path+'close_hess_BFGS_H2Opvdz.txt','BFGS aux','green',E_H2O]
full_hess_SR1_H2O  = [path+'full_hess_SR1_H2Opvdz.txt','SR1 full','lightcoral',E_H2O]
full_hess_BFGS_H2O = [path+'full_hess_BFGS_H2Opvdz.txt','BFGS full','limegreen',E_H2O]
close_hess_SR1_H2O_exastart = [path+'close_hess_SR1_H2Opvdz_exastart.txt','SR1 aux exainit','maroon',E_H2O]
close_hess_BFGS_H2O_exastart = [path+'close_hess_BFGS_H2Opvdz_exastart.txt','BFGS aux exainit','maroon',E_H2O]
full_hess_SR1_H2O_exastart  = [path+'full_hess_SR1_H2Opvdz_exastart.txt','SR1 full exainit','darkgreen',E_H2O]
full_hess_BFGS_H2O_exastart  = [path+'full_hess_BFGS_H2Opvdz_exastart.txt','BFGS full exainit','greenyellow',E_H2O]

hess_approx_H2O = [exa_hess_H2O,close_hess_BFGS_H2O,close_hess_SR1_H2O,
                   full_hess_BFGS_H2O,full_hess_SR1_H2O,
                   close_hess_SR1_H2O_exastart,close_hess_BFGS_H2O_exastart,
                   full_hess_SR1_H2O_exastart,full_hess_BFGS_H2O_exastart]

close_hess_SR1_H2O_exastart_gtol = [path+'close_hess_SR1_H2Opvdz_exastart_gtol.txt','SR1 aux exainit grad cond','gold',E_H2O]
close_hess_SR1_H2O_gtol = [path+'close_hess_SR1_H2Opvdz_gtol.txt','SR1 aux grad cond','olive',E_H2O] 
full_hess_SR1_H2O_exastart_gtol = [path+'full_hess_SR1_H2Opvdz_exastart_gtol.txt','SR1 exainit grad cond','gold',E_H2O]

sdecent_h2o = [path+'sdecent_h2o_gtol.txt','steepest descent','tan',E_h2o]
sdecent_h2o_behaviour = [path+'sdecent_h2o_gtol_behaviour.txt','steepest descent','tan',E_h2o]
sdecent_H2O = [path+'sdecent_H2Opvdz.txt','steepest descent','tan',E_H2O]

hess_SR1_aux_H2O_correction = [close_hess_SR1_H2O,close_hess_SR1_H2O_exastart,
                               close_hess_SR1_H2O_exastart_gtol,sdecent_H2O]
hess_SR1_H2O_correction = [full_hess_SR1_H2O,full_hess_SR1_H2O_exastart,
                           full_hess_SR1_H2O_exastart_gtol,sdecent_H2O]

close_hess_BFGS_h2o_abs= [path+'close_hess_BFGS_h2o_abs.txt','BFGS aux','green',E_H2O]

close_hess_SR1_h2o_exastart = [path+'close_hess_SR1_h2o_exastart.txt','SR1 aux','red',E_h2o]
close_hess_BFGS_h2o_exastart= [path+'close_hess_BFGS_h2o_exastart.txt','BFGS aux','green',E_h2o]
full_hess_SR1_h2o_exastart  = [path+'full_hess_SR1_h2o_exastart.txt','SR1 full','lightcoral',E_h2o]
full_hess_BFGS_h2o_exastart = [path+'full_hess_BFGS_h2o_exastart.txt','BFGS full','limegreen',E_h2o]

hess_approx_h2o_exastart = [exa_hess_h2o,close_hess_SR1_h2o_exastart,close_hess_BFGS_h2o_exastart,
                            full_hess_SR1_h2o_exastart,full_hess_BFGS_h2o_exastart]

close_hess_SR1_h2o_b = [path+'close_hess_SR1_h2o.txt','no exa block','black',E_h2o]
close_hess_SR1_h2o_nn      = [path+'close_hess_SR1_h2o_nn.txt','exa occ-occ','blue',E_h2o]
close_hess_SR1_h2o_nono    = [path+'close_hess_SR1_h2o_NONO.txt','exa NO-NO','gold',E_h2o]
close_hess_SR1_h2o_nno     = [path+'close_hess_SR1_h2o_nNO.txt','exa occ-NO','red',E_h2o]
close_hess_SR1_h2o_nn_nono = [path+'close_hess_SR1_h2o_nn_NONO.txt','exa occ-occ & NO-NO','green',E_h2o]
close_hess_SR1_h2o_nn_nno  = [path+'close_hess_SR1_h2o_nn_nNO.txt','exa occ-occ & occ-NO','purple',E_h2o]
close_hess_SR1_h2o_nono_nno= [path+'close_hess_SR1_h2o_NONO_nNO.txt','exa NO-NO & occ-NO','orange',E_h2o]

hess_approx_h2o_SR1_aux = [close_hess_SR1_h2o_b, close_hess_SR1_h2o_nn, close_hess_SR1_h2o_nono,
                           close_hess_SR1_h2o_nno, close_hess_SR1_h2o_nn_nono, close_hess_SR1_h2o_nn_nno,
                           close_hess_SR1_h2o_nono_nno]

close_hess_BFGS_h2o_b = [path+'close_hess_BFGS_h2o.txt','no exa block','black',E_h2o]
close_hess_BFGS_h2o_nn      = [path+'close_hess_BFGS_h2o_nn.txt','exa occ-occ','blue',E_h2o]
close_hess_BFGS_h2o_nono    = [path+'close_hess_BFGS_h2o_NONO.txt','exa NO-NO','gold',E_h2o]
close_hess_BFGS_h2o_nno     = [path+'close_hess_BFGS_h2o_nNO.txt','exa occ-NO','red',E_h2o]
close_hess_BFGS_h2o_nn_nono = [path+'close_hess_BFGS_h2o_nn_NONO.txt','exa occ-occ & NO-NO','green',E_h2o]
close_hess_BFGS_h2o_nn_nno  = [path+'close_hess_BFGS_h2o_nn_nNO.txt','exa occ-occ & occ-NO','purple',E_h2o]
close_hess_BFGS_h2o_nono_nno= [path+'close_hess_BFGS_h2o_NONO_nNO.txt','exa NO-NO & occ-NO','orange',E_h2o]

hess_approx_h2o_BFGS_aux = [close_hess_BFGS_h2o_b, close_hess_BFGS_h2o_nn, close_hess_BFGS_h2o_nono,
                           close_hess_BFGS_h2o_nno, close_hess_BFGS_h2o_nn_nono, close_hess_BFGS_h2o_nn_nno,
                           close_hess_BFGS_h2o_nono_nno]

full_hess_BFGS_H2O_b = [path+'full_hess_BFGS_H2Opvdz.txt','no exa block','black',E_H2O]
full_hess_BFGS_H2O_nn      = [path+'full_hess_BFGS_H2Opvdz_nn.txt','exa occ-occ','blue',E_H2O]
full_hess_BFGS_H2O_nono    = [path+'full_hess_BFGS_H2Opvdz_NONO.txt','exa NO-NO','gold',E_H2O]
full_hess_BFGS_H2O_nno     = [path+'full_hess_BFGS_H2Opvdz_nNO.txt','exa occ-NO','red',E_H2O]
full_hess_BFGS_H2O_nn_nono = [path+'full_hess_BFGS_H2Opvdz_nn_NONO.txt','exa occ-occ & NO-NO','green',E_H2O]
full_hess_BFGS_H2O_nn_nno  = [path+'full_hess_BFGS_H2Opvdz_nn_nNO.txt','exa occ-occ & occ-NO','purple',E_H2O]
full_hess_BFGS_H2O_nono_nno= [path+'full_hess_BFGS_H2Opvdz_NONO_nNO.txt','exa NO-NO & occ-NO','orange',E_H2O]

hess_approx_H2O_BFGS = [full_hess_BFGS_H2O_b, full_hess_BFGS_H2O_nn, full_hess_BFGS_H2O_nono,
                           full_hess_BFGS_H2O_nno, full_hess_BFGS_H2O_nn_nono, full_hess_BFGS_H2O_nn_nno,
                           full_hess_BFGS_H2O_nono_nno]


close_hess_BFGS_H2O_b = [path+'close_hess_BFGS_H2Opvdz.txt','no exa block','black',E_H2O]
close_hess_BFGS_H2O_nn      = [path+'close_hess_BFGS_H2Opvdz_nn.txt','exa occ-occ','blue',E_H2O]
close_hess_BFGS_H2O_nono    = [path+'close_hess_BFGS_H2Opvdz_NONO.txt','exa NO-NO','gold',E_H2O]
close_hess_BFGS_H2O_nno     = [path+'close_hess_BFGS_H2Opvdz_nNO.txt','exa occ-NO','red',E_H2O]
close_hess_BFGS_H2O_nn_nono = [path+'close_hess_BFGS_H2Opvdz_nn_NONO.txt','exa occ-occ & NO-NO','green',E_H2O]
close_hess_BFGS_H2O_nn_nno  = [path+'close_hess_BFGS_H2Opvdz_nn_nNO.txt','exa occ-occ & occ-NO','purple',E_H2O]
close_hess_BFGS_H2O_nono_nno= [path+'close_hess_BFGS_H2Opvdz_NONO_nNO.txt','exa NO-NO & occ-NO','orange',E_H2O]

hess_approx_H2O_BFGS_aux = [close_hess_BFGS_H2O_b, close_hess_BFGS_H2O_nn, close_hess_BFGS_H2O_nono,
                           close_hess_BFGS_H2O_nno, close_hess_BFGS_H2O_nn_nono, close_hess_BFGS_H2O_nn_nno,
                           close_hess_BFGS_H2O_nono_nno]

full_hess_BFGS_hybrid_H2O = [path+'full_hess_BFGS_hybrid_H2Opvdz.txt','BFGS hybrid','limegreen',E_H2O]
full_hess_SR1_hybrid_H2O = [path+'full_hess_SR1_hybrid_H2Opvdz.txt','SR1 hybrid','lightcoral',E_H2O]
close_hess_BFGS_hybrid_H2O = [path+'close_hess_BFGS_hybrid_H2Opvdz.txt','BFGS aux hybrid','green',E_H2O]
close_hess_SR1_hybrid_H2O = [path+'close_hess_SR1_hybrid_H2Opvdz.txt','SR1 aux hybrid','red',E_H2O]

hess_hybrid_H2O = [exa_hess_H2O,full_hess_BFGS_hybrid_H2O,full_hess_SR1_hybrid_H2O,
                   close_hess_BFGS_hybrid_H2O,close_hess_SR1_hybrid_H2O]

close_hess_BFGS_hybrid_3_H2O = [path+'close_hess_BFGS_H2O_hybrid_3.txt','BFGS aux hybrid 3','dimgray',E_H2O]
close_hess_BFGS_hybrid_5_H2O = [path+'close_hess_BFGS_H2O_hybrid_5.txt','BFGS aux hybrid 5','darkgreen',E_H2O]
close_hess_BFGS_hybrid_10_H2O = [path+'close_hess_BFGS_hybrid_H2Opvdz.txt','BFGS aux hybrid 10','seagreen',E_H2O]
close_hess_BFGS_hybrid_20_H2O = [path+'close_hess_BFGS_H2O_hybrid_20.txt','BFGS aux hybrid 20','limegreen',E_H2O]
close_hess_BFGS_hybrid_50_H2O = [path+'close_hess_BFGS_H2O_hybrid_50.txt','BFGS aux hybrid 50','lime',E_H2O]
close_hess_BFGS_H2O_g = [path+'close_hess_BFGS_H2Opvdz.txt','BFGS aux without hybrid','greenyellow',E_H2O]

hess_hybrid_BFGS_H2O =[exa_hess_H2O,close_hess_BFGS_H2O_g,close_hess_BFGS_hybrid_3_H2O,
                       close_hess_BFGS_hybrid_5_H2O, close_hess_BFGS_hybrid_10_H2O,
                       close_hess_BFGS_hybrid_20_H2O,close_hess_BFGS_hybrid_50_H2O]

def plot_conv_hess_diff(file, figsize = (11,6), itermax=10000, title=""):
    '''Plots the error in the energy and Hessian difference with the iterations
    Args: file: one of the above lists for 1 call to compute_1RDM, will read it 
                to plot the corresponding values
          itermax: maximum number of iterations to plot
          title: title of the plot'''
    fig,ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    plt.title(title)
    
    E = []; dH = [] ;dHnn = []; dHnono = []; dHnno =[]; iter_ = 0
    data = open(file[0],'r')
    for line in data:
        if iter_>itermax: 
            break;
        if line=="---------\n":
            plt.axvline(iter_, color='black', linestyle=':')
        else:
            
            data_i = re.findall(r'-?(?:\d+\.)?\d+e[+-]?\d+|-?(?:\d+\.)?\d+', line)
            E.append(float(data_i[0])); iter_+=1          
            dH.append(float(data_i[1]))
            dHnn.append(float(data_i[2]))
            dHnono.append(float(data_i[3]))
            dHnno.append(float(data_i[4]))
    E = np.array(E) ; dH = np.array(dH); dHnn = np.array(dHnn)
    dHnono = np.array(dHnono); dHnno = np.array(dHnno)
    data.close()
    x = np.arange(1, len(E)+1)   
    ax2.plot(x, abs(E - file[3])/abs(file[3]), label = file[1], color=file[2],zorder=2)
    ax.plot(x, dH,label='full H',color='black',linestyle='--')
    ax.plot(x, dHnn,label='H occ ',color='blue',linestyle='--')
    ax.plot(x, dHnono,label='H NO ',color='cyan',linestyle='--')
    ax.plot(x, dHnno,label='H coupled ',color='purple',linestyle='--')
     
    ax.set_yscale('log')
    ax2.set_yscale('log')
    if(dH.max()>1e6):
        ax.set_ylim(top=1e6)
    if(dH.min()<1e-3):
        ax.set_ylim(bottom=1e-3)
    
    
    ax.set_ylabel('absolute Hessian difference (dotted line)')
    ax2.set_ylabel('realitve E error ('+file[2]+'solid line)')
    ax.set_xlabel('iteration')
    
    ax.legend(loc=2)
    ax2.legend(loc=1)
    plt.show()    
    
def plot_conv(files, figsize = (11,6), maxiter=10000, title="",precision=1e-6):
    '''Plots the error in the energy for a list of calls to compute_1RDM with the iterations
    Args: file: one of the above lists of calls to compute_1RDM, will read them 
                to plot the corresponding values
          maxiter: maximum number of iterations to plot
          title: title of the plot'''
    fig = plt.figure(figsize=figsize); plt.title(title); lmax = 0
    for file in files:
        data = open(file[0],'r'); E= []; iter_ =0
        for line in data:
            if iter_>maxiter: 
                break;
            if line!="---------\n":
                data_i = re.findall(r'-?(?:\d+\.)?\d+e[+-]?\d+|-?(?:\d+\.)?\d+', line)
                E.append(float(data_i[0])); iter_+=1 
        data.close(); E = np.array(E) ; l = len(E)
        if (l>lmax): lmax = l
        x = np.arange(1, l+1)
        plt.plot(x,abs(E-file[3])/abs(file[3]),label = file[1],color=file[2])
    plt.plot(np.arange(0,lmax,1),[precision]*lmax,color='grey', linestyle=':')
    plt.yscale('log')
    plt.ylabel('relative E error')
    plt.xlabel('iteration')
    plt.legend()
    plt.plot()
    
exa_h2o_behaviour = [path+'exa_hess_beavior_h2o.txt','exa H2O','blue',E_h2o]
exa_H2O_behaviour = [path+'exa_hess_beavior_H2Opvdz.txt','exa H2O','darkblue',E_H2O]
SR1_H2O_behaviour_exastart = [path+'full_hess_SR1_H2Opvdz_exastart_behaviour.txt','SR1 H2O','red',E_H2O]
SR1_H2O_behaviour_exastart_gtol = [path+'full_hess_SR1_H2Opvdz_exastart_gtol_behaviour.txt','SR1 H2O','darkred',E_H2O]

def plot_hess_behaviour(file,figsize=(10,12),itermax=10000,title=""):
    '''Plots the error in the energy, evolution of the exact Hessian, number of <0
    eigenvalues of the Hessian, step and gradient size with the iterations
    Args: file: one of the above lists with behavior key word, will read it 
                to plot the corresponding values
          itermax: maximum number of iterations to plot
          title: title of the plot'''
    fig,ax = plt.subplots(5,figsize=figsize)
    fig.suptitle(title)
    
    E = []; neigvl = []; dH = []; s = []; grad = []; iter_ = 0
    data = open(file[0],'r')
    for line in data:
        if iter_>itermax: 
            break;
        if line=="---------\n":
            plt.axvline(iter_, color='black', linestyle=':')
        else:
            data_i = re.findall(r'-?(?:\d+\.)?\d+e[+-]?\d+|-?(?:\d+\.)?\d+', line)
            E.append(float(data_i[0])); iter_+=1   
            neigvl.append(float(data_i[1]))
            dH.append(float(data_i[3]))
            s.append(float(data_i[4]))
            grad.append(float(data_i[5]))
    E = np.array(E); neigvl = np.array(neigvl); dH = np.array(dH) 
    x = np.arange(1,len(E)+1)
    ax[0].plot(x, abs(E-file[3])/abs(file[3]), color=file[2])
    ax[0].set(ylabel="relative error in E",yscale="log")
    ax[1].plot(x, dH, color=file[2])
    ax[1].set(ylabel="$||H_{exa_k}-H_{exa_{k-1}}||$",yscale="log")
    if(dH.min()<1e-5):
        ax[1].set_ylim(bottom=1e-5)
    ax[2].plot(x, neigvl,color=file[2])
    ax[2].set(ylabel="# of eigvl<0")
    ax[3].plot(x, s, color=file[2])
    ax[3].set(ylabel="||s||",yscale="log")
    ax[4].plot(x, grad, color=file[2])
    ax[4].set(ylabel="||grad||",yscale="log")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    