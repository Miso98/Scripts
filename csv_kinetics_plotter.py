
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.optimize as opt
matplotlib.rcParams.update({'font.size': 16})

absorbtivity = 20000 * 0.24 

def read_lockin_data(file, delimeter=";", channels=1 ):
    if channels==1:
        skiprows=5
    elif channels==2:
        skiprows=7
    df = pd.read_csv(file, delimiter=delimeter, skiprows=skiprows).to_numpy()
    if channels > 1:
        channel2_index = np.where(abs(df[1:,0]-df[:-1,0]) > 0.5*np.max(abs(df[:,0])))[0]
        # if len(channel2_index) > 1:
        #     channel2 = df[channel2_index[0]+1:,1]
        #     df = np.concatenate((df[:channel2_index[0]+1,:], channel2.reshape(len(channel2),1)), axis=1)
        # if len(df[:,0]) % 2 == 1:
        #     df = df[:-1,:]
        channel2 = df[channel2_index[0]+1:,1]
        if len(channel2) != len(df[:channel2_index[0]+1,0]):
            channel2 = channel2[:min( len(channel2), len(df[:channel2_index[0]+1,0]) )]
            df = df[:min( len(channel2), len(df[:channel2_index[0]+1,0]) ), :]
        df = np.concatenate((df[:channel2_index[0]+1,:], channel2.reshape(len(channel2),1)), axis=1)
    return df

def vol2transmission(df, zero_val=None):
    df2 = df.copy()
    if zero_val is None:
        # df2[:,1] = df2[:,1] / np.average(df2[:,1])
        zero_val = np.average(df2, axis=0)[1:]
    if np.size(df2,axis=1) > 2:
        df2[:,1:] = np.einsum('ij,j->ij', df2[:,1:], zero_val**-1)
    else:
        df2[:,1] = df2[:,1] * zero_val**-1
    return df2

def adjust_time(df,cutoff=None, ms=False, minutes=True):
    if cutoff is not None:
        df = df[df[:,1]<cutoff,:]
    df[:,0] = df[:,0] - df[0,0]
    if ms:
        df[:,0] = df[:,0] / 1e3
    if minutes == True:
        df[:,0] = df[:,0] / 60
    return df
    
def plot(df, label=""):
    plt.plot(df[:,0], df[:,1:], label=label)

#possible reaction rates (returns transmission)
def logistic_rxn(t, Cmax, k, t0):
    C = Cmax * (1 + np.exp(-1*k*(t - t0)))**-1
    T = 10**(-1*absorbtivity*C)
    return T
def exponential_rxn(t, Cmax, k, t0):
    C = Cmax * (1 - np.exp(-1*k*(t-t0)))
    T = 10**(-1*absorbtivity*C)
    return T
def secondorder_rxn(t, Cmax, k, alpha, t0):
    C = Cmax*(1 - (-1*(t-t0)*k*(1 - alpha) + 1)**(1/(1-alpha)))
    T = 10**(-1*absorbtivity*C)
    return T
def double_exp_rxn(t, Cmax, k1, k2, t0):
    C = Cmax * (1 - 1*(k2*np.exp(-1*k1*(t-t0)) -k1*np.exp(-1*k2*(t-t0))) / (k2-k1))
    T = 10**(-1*absorbtivity*C)
    return T

def triple_exp_rxn(t, Cmax, k1, k2, k3, t0):
    #not currently fitting the absorbtivity factor
    agg_factor = 10.0
    
    C3 = Cmax*k1*k2*( np.exp(-1*k1*(t-t0)) / ( (k1-k2)*(k1-k3) )
                     + np.exp(-1*k2*(t-t0)) / ( (k2-k1)*(k2-k3) )
                     + np.exp(-1*k3*(t-t0)) / ( (k3-k1)*(k3-k2) )
                     )
    C4 = Cmax + Cmax / ( (k1-k2)*(k2-k3)*(k1-k3) ) * (
        k2*k3*(k3-k2)*np.exp(-1*k1*(t-t0))
        + k1*k3*(k1-k3)*np.exp(-1*k2*(t-t0))
        + k1*k2*(k2-k1)*np.exp(-1*k3*(t-t0)))
    
    T = 10**( -1*absorbtivity*(C3 + agg_factor*C4) )
    return T

#folder = "C:/Users/jmyles/Leiden Tech Dropbox/Projects - Shared/2008 - Phosphator Phase II/8-Data/Lock-in Optics Testing Data/Kinetics Data/"
#1 day kinetics
#folder = "C:/Users/mso/Leiden Tech Dropbox/Projects - Shared/2008 - Phosphator Phase II/8-Data/Lock-in Optics Testing Data/Kinetics Data/Kinetics data 4-18-24 asc stability 100uM/1 Day/"
#7 day kinetics
#folder = "C:/Users/mso/Leiden Tech Dropbox/Projects - Shared/2008 - Phosphator Phase II/8-Data/Lock-in Optics Testing Data/Kinetics Data/Kinetics data 4-18-24 asc stability 100uM/7 Day/"
#repeatability experiment
#folder = "C:/Users/mso/Leiden Tech Dropbox/Projects - Shared/2008 - Phosphator Phase II/8-Data/Lock-in Optics Testing Data/Kinetics Data/Kinetics data 4-23-24 repeatability 94uM/"
folder = "C:/Users/mso/Leiden Tech Dropbox/Projects - Shared/2008 - Phosphator Phase II/8-Data/Lock-in Optics Testing Data/Kinetics Data/20240424 Ladder/"
#cutoff = 0.75
#cutoff is the value to start plotting at, so for cases of 0uM and 1uM the df will be a null set unless specified to a higher cutoff of transmission here it is 10
cutoff = 10


files = ["20240424-Ladder-0uM.txt",
         "20240424-Ladder-1uM.txt",
         "20240424-Ladder-5uM.txt",
         "20240424-Ladder-20uM-after-3min.txt",
         "20240424-Ladder-150uM.txt",
         "20240424-Ladder-250uM.txt"
         ]
baselines = ["20240424-Ladder-Baseline-1.txt",
             "20240424-Ladder-Baseline-2.txt",
             "20240424-Ladder-Baseline-3.txt",
             "20240424-Ladder-Baseline-4.txt",
             "20240424-Ladder-Baseline-5.txt",
             "20240424-Ladder-Baseline-6.txt"
         ]

labels = ["0uM",
          "1uM",
          "5uM",
          "20uM",
          "150uM",
          "250uM"
          ]

pin0=True
plt.figure(figsize=(12,9))

for file, baseline_path, label in zip(files, baselines, labels):
    baseline = read_lockin_data(
        folder + baseline_path,
        delimeter=";", channels=2)
    
    experiment = read_lockin_data(
        folder + file,
        delimeter=";", channels=2)
    experiment = vol2transmission(experiment, zero_val=np.array([np.average(baseline[:,1]), np.average(experiment[:,2])]))
    experiment = adjust_time(experiment[:,:2], cutoff=cutoff)
    plt.plot(experiment[:,0], experiment[:,1], label=label)
    
    
    half = int(len(experiment)/1)
    # half = np.argmax(experiment[:,0] > 5) #3 minutes
    if pin0:
        temp = experiment[0,1] * np.ones((int(half/4),2))
        temp[:,0] = 0
        experiment =np.concatenate((temp, experiment[:,0:2]))
        
        
    # f = triple_exp_rxn
    # popt, pcov = opt.curve_fit(f, 
    #                             experiment[:half,0], experiment[:half,1],
    #                             bounds=([0, 2e-3,1e-3,0.5e-6,-1e3],[1e-3, 1e1, 1e1, 3e-4, 1e3]),
    #                             x_scale=[1000, 1, 1, 10, 1]
    #                             )
    # print(popt[3])
    # f = double_exp_rxn
    # popt, pcov = opt.curve_fit(f, 
    #                             experiment[:half,0], experiment[:half,1],
    #                             bounds=([0, 2e-3,1e-3,-1e1],[1e-3, 1e1, 1e1, 1e1]),
    #                             x_scale=[1000, 1, 1,  1]
    #                             )
    
    
    #removing the curve fitting for exponential reaction 
    
    #f = exponential_rxn
    #popt, pcov = opt.curve_fit(f, 
                                #experiment[:half,0], experiment[:half,1],
                                #bounds=([0, 2e-3,-1e3],[1e-3, 1e1,  1e3]),
                                #x_scale=[1000, 1,  1]
                                #)
    # print(popt)
    # plt.axhline(10**(-1*popt[0]*absorbtivity), color="grey")
   # print(str(np.sum((f(experiment[:,0], *popt) - experiment[:,1])**2)/half ) 
    #      + ", " + str(popt[0]) )
    
    # plt.semilogy(experiment[:,0], 1*np.log10(f(experiment[:,0], *popt))/absorbtivity + popt[0], "--",label=label)
    # plt.semilogy(experiment[:,0], 1*np.log10(experiment[:,1:])/absorbtivity  + popt[0],label=label)
    #plt.plot(experiment[:,0], experiment[:,1], label=label)
   # plt.plot(experiment[:,0], (f(experiment[:,0], *popt)), "--", label=label)
    

plt.xlabel("time (min)")
plt.ylabel("Transmission")
plt.ylim([-0.01,1.01])
plt.title("Reaction Kinetics Ladder")
plt.legend()
plt.tight_layout()
#does not automatically plot graph needs this for powershell and embedded vs terminal
plt.show()