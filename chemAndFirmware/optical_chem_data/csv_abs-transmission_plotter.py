# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import curve_fit
matplotlib.rcParams.update({'font.size': 16})

#This value is pulled from literature and used as a conversion factor
absorbtivity = 20000 * 0.24 

#read lockin_data
def read_lockin_data(file, delimeter=";", channels=1 ):
    if channels==1:
        skiprows=5
    elif channels==2:
        skiprows=7
    df = pd.read_csv(file, delimiter=delimeter, skiprows=skiprows).to_numpy()
    if channels > 1:
        channel2_index = np.where(abs(df[1:,0]-df[:-1,0]) > 0.5*np.max(abs(df[:,0])))[0]
        channel2 = df[channel2_index[0]+1:,1]
        if len(channel2) != len(df[:channel2_index[0]+1,0]):
            channel2 = channel2[:min( len(channel2), len(df[:channel2_index[0]+1,0]) )]
            df = df[:min( len(channel2), len(df[:channel2_index[0]+1,0]) ), :]
        df = np.concatenate((df[:channel2_index[0]+1,:], channel2.reshape(len(channel2),1)), axis=1)
    return df

#takes voltage values read from lockin and converts to a transmission value
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
#adjusts the timeset of the lockin
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

cutoff = 10

# folder = "C:/Users/Mitch So/OneDrive/Documents/Python files/"
folder = "C:/Users/mso/Leiden Tech Dropbox/Projects - Shared/2008 - Phosphator Phase II/8-Data/Lock-in Optics Testing Data/Kinetics Data/20240424 Ladder/"

#input files here
files = ["20240424-Ladder-0uM.txt",
         "20240424-Ladder-1uM.txt",
         "20240424-Ladder-5uM.txt",
         "20240424-Ladder-20uM-after-3min.txt",
         "20240424-Ladder-150uM.txt",
         "20240424-Ladder-250uM.txt"
         ]
#input baseline file names here
baselines = ["20240424-Ladder-Baseline-1.txt",
             "20240424-Ladder-Baseline-2.txt",
             "20240424-Ladder-Baseline-3.txt",
             "20240424-Ladder-Baseline-4.txt",
             "20240424-Ladder-Baseline-5.txt",
             "20240424-Ladder-Baseline-6.txt"
         ]
#label names array here
labels = ["0uM",
          "1uM",
          "5uM",
          "20uM",
          "150uM",
          "250uM"
          ]

pin0=True

#ladder concentrations
exp_concs = [0, 1, 5, 20, 150, 250]

#initializing one figure two subplots
fig, axs = plt.subplots(1, 2, figsize=(15,6))
for file in range(len(exp_concs)):
    baseline = read_lockin_data(folder + baselines[2*i], delimiter=";", channels=2)
    experiment = read_lockin_data(folder + files[2*i], delimiter=";", channels=2)
    
    # Convert voltage to transmission and adjust time
    experiment = vol2transmission(experiment, zero_val=np.array([np.average(baseline[:, 1]), np.average(experiment[:, 2])]))
    experiment = adjust_time(experiment[:, :2], cutoff=cutoff)
    transmission_values = experiment[:,1]   
    time_values = experiment[:,0]
    
    #absorbance from transmission A = -log(T) 
    absorbance_values = -np.log10(transmission_values)
    #absorbance values are more or less irrelevant against time, looking for asymptotic values for linearity
    
    # Scatter plot of original data
    # axs[i].scatter(experiment[:, 0], experiment[:, 1], label='Original data')
    
    # Curve fitting
    # popt, pcov = curve_fit(exp_func, experiment[:, 0], experiment[:, 1])
    # x_fit = np.linspace(min(time_values), max(time_values), 100)
    # y_fit = exp_func(x_fit, *popt)
    # axs[i].plot(x_fit, y_fit, color='red', label='Exponential Curve Fit')
    
#  Set labels and title
axs[1].set_xlabel('Time (min)')
axs[1].set_ylabel('Transmission')
axs[1].grid(True)
axs[1].legend()
axs[2].set_title(f'absorbance over time')
axs[1].set_title(f'transmission over time')

plt.tight_layout()

#does not automatically plot graph needs this for powershell and embedded vs terminal
plt.show()

#This plots the absorbance vs time graphs use for ladder experiments to check linearity of resultants   
