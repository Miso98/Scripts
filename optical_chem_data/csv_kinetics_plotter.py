# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import curve_fit
matplotlib.rcParams.update({'font.size': 16})

absorbtivity = 20000 * 0.24 

#reading CSV for raw lockin voltage data 
def read_lockin_data(file, delimiter=";", channels=1):
    if channels == 1:
        skiprows = 5
    elif channels == 2:
        skiprows = 7
    df = pd.read_csv(file, delimiter=delimiter, skiprows=skiprows).to_numpy()
    if channels > 1:
        channel2_index = np.where(abs(df[1:,0]-df[:-1,0]) > 0.5*np.max(abs(df[:,0])))[0]
        channel2 = df[channel2_index[0]+1:,1]
        if len(channel2) != len(df[:channel2_index[0]+1,0]):
            channel2 = channel2[:min(len(channel2), len(df[:channel2_index[0]+1,0]))]
            df = df[:min(len(channel2), len(df[:channel2_index[0]+1,0])), :]
        df = np.concatenate((df[:channel2_index[0]+1,:], channel2.reshape(len(channel2),1)), axis=1)
    return df


#converting lockin voltage data into a transmission value
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


#adjusting the weird lock in time values that are read in milliseconds
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

#not currently fitting the absorbtivity factor
  
def exp_func(x, a, b):
        return a*  np.exp(b * x)

cutoff = 0.75
#cutoff is the value to start plotting, so for 0uM and 1uM the df will be a null set 
#unless specified to a higher cutoff of transmission here it is 10
#cutoff = 10
#folder = "C:/Users/Mitch So/OneDrive/Documents/Python files/"
folder = "C:/Users/mso/Leiden Tech Dropbox/Projects - Shared/2008 - Phosphator Phase II/8-Data/Lock-in Optics Testing Data/Kinetics Data/20240501 Asc 14 day stability/"
files = ["20240501-14Day-5C-1.txt",
         "20240501-14Day-5C-2.txt",
         "20240501-14Day-20C-2.txt",
         "20240501-14Day-20C-1.txt",
         "20240501-14Day-35C-1.txt",
         "20240501-14Day-35C-2.txt"
         ]
baselines = ["20240501-Baseline-1.txt",
             "20240501-Baseline-2.txt",
             "20240501-Baseline-3.txt",
             "20240501-Baseline-4.txt",
             "20240501-Baseline-5.txt",
             "20240501-Baseline-6.txt"
         ]

labels = ["5C-1",
          "5C-2",
          "20C-1",
          "20C-2",
          "35C-1",
          "35C-2"
          ]
pin0=True
#plt.figure(figsize=(12,9))
exp_temps = [5, 20, 35]
fig, axs = plt.subplots(1, len(exp_temps), figsize=(15,6))

#iterating through each file and fitting a curve to add to subplot   
for i in range(len(exp_temps)):

    baseline = read_lockin_data(folder + baselines[2*i], delimiter=";", channels=2)
    experiment = read_lockin_data(folder + files[2*i], delimiter=";", channels=2)
    
    # Convert voltage to transmission and adjust time
    experiment = vol2transmission(experiment, zero_val=np.array([np.average(baseline[:, 1]), np.average(experiment[:, 2])]))
    experiment = adjust_time(experiment[:, :2], cutoff=cutoff)
    transmission_values = experiment[:,1]   
    time_values = experiment[:,0]
    # Scatter plot of original data
    axs[i].scatter(experiment[:, 0], experiment[:, 1], label='Original data')
    
    # Curve fitting
    popt, pcov = curve_fit(exp_func, experiment[:, 0], experiment[:, 1])
    x_fit = np.linspace(min(time_values), max(time_values), 100)
    y_fit = exp_func(x_fit, *popt)
    axs[i].plot(x_fit, y_fit, color='red', label='Exponential Curve Fit')
    
    #  Set labels and title
    axs[i].set_xlabel('Time (min)')
    axs[i].set_ylabel('Transmission')
    axs[i].grid(True)
    axs[i].legend()
    axs[i].set_title(f'{exp_temps[i]}°C')
    
    # Display the equation of the exponential curve
    equation_text = f'y = {popt[0]:.2f} * e^({popt[1]:.2f} * x)'
    axs[i].text(0.1, 0.1, equation_text, transform=axs[i].transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()



