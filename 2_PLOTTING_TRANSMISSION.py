# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 07:08:50 2026

@author: matth
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.signal import welch


class Band:
    def __init__(self, name, lf, uf):
        self.name = name
        self.lf = lf
        self.uf = uf

Gband = Band("Gband", 140, 220)
Fband = Band("Fband", 90, 140)
Eband = Band("Eband", 60, 90)

class schema:
    freq = 'freq_data' 
    trans = 'trans_data' 
    phase = 'phase_data' 
    realdata = 'real_data' 
    imagdata = 'imag_data' 
    freqfit = 'freq_fit' 
    transfit = 'trans_fit' 
    realfit = 'real_fit' 
    imagfit ='imag_fit' 
    n = 'fit_n' 
    loss = 'fit_los' 
    nerr = 'fit_n_err'
    losserr = 'fit_los_err'
    transdiff = 'trans_diff'

def importfile(file):
    f = np.load(file)
    return f

def maxtransmissions 
    

def simpleplot(dfsam, dfref):
    plt.figure()
    plt.plot(dfref["freq"], dfsam["data"]/ dfref["data"])
    plt.show()





DIR = ( Path.cwd() /  "Data")

BAND = ["GBand", "FBand"]


SAMPLE = "20260116_MF2_rotation_test"
#"detailed" / "IMAG"
angle = [0]



n_arr = []
n_arr_err = []

loss_arr = []
loss_arr_err = []

avT = []
avTerr = []

PeakT = []
TroughT = []

for b in BAND:
    for a in angle: 
        ANG = str(a)+"deg/"
        directory_in_str = DIR+b+SAMPLE+ANG
        pathlist = Path(directory_in_str).glob('**/*_fit.npz')
        for path in pathlist:
            path_in_str = str(path)   
            file = np.load(path_in_str)
            
            ## Loads all of the _fit.npz iteratively  
            n_arr = np.append(n_arr, file[schema.n])
            n_arr_err = np.append(n_arr_err, file[schema.nerr])
            
            loss_arr = np.append(loss_arr, file[schema.loss])
            loss_arr_err = np.append(loss_arr_err, file[schema.losserr])
    
            #avT = np.append(avT, np.average(file[schema.transfit]))
            avT = np.append(avT, np.trapz(file[schema.transfit])/len(file[schema.transfit]))
            avTerr = np.append(avTerr, np.std(file[schema.transfit]))
            
            PeakT = np.append(PeakT, np.max(file[schema.transfit]))
            TroughT = np.append(TroughT, np.max(file[schema.transfit]))
        
    print(angle)
    print(n_arr)
    print(n_arr_err)
    
    print(loss_arr)
    print(loss_arr_err)
    
    print(avT)
    print(avTerr)
    
    
    

    
    
    