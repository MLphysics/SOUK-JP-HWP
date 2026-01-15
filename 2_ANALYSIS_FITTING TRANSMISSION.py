# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 06:15:33 2026

@author: matth
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def importfile(file):
    f = np.load(file)
    freq = f["freq"]
    data = f["data"]
    
    df = pd.DataFrame({"freq": freq, 
                       "data": data})
    return df
    
def resample(df, nf):
    newdata = np.interp(nf, df["freq"], df["data"])
    newdf = pd.DataFrame({"freq": nf, 
                           "data": newdata})
    return newdf

def cleandata(df):
    freq = df["freq"]
    data = df["data"]
    data.rolling(window=3, min_periods=1).mean()
    
    df = pd.DataFrame({"freq": freq, 
                       "data": data})
    return df
    
def simpleplot(dfsam, dfref):
    plt.figure()
    plt.plot(dfref["freq"], dfsam["data"]/ dfref["data"])
    plt.show()

def main(pathref, pathsam):
    
    dfref = importfile(pathref)
    dfsam = importfile(pathsam)
    
    f = np.linspace(140, 220, 1000)
    
    dfref = resample(dfref,f)
    dfsam = resample(dfsam,f ) 
    
    dfref = cleandata(dfref)
    dfsam = cleandata(dfsam ) 
    
    simpleplot(dfsam, dfref)
    


main( "C:/Users/matth/OneDrive - Cardiff University/PhD/JP_Sapphire-testing/Data/Alumina/Air-after-ppol-inc7-20260115-1_1000Hz_G_band_N_1001_20260115150828.npz"
     ,"C:/Users/matth/OneDrive - Cardiff University/PhD/JP_Sapphire-testing/Data/Alumina/D505-Alumina-2-3-ppol-inc7-20260115-1_1000Hz_G_band_N_1001_20260115150701.npz")