#######Analysis borrowed from Toptica project 



# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:51:22 2024

@author: c21062215
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as plt1
import pandas as pd

from scipy.constants import speed_of_light as c

from ARCS.tmm import ARrecipe

from ARCS.schema import RefInd as RI
from ARCS.schema import DataSchema as DS

from TOPSAM.schema import DataSchema as ds
from   TOPSAM.opus import ImportTransmission, ManualTransmissionTOP


class schema:
    data = "data"
    freq_ = "freq"
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
    freq = f[schema.freq]
    data = f[schema.trans]
    
    df = pd.DataFrame({schema.freq: freq, 
                       schema.trans: data})
    return df


def AR1layer():
    ### converts theta parameters into recipe format for AR model 

    #                       L1            LG           Ll             LG          L1
    thickness = (np.inf, theta.L1    , theta.LG   , theta.Ll     , theta.LG   , theta.L1    , np.inf)
    refindex  = (RI.FS , theta.npPTFE, theta.nLDPE, theta.nUHMWPE, theta.nLDPE, theta.npPTFE, RI.FS )
    return thickness, refindex

def AR0layer():
    ### converts theta parameters into recipe format for AR model 
    #                         Ll
    thickness = (np.inf, theta.Ll     , np.inf)
    refindex  = (RI.FS , theta.nUHMWPE, RI.FS )
    return thickness, refindex

def main(AR, PATH, num,ang, FILETYPE):
    
    
   
    #importfile(file)

    df  =  importfile(PATH) 
    
    #generate model using TMM code
    t_listsaved, n_list = AR
    #
    print(n_list)
    if FILETYPE == "OPUS" or FILETYPE == "TERASCAN" or FILETYPE == "VNA":
        ks_= np.linspace((df[schema.freq].iloc[0]*10**9) /c,
                         (df[schema.freq].iloc[-1]*10**9)/c,
                         num=num)
    else:
        ff = 1000
        ks_ = np.linspace(0,ff*10**9/c,2000)
    #
    slabs = ARrecipe(t_listsaved, n_list, ks_, "-",angle=ang, plot=False)
    
    
      
    
    ### plot(!)
    fig, ax1 = plt.subplots()
    ax1.plot(c*slabs[DS.ks]/10**9, slabs[DS.T]   , 'blue',label="TMM"    , alpha=0.5)
    
    if FILETYPE == "OPUS" or FILETYPE == "TERASCAN" or FILETYPE == "VNA":
        ax1.plot(df[schema.freq]   , df[schema.trans] , 'red' ,label=FILETYPE, alpha=0.5)
    else:
        pass
    
    ax1.legend()
    
    plt.grid()
    ax1.set_ylabel(DS.T)
    ax1.set_xlabel(DS.F)

    ax1.set_ylim(0, 1)
    #ax1.set_xscale("log")
    plt.title("TMM Vs Data: Dielectric loss", fontweight= "bold")
    plt.show()
    
    
    
    
    
    #############################################


if __name__ == "__main__":
    ### FILE SELECTION
    DIR = "C:/Users/matth/OneDrive - Cardiff University/PhD/JP_Sapphire-testing/Data/GBand/Alumina/0deg/"
    
    sam = "D505-Alumina-2-3-ppol-inc7-20260115-1_1000Hz_G_band_N_1001_20260115150701_fit.npz"
    PATH = DIR+sam
    
    class theta:
        L1 = 0#0.450e-3 *0.94 # 4.47909095e-04
        LG = 0#6e-6 #5.97474649e-06
        Ll = 3.082e-3 #10240e-06
        npPTFE   = 0#complex(1.2559e+00, 0)
        nLDPE    = 0#complex(1.5141e+00, 0)
        nUHMWPE = complex(3.12,0.0005) #1.23e+00, 0)#complex(1.526e+00, 5e-4) #complex(1.52605269e+00, 8.72272763e-05)#
    #
    n = 2000
    main(AR0layer(), PATH=PATH, num = n, ang=7, FILETYPE="VNA")









"""

### IMPORTS

import pathlib
#import opusFC

import matplotlib.pyplot as plt 
import numpy             as np
import pandas            as pd

from   pathlib           import Path
from   scipy.signal      import find_peaks
from   scipy.constants   import speed_of_light as c

from   TOPSAM.schema        import DataSchema
from   TOPSAM.data          import load_raw_data, collectphotocurr
from   TOPSAM.standingwaves import fft2cav#, PATHLEN
#from   uncertainties     import ufloat


from   scipy.interpolate    import interp1d
from   scipy.constants      import speed_of_light
from   scipy.signal.windows import flattop
from   scipy.signal         import peak_widths
from   scipy.ndimage        import uniform_filter1d



class schema:
    data = "data"
    freq_ = "freq"
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
    freq = f["freq"]
    data = f[schema.data]
    
    df = pd.DataFrame({"freq": freq, 
                       schema.data: data})
    return df

def PATHLEN(df, truncate=0, PLOT=False, IFG = True):
    ### INPUT
    # df, dataframe: (freq, signal)
    ### OUTPUT
    # peak   : peak position
    # peakerr: uncertainty on peak postion
    dffft = fft2cav(df, window=True, zeropad=True).iloc[truncate:]                            # perform fft on df to find the cavity length domain 
    if IFG == True:
        i, j = int(0.3 * len(dffft[DataSchema.i])), len(dffft[DataSchema.i])
    else: 
        i, j = 0, len(dffft[DataSchema.i])
    peak_idx = np.where(dffft[DataSchema.i] == dffft[DataSchema.i][i:j].max())[0] # locate the index of the tallest peak 
    peak_width = peak_widths(dffft[DataSchema.i], peak_idx, rel_height=0.5)[0][0] # find the width of the peak so that it can be fit to
    # fit a polynomial around peak width to find max
    peak_segment = dffft.iloc[ peak_idx[0] - int(0.5 * peak_width):
                               peak_idx[0] + int(0.5 * peak_width) ]       # find all the points within the width of the peak 
    if PLOT:
        plt.figure()
        plt.plot(dffft[DataSchema.cavity_length], dffft[DataSchema.i])
        plt.scatter(dffft.iloc[peak_idx][DataSchema.cavity_length], dffft.iloc[peak_idx][DataSchema.i])
        plt.plot(peak_segment[DataSchema.cavity_length], peak_segment[DataSchema.i])
        plt.xscale("log")
        plt.show()
        
    p = np.polyfit( peak_segment[DataSchema.cavity_length],
                    peak_segment[DataSchema.i], 2)                         # fit a parabola to it 
    peak_length = -p[1] / (2 * p[0])                                       # differentiate to find the peak of the parabola
    peakerr = dffft[DataSchema.cavity_length].diff().mean()                # error based on sample size
    
    return peak_length, peakerr



def main(DIR, sam, ref, THICK, smooth): 
     # Load the data
    
    #importfile(file)

    dfsam  =  importfile(DIR+"/"+sam) 
    dfbkgd =  importfile(DIR+"/"+ref) 
    cf = 0 


        
    if smooth == 1:
        pass
    else:
        dfsam[schema.data]  = uniform_filter1d(dfsam[schema.data] ,smooth)
        dfbkgd[schema.data] = uniform_filter1d(dfbkgd[schema.data],smooth)
        
    
    PLsam , PLsamerr  = PATHLEN(df = dfsam , truncate=cf, PLOT = True, IFG = True)
    PLbkgd, PLbkgderr = PATHLEN(df = dfbkgd, truncate=cf, PLOT = True, IFG = True)


    # cut the bad bits off the bruker data
    print(cf)
    dfsam =  dfsam.iloc[cf:]
    dfbkgd= dfbkgd.iloc[cf:]
    print(dfsam)



    print("path diff: ", PLsam - PLbkgd )
    n = 2 * (PLsam - PLbkgd) / THICK[0] + 1
        
    #uncertainties:
    dnds = (  2/THICK[0] * PLsamerr  )**2
    dndb = ( -2/THICK[0] * PLbkgderr )**2
    dndt = ( -2*(PLsam-PLbkgd)*(THICK[0]**-2) * THICK[1] )**2
    dn = (dnds+dndb+dndt)**0.5
    return n, dn
    
if __name__ == "__main__":

    DIR = "C:/Users/matth/OneDrive - Cardiff University/PhD/JP_Sapphire-testing/Data/GBand/Alumina/0deg"
    
    sam = "D505-Alumina-2-3-ppol-inc7-20260115-1_1000Hz_G_band_N_1001_20260115150701.npz"
    ref = "Air-before-ppol-inc7-20260115-1_1000Hz_G_band_N_1001_20260115150453.npz"
    
    #Cycle 2
    THICK =  ((3.082e-3,0.05e-3),) 
    files  =  ("20260115150701_Alumina_0deg",)
    
    smooth = 1
    
 
    for  i in range(len(files)):
        n, de = main(DIR,sam, ref, THICK[i], smooth=smooth)
        print( files[i], ": ", n, de)

"""
