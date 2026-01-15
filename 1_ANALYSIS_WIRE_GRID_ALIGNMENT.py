# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 04:16:37 2026

@author: matth
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

# default is unchanged angle of the WG1

directory_in_str = "C:/Users/matth/OneDrive - Cardiff University/PhD/JP_Sapphire-testing/Data/Wiregrid2"



dataaverage= [] 
ang = np.arange(270,280+1)
ang = np.append(ang, 273)
    
plt.figure()
pathlist = Path(directory_in_str).glob('**/*.npz')
for path in pathlist:
        # because path is object not string
        path_in_str = str(path)   
        file = np.load(path_in_str)
        # print(path_in_str)
        print(path_in_str)
        freq = file["freq"]
        data = file ["data"] 
        
        dataAv = np.average(data)
        
        dataaverage = np.append(dataaverage, dataAv)
        
        plt.plot(freq, data)
plt.show()
#dataref = dataaverage[0]
#dataaverage = dataaverage[1:]
    
print(len(ang))
print(len(dataaverage))
#print(dataref)
    
plt.figure()
plt.scatter(ang, dataaverage)
    
plt.ylabel("dB")
plt.xlabel("Angle Degree")
plt.grid()
plt.show()
    
