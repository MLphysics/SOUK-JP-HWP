# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 05:18:41 2026

@author: matth
"""

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

WG1= "C:/Users/matth/OneDrive - Cardiff University/PhD/JP_Sapphire-testing/Data/Wiregrid1"
WG1= "C:/Users/matth/OneDrive - Cardiff University/PhD/JP_Sapphire-testing/Data/Wiregrid2"



def wiregridplot(directory_in_str, WG=1):
    # Define angles (currently identical for WG 1 and 2)
    ang = np.arange(95, 102)
    ang = np.append(ang, 94.5)

    dataaverage = []
    freqs = []

    plt.figure()

    pathlist = sorted(Path(directory_in_str).glob("**/*.npz"))

    if len(pathlist) == 0:
        raise ValueError("No .npz files found in directory")

    for path in pathlist:
        file = np.load(path)
        freq = file["freq"]
        data = file["data"]

        freqs.append(freq)
        dataaverage.append(np.mean(data))

        plt.plot(freq, data, alpha=0.5)

    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Wiregrid Spectra")
    plt.show()

    dataaverage = np.array(dataaverage)

    # Use first file as reference
    dataref = dataaverage[0]
    dataaverage = dataaverage[1:]

    if len(ang) != len(dataaverage):
        raise ValueError(
            f"Angle count ({len(ang)}) does not match data count ({len(dataaverage)})"
        )

    print("Angles:", ang)
    print("Averages:", dataaverage)
    print("Reference:", dataref)

    plt.figure()
    plt.scatter(ang, dataaverage)
    plt.xlabel("Angle (degrees)")

    

wiregridplot(WG1, WG=1)
#wiregridplot(PATH=WG2, WG=2)