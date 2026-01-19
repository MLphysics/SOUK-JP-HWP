# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 06:15:33 2026

@author: matth
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.signal import find_peaks

# Band definition

class Band:
    def __init__(self, name, lf, uf):
        self.name = name
        self.lf = lf
        self.uf = uf


Gband = Band("Gband", 140, 220)
Fband = Band("Fband", 90, 140)



def importfile(file: Path) -> pd.DataFrame:
    f = np.load(file)
    return pd.DataFrame({
        "freq": f["freq"],
        "data": f["data"]   # assumed dB (e.g. MLOG)
    })


def resample(df: pd.DataFrame, nf: np.ndarray) -> pd.DataFrame:
    newdata = np.interp(nf, df["freq"], df["data"])
    return pd.DataFrame({
        "freq": nf,
        "data": newdata
    })


def cleandata(df: pd.DataFrame) -> pd.DataFrame:
    data_smoothed = df["data"].rolling(window=1, min_periods=1).mean()
    return pd.DataFrame({
        "freq": df["freq"],
        "data": data_smoothed
    })


def transmissioncalc_dB(dfsam: pd.DataFrame,
                        dfref: pd.DataFrame,
                        freq: np.ndarray) -> pd.DataFrame:
    
    #Transmission calculated in dB space.
    #This avoids amplifying small VNA drifts.
    
    return pd.DataFrame({
        "freq": freq,
        "trans_dB": dfsam["data"] - dfref["data"]
    })


def savedata(newpath: Path,
             material: str,
             band: str,
             date: str,
             angle: float,
             df: pd.DataFrame) -> None:

    newpath.mkdir(parents=True, exist_ok=True)
    filename = f"{band}_trans_{angle:.1f}deg_{material}_{date}.txt"
    df.to_csv(newpath / filename, sep=",", index=False)

def fit_peak_quadratic(x, y, idx, window=5):
    #Fits a quadratic to points around a peak and returns interpolated peak position and height.
    
    i0 = max(idx - window, 0)
    i1 = min(idx + window + 1, len(x))

    xloc = x[i0:i1]
    yloc = y[i0:i1]

    # Fit y = ax^2 + bx + c
    a, b, c = np.polyfit(xloc, yloc, 2)

    # Vertex of parabola
    x_peak = -b / (2 * a)
    y_peak = c - b**2 / (4 * a)

    xline = xloc
    yline = a*xline**2 + b*xline + c

    idxx = np.argsort(xline)
    
    xline=xline[idxx]
    yline=yline[idxx]

    return x_peak, y_peak, xline, yline 


def main(pathref, scan, material, date, SUBFOLD, band, INTERPPEAKS = True):

    # Common frequency space
    freq_common = np.linspace(band.lf, band.uf, 1000)
    base_dir = Path.cwd()

    # Reference
    ref_path = base_dir / "Data" / band.name / pathref
    dfref = importfile(ref_path)
    dfref = resample(dfref, freq_common)
    dfref = cleandata(dfref)

    #  Samples 
    scanfold = f"{date}_{scan}"
    sample_dir = base_dir / "Data" / band.name / scanfold / SUBFOLD

    # Stable, deterministic ordering
    pathlist = sorted(sample_dir.glob("**/*.npz"))
    n_lines = len(pathlist)

    colors = plt.cm.viridis(np.linspace(0, 1, n_lines)) # supposably more intuitive
    #colors = plt.cm.autumn(np.linspace(0, 1, n_lines)) # 2 colours 
    
    angle_arr = []
    avg_trans_dB = []

    ###################
    # Transmission vs frequency
    ###################
    plt.figure(figsize=(9, 6))
    plt.title("Transmission vs Frequency")

    for color, path in zip(colors, pathlist):

        filename = path.name
        angle_str = filename.split("_")[2]
        angle_deg = float(angle_str.replace("deg", ""))

        dfsam = importfile(path)
        dfsam = resample(dfsam, freq_common)
        dfsam = cleandata(dfsam)

        dft = transmissioncalc_dB(dfsam, dfref, freq_common)

        # Save per-angle transmission
        outdir = base_dir / "Data" / "Transmission"
        savedata(outdir, material, band.name, date, angle_deg, dft)

        # Average in dB space 
        avg_trans_dB.append(np.mean(dft["trans_dB"]))
        angle_arr.append(angle_deg)

        plt.plot(
            dft["freq"],
            dft["trans_dB"],
            color=color,
            label=f"{angle_deg:.1f}°"
        )

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission (dB)")
    plt.grid(True)
    plt.legend(title="Angle", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()
    ###################
    # Transmission Vs Frequency end
    ###################

    # order the data
    angle_arr = np.array(angle_arr)   ### make sure this is arr not strings 
    avg_trans_dB = np.array(avg_trans_dB)
    
    idx_ang = np.argsort(angle_arr)
    
    angle_arr    = angle_arr[idx_ang]
    avg_trans_dB = avg_trans_dB[idx_ang]

    ###################
    ### finding peaks 
    ###################
    # angle_arr, avg_trans_dB
    peaks, props = find_peaks(
        avg_trans_dB,
        prominence=0.2,     # dB 
        distance=10         # minimum # of points between peaks
        )

    peak_ang  = angle_arr[peaks]
    peak_vals = avg_trans_dB[peaks]

    if INTERPPEAKS == True:
        # interpolation 
        peak_ang_interp = []
        peak_vals_interp = []  ## peak oositions
        
        peak_ang_interp_line = []
        peak_vals_interp_line = []  ## peak oositions
    
        for idx in peaks:
            xp, yp, xline, yline = fit_peak_quadratic(angle_arr, avg_trans_dB, idx, window=5) # I.d. your peaks
            peak_ang_interp.append(xp)
            peak_vals_interp.append(yp) # put them in the array for plotting
            peak_ang_interp_line.append(xline)
            peak_vals_interp_line.append(yline) # put them in the array for plotting
        
        
        peak_ang_interp = np.array(peak_ang_interp)
        peak_vals_interp = np.array(peak_vals_interp)## MAKE SURE THEYRE AN ARRAY OTHERWISE IT CRASHES 
    
    ###################
    ### peaks found 
    ###################

    ###################
    # Average transmission vs angle plot start
    ###################



    plt.figure(figsize=(7, 5))
    plt.title("Average Transmission vs Angle")
    
    plt.scatter(peak_ang , peak_vals, marker = "x", c= "r", label="peaks") ### add peaks to our plot
    
    
    # Label raw peak points
    for x, y in zip(peak_ang, peak_vals):
        plt.annotate(
            f"({x:.1f}°, {y:.2f} dB)",
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="red"
        )
    
    if INTERPPEAKS == True:
        plt.scatter(peak_ang_interp, peak_vals_interp, marker = "x", c= "black", alpha= 0.3, label="peaks from quadratic") ### add peaks to our plot
        
        idxx = np.argsort(np.array(peak_ang_interp_line))
        
        peak_ang_interp_line=np.array(peak_ang_interp_line)[idxx]
        peak_vals_interp_line=np.array(peak_vals_interp_line)[idxx]
        
        plt.plot(peak_ang_interp_line, peak_vals_interp_line, c="black", alpha=0.3)
        
        
        # Label quadratic-interpolated peak points
        for x, y in zip(peak_ang_interp, peak_vals_interp):
            plt.annotate(
                f"({x:.2f}°, {y:.2f} dB)",
                xy=(x, y),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
                color="black",
                alpha=0.3
            )

    
    
    
    plt.plot(
        angle_arr,
        avg_trans_dB,
        marker="o",
        ms=3,
        linestyle="-"
    )
    
    
    
    plt.xlabel("Angle (deg)")
    plt.ylabel("Average Transmission (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
    ###################
    # Average transmission vs angle plot start
    ###################
    





### RUN HERE 
main(
    pathref=Path.cwd()
    / "Data"
    / "GBand"
    / "20260116_D505mm-Alumina"
    / "Air-after-ppol-20260116-1_1000Hz_G_band_N_1001_20260116133606.npz",
    scan="MF2_rotation_test",
    material="sapphire_MF2",
    date="20260116",
    SUBFOLD="MLOG",
    band=Gband
)



