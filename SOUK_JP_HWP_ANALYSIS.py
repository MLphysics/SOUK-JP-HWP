# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 06:15:33 2026

@author: matth
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re


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
    data_smoothed = df["data"].rolling(window=3, min_periods=1).mean()
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


def main(pathref, scan, material, date, SUBFOLD, title, band):

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

    # Collect (angle, path) pairs
    angle_path_pairs = []

    for path in sample_dir.glob("**/*.npz"):
        match = re.search(r"([\d.]+)deg", path.name)
        if not match:
            raise ValueError(f"Angle not found in filename: {path.name}")
        angle = float(match.group(1))
        angle_path_pairs.append((angle, path))

    # Sort by angle
    angle_path_pairs.sort(key=lambda x: x[0])

    # Unpack
    angles_sorted = [ap[0] for ap in angle_path_pairs]
    paths_sorted  = [ap[1] for ap in angle_path_pairs]

    n_lines = len(paths_sorted)
    colors = plt.cm.viridis(np.linspace(0, 1, n_lines))

    angle_arr = []
    avg_trans_dB = []


    # Transmission vs frequency

    plt.figure(figsize=(9, 6))
    plt.title("Transmission vs Frequency - " + title)

    for color, angle_deg, path in zip(colors, angles_sorted, paths_sorted):


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

    norm = mcolors.Normalize(vmin=min(angles_sorted), vmax=max(angles_sorted))
    sm = cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    ax = plt.gca()
    plt.colorbar(sm, ax=ax, label="Angle (deg)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Average transmission vs angle

    angle_arr = np.array(angle_arr)   ### make sure this is arr not strings 
    avg_trans_dB = np.array(avg_trans_dB)

   
    idx = np.argsort(angle_arr)  # Sort by angle

    plt.figure(figsize=(7, 5))
    plt.title("Average Transmission vs Angle - " + title)
    plt.plot(
        angle_arr[idx],
        avg_trans_dB[idx],
        marker="o",
        linestyle="-"
    )
    plt.xlabel("Angle (deg)")
    plt.ylabel("Average Transmission (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



