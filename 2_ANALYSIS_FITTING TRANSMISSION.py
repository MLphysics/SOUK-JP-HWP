# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 06:15:33 2026

@author: matth
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path



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


def main(pathref, scan, material, date, SUBFOLD, band):

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


    # Transmission vs frequency

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

        # Average in dB space (CRITICAL FIX)
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


    # Average transmission vs angle

    angle_arr = np.array(angle_arr)   ### make sure this is arr not strings 
    avg_trans_dB = np.array(avg_trans_dB)

   
    idx = np.argsort(angle_arr)  # Sort by angle

    plt.figure(figsize=(7, 5))
    plt.title("Average Transmission vs Angle")
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
