# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 06:15:33 2026

@author: matth
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


# ------------------------
# Band definition
# ------------------------
class Gband:
    name = "Gband"
    bandletter = name
    lf = 140
    uf = 220


# ------------------------
# I/O and processing
# ------------------------
def importfile(file: Path) -> pd.DataFrame:
    f = np.load(file)
    return pd.DataFrame({
        "freq": f["freq"],
        "data": f["data"]
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


def simpleplot(dfsam: pd.DataFrame, dfref: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(dfref["freq"], dfsam["data"] / dfref["data"])
    plt.xlabel("Frequency")
    plt.ylabel("Transmission")
    plt.show()


def transmissioncalc(dfsam: pd.DataFrame,
                     dfref: pd.DataFrame,
                     freq: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "freq": freq,
        "trans": 10 ** ((dfsam["data"] - dfref["data"])/10)
    })


def savedata(newpath: Path,
             material: str,
             band: str,
             date: str,
             angle: str,
             df: pd.DataFrame) -> None:

    newpath.mkdir(parents=True, exist_ok=True)
    filename = f"{band}_trans_{angle}_{material}_{date}.txt"
    df.to_csv(newpath / filename, sep=",", index=False)


# ------------------------
# Main workflow
# ------------------------
def main(pathref, scan, material, date, SUBFOLD, band):
    # Common frequency space
    freq_common = np.linspace(band.lf, band.uf, 1000)

    base_dir = Path.cwd()

    # ----- Reference -----
    ref_path = base_dir / "Data" / band.name / pathref
    dfref = importfile(ref_path)
    dfref = resample(dfref, freq_common)
    dfref = cleandata(dfref)

    # ----- Samples -----
    scanfold = f"{date}_{scan}"
    sample_dir = base_dir / "Data" / band.name / scanfold / SUBFOLD

    pathlist = sample_dir.glob("**/*.npz")

    plt.figure()
    for path in pathlist:
        print(path)

        filename = path.name
        angle = filename.split("_")[2]

        dfsam = importfile(path)
        dfsam = resample(dfsam, freq_common)
        dfsam = cleandata(dfsam)

        dft = transmissioncalc(dfsam, dfref, freq_common)

        outdir = base_dir / "Data" / "Transmission"
        savedata(outdir, material, band.name, date, angle, dft)
    
    
        plt.plot(dfsam["freq"], dfsam["data"])
        plt.plot(dft["freq"], dft["trans"])
    plt.show()

# ------------------------
# Run
# ------------------------
main(
    pathref="C:/Users/matth/OneDrive/Documents/PhD Cardiff/Git repos/SOUK-JP-HWP-/Data/GBand/20260116_D505mm-Alumina/Air-after-ppol-20260116-1_1000Hz_G_band_N_1001_20260116133606.npz",
    scan="MF2_rotation_test",
    material="sapphire_MF2",
    date="20260116",
    SUBFOLD="MLOG",  # or "detailed"
    band=Gband
)
