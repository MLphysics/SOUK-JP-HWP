# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 06:15:33 2026

@author: matth
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from enum import Enum
from scipy.signal import find_peaks
from matplotlib.lines import Line2D

#####
# Band definition

class Band:
    def __init__(self, name, lf, uf):
        self.name = name
        self.lf = lf
        self.uf = uf

Gband = Band("Gband", 140, 220)
Fband = Band("Fband", 90, 140)

#####
# Transmission scale

class TransScale(Enum):
    DB = "dB"
    LINEAR = "linear"

def dB_to_linear(db):
    return 10 ** (db / 20)

#####
#  preprocessing

def importfile(file: Path) -> pd.DataFrame:
    f = np.load(file)
    return pd.DataFrame({"freq": f["freq"], "data": f["data"]})

def resample(df, nf):
    return pd.DataFrame({
        "freq": nf,
        "data": np.interp(nf, df["freq"], df["data"])
    })

def cleandata(df):
    return pd.DataFrame({
        "freq": df["freq"],
        "data": df["data"].rolling(window=1, min_periods=1).mean()
    })

def transmissioncalc_dB(dfsam, dfref, freq):
    return pd.DataFrame({
        "freq": freq,
        "trans_dB": dfsam["data"] - dfref["data"]
    })

def savedata(newpath, material, band, date, angle, df):
    newpath.mkdir(parents=True, exist_ok=True)
    filename = f"{band}_trans_{angle:.1f}deg_{material}_{date}.txt"
    df.to_csv(newpath / filename, index=False)

#####
# Peak fitting
def fit_peak_quadratic(x, y, idx, window=5):
    i0 = max(idx - window, 0)
    i1 = min(idx + window + 1, len(x))

    xloc = x[i0:i1]
    yloc = y[i0:i1]

    a, b, c = np.polyfit(xloc, yloc, 2)

    x_peak = -b / (2 * a)
    y_peak = c - b**2 / (4 * a)

    xline = np.linspace(x[i0], x[i1], 100)
    yline = a * xline**2 + b * xline + c

    return x_peak, y_peak, xline, yline

def process_scan(sample_paths, dfref, freq_common, TRANS_SCALE, INTERPPEAKS=True, detail = False):
    """Process a list of sample files and return average transmission and peaks."""
    angle_arr = []
    avg_trans_dB = []
    trans_curves = []

    for path in sample_paths:
        angle = float(path.name.split("_")[2].replace("deg", ""))
        dfsam = cleandata(resample(importfile(path), freq_common))
        dft = transmissioncalc_dB(dfsam, dfref, freq_common)

        trans_curves.append(dft["trans_dB"].values)
        angle_arr.append(angle)
        avg_trans_dB.append(np.mean(dft["trans_dB"]))

    angle_arr = np.array(angle_arr)
    trans_curves = np.array(trans_curves)
    avg_trans_dB = np.array(avg_trans_dB)

    idx = np.argsort(angle_arr)
    angles_sorted = angle_arr[idx]
    trans_curves_sorted = trans_curves[idx]
    avg_trans_dB_sorted = avg_trans_dB[idx]

    if detail == False:
        peaks, _ = find_peaks(avg_trans_dB_sorted, prominence=0.2, distance=10)
    if detail == True:
        peaks, _ = find_peaks(avg_trans_dB_sorted, prominence=0.00002, distance=1)
    peak_angles = angles_sorted[peaks]
    peak_vals = avg_trans_dB_sorted[peaks]

    quad_fits = []
    if INTERPPEAKS:
        for i in peaks:
            quad_fits.append(fit_peak_quadratic(angles_sorted, avg_trans_dB_sorted, i))

    yavg = avg_trans_dB_sorted if TRANS_SCALE == TransScale.DB else dB_to_linear(avg_trans_dB_sorted)
    ypeaks = peak_vals if TRANS_SCALE == TransScale.DB else dB_to_linear(peak_vals)

    return angles_sorted, yavg, peak_angles, ypeaks, quad_fits

# MAIN
def main(scan, material, date, SUBFOLD, band,
         INTERPPEAKS=True,
         TRANS_SCALE=TransScale.DB):

    base_dir = Path.cwd()
    freq_common = np.linspace(band.lf, band.uf, 1000)
    scanfold = f"{date}_{scan}"

    #####
    # Samples
    sample_dir = base_dir / "Data" / band.name / scanfold / SUBFOLD
    sample_paths = sorted(sample_dir.glob("*.npz"))

    #####
    # Reference (average air)
    ref_dir = base_dir / "Data" / band.name / scanfold
    ref_paths = sorted(ref_dir.glob("*.npz"))
    print(ref_dir)
    if not ref_paths:
        raise RuntimeError("No reference (.npz) files found")

    ref_stack = []
    for p in ref_paths:
        df = cleandata(resample(importfile(p), freq_common))
        ref_stack.append(df["data"].values)

    dfref = pd.DataFrame({
        "freq": freq_common,
        "data": np.mean(np.vstack(ref_stack), axis=0)
    })

    ##### -----------------------------------------------------------------------
    # Transmission vs frequency
    angle_arr = []
    avg_trans_dB = []
    trans_curves = []

    for path in sample_paths:
        angle = float(path.name.split("_")[2].replace("deg", ""))
        dfsam = cleandata(resample(importfile(path), freq_common))
        dft = transmissioncalc_dB(dfsam, dfref, freq_common)

        trans_curves.append(dft["trans_dB"].values)
        angle_arr.append(angle)
        avg_trans_dB.append(np.mean(dft["trans_dB"]))

    angle_arr = np.array(angle_arr)
    trans_curves = np.array(trans_curves)
    avg_trans_dB = np.array(avg_trans_dB)

    # Sort by angle
    idx = np.argsort(angle_arr)
    angles = angle_arr[idx]
    trans_curves = trans_curves[idx]
    avg_trans_dB = avg_trans_dB[idx]

    # Find peaks for opacity
    peaks, _ = find_peaks(avg_trans_dB, prominence=0.2, distance=10)
    is_peak_angle = np.zeros_like(angles, dtype=bool)
    is_peak_angle[peaks] = True

    # Colormap setup
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=angles.min(), vmax=angles.max())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"{scan}: Transmission vs Frequency")

    peak_legend_handles = []
    for angle, trans_dB, is_peak in zip(angles, trans_curves, is_peak_angle):
        yplot = trans_dB if TRANS_SCALE == TransScale.DB else dB_to_linear(trans_dB)
        color = cmap(norm(angle))
        ax.plot(
            freq_common,
            yplot,
            ls =":" if is_peak else "-",
            color="r" if is_peak else color,
            lw=2.5 if is_peak else 1.5,
            alpha=1.0 if is_peak else 0.2,
            zorder=3 if is_peak else 1
        )
        if is_peak:
            peak_legend_handles.append(
                Line2D([0], [0], color=color, lw=2.5, label=f"Peak spectrum: {angle:.2f}°")
            )

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21| (linear)")
    plt.grid(True)

    # ---- Colorbar ----
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Angle (deg)")

    # ---- Legend for opacity + peaks ----
    opacity_handles = [
        Line2D([0], [0], color="r", ls=":", lw=2.5, alpha=1.0, label="Peak-angle spectra"),
        Line2D([0], [0], color="k", ls="-", lw=1.5, alpha=0.2, label="Non-peak spectra"),
    ]
    legend_handles = opacity_handles + peak_legend_handles
    ax.legend(handles=legend_handles, title="Transmission spectra", fontsize=8, title_fontsize=9, loc="best")

    plt.tight_layout()
    plt.show()

    ##### -----------------------------------------------------------------------
    # Average transmission vs angle
    angles_sorted, yavg, peak_ang, ypeaks, quad_fits = process_scan(sample_paths, dfref, freq_common, TRANS_SCALE, INTERPPEAKS)

    plt.figure(figsize=(7, 5))
    plt.title(f"{scan}: Average Transmission vs Angle")

    plt.plot(angles_sorted, yavg, "-o", ms=3, label="Average transmission")
    plt.scatter(peak_ang, ypeaks, c="r", marker="x", label="Raw peaks")

    # RAW PEAK ANNOTATIONS
    for x, y_db in zip(peak_ang, ypeaks):
        plt.annotate(
            f"({x:.1f}°, {y_db:.2f} dB)",
            xy=(x, y_db),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="red"
        )

    # QUADRATIC PEAKS + ANNOTATIONS
    if INTERPPEAKS:
        for xp, yp, xline, yline in quad_fits:
            yline_plot = yline if TRANS_SCALE == TransScale.DB else dB_to_linear(yline)
            plt.plot(xline, yline_plot, "k", alpha=0.3, lw=4)
            yp_plot = yp if TRANS_SCALE == TransScale.DB else dB_to_linear(yp)
            plt.scatter(xp, yp_plot, c="black", alpha=0.4)
            plt.annotate(
                f"({xp:.2f}°, {yp:.2f} dB)",
                xy=(xp, yp_plot),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
                color="black",
                alpha=0.5
            )

    # ---- Check for detailed folder ----
    detailed_dir = base_dir / "Data" / band.name / scanfold / "detailed" / SUBFOLD
    
    if detailed_dir.exists():
        detailed_paths = sorted(detailed_dir.glob("*.npz"))
        if detailed_paths:
            angles_d, yavg_d, peak_ang_d, ypeaks_d, quad_fits_d = process_scan(
                detailed_paths, dfref, freq_common, TRANS_SCALE, INTERPPEAKS
            )
            plt.plot(angles_d, yavg_d, "-s", ms=3, label="Detailed scan")
            plt.scatter(peak_ang_d, ypeaks_d, c="b", marker="x", label="Detailed raw peaks")
            
            # Annotate detailed raw peaks
            for x, y_db in zip(peak_ang_d, ypeaks_d):
                plt.annotate(
                    f"({x:.1f}°, {y_db:.2f} dB)",
                    xy=(x, y_db),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="blue"
                )
    
            # Quadratic-interpolated peaks for detailed scan
            if INTERPPEAKS:
                for xp, yp, xline, yline in quad_fits_d:
                    yline_plot = yline if TRANS_SCALE == TransScale.DB else dB_to_linear(yline)
                    plt.plot(xline, yline_plot, "b", alpha=0.3, lw=4)
                    yp_plot = yp if TRANS_SCALE == TransScale.DB else dB_to_linear(yp)
                    plt.scatter(xp, yp_plot, c="darkblue", alpha=0.4)
                    plt.annotate(
                        f"detailed interp: ({xp:.2f}°, {yp:.2f} dB)",
                        xy=(xp, yp_plot),
                        xytext=(5, -10),
                        textcoords="offset points",
                        fontsize=8,
                        color="black",
                        alpha=0.5
                    )


    plt.xlabel("Angle (deg)")
    plt.ylabel("Avg Transmission (dB)" if TRANS_SCALE == TransScale.DB else "Avg |S21| (linear)")
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
