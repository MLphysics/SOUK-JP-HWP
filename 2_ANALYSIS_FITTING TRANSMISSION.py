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
from scipy.signal import find_peaks, peak_widths
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

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

# ===== Updated: Fit quadratic and return x_peak uncertainty =====
def fit_peak_quadratic(x, y, idx, window=5):
    window = min(window, idx, len(x)-idx-1)
    i0 = max(idx - window, 0)
    i1 = min(idx + window + 1, len(x))
    xloc = x[i0:i1]
    yloc = y[i0:i1]
    
    a, b, c = np.polyfit(xloc, yloc, 2)
    x_peak = -b / (2 * a)
    y_peak = c - b**2 / (4 * a)
    xline = np.linspace(xloc[0], xloc[-1], 100)
    yline = a * xline**2 + b * xline + c

    # Error propagation: variance of residuals
    residuals = yloc - (a * xloc**2 + b * xloc + c)
    sigma_y2 = np.sum(residuals**2) / (len(xloc) - 3)
    Sxx = np.sum((xloc - np.mean(xloc))**2)
    sigma_x = np.sqrt(sigma_y2 / (4 * a**2 * Sxx))
    
    return x_peak, y_peak, xline, yline, sigma_x

def process_scan(sample_paths, dfref, freq_common, TRANS_SCALE,
                 INTERPPEAKS=True, detail=False, trimfunnydata=False):
    """Process sample files and return avg transmission and peaks."""

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
    idx_sort = np.argsort(angle_arr)
    angles_sorted = angle_arr[idx_sort]
    trans_curves_sorted = trans_curves[idx_sort]
    avg_trans_dB_sorted = avg_trans_dB[idx_sort]

    if detail and trimfunnydata:
        trim_threshold = -4.0
        rm_idx = np.where(avg_trans_dB_sorted < trim_threshold)[0]
        if len(rm_idx) > 0:
            angles_sorted = np.delete(angles_sorted, rm_idx)
            avg_trans_dB_sorted = np.delete(avg_trans_dB_sorted, rm_idx)
            trans_curves_sorted = np.delete(trans_curves_sorted, rm_idx, axis=0)
            print("# of points removed in trimming: ", len(rm_idx))

    if detail:
        peaks, _ = find_peaks(avg_trans_dB_sorted, distance=3, height=(None, 10))
    else:
        peaks, _ = find_peaks(avg_trans_dB_sorted, prominence=0.2, distance=10)

    peak_angles = angles_sorted[peaks]
    peak_vals = avg_trans_dB_sorted[peaks]

    quad_fits = []

    if INTERPPEAKS:
        if detail:
            n = len(avg_trans_dB_sorted)
            seg_edges = np.linspace(0, n, 5, dtype=int)
            for s in range(4):
                seg_idx = range(seg_edges[s], seg_edges[s+1])
                seg_peaks = [i for i in peaks if i in seg_idx]
                for i in seg_peaks:
                    quad_fits.append(fit_peak_quadratic(
                        angles_sorted, avg_trans_dB_sorted, i,
                        window=max(3, len(seg_idx)//4)
                    ))
        else:
            for i in peaks:
                quad_fits.append(fit_peak_quadratic(
                    angles_sorted, avg_trans_dB_sorted, i
                ))

    yavg = avg_trans_dB_sorted if TRANS_SCALE == TransScale.DB else dB_to_linear(avg_trans_dB_sorted)
    ypeaks = peak_vals if TRANS_SCALE == TransScale.DB else dB_to_linear(peak_vals)

    return angles_sorted, yavg, peak_angles, ypeaks, quad_fits, trans_curves_sorted

def create_peak_table(peak_ang_5, ypeaks_5, quad_fits_5, peak_ang_d, ypeaks_d, quad_fits_d,
                      peak_labels_5=None, peak_labels_d=None):
    n_peaks = max(len(peak_ang_5), len(quad_fits_5), len(peak_ang_d), len(quad_fits_d))
    
    data = {}
    for i in range(n_peaks):
        raw_5 = f"({peak_ang_5[i]:.2f}°, {ypeaks_5[i]:.2f} dB)" if i < len(peak_ang_5) else "-"
        if peak_labels_5 and i < len(peak_ang_5):
            raw_5 = raw_5 + f" [{peak_labels_5[peak_ang_5[i]]}]"

        interp_5 = (f"({quad_fits_5[i][0]:.2f}±{quad_fits_5[i][4]:.2f}°, {quad_fits_5[i][1]:.2f} dB)"
                    if i < len(quad_fits_5) else "-")
        raw_1 = f"({peak_ang_d[i]:.2f}°, {ypeaks_d[i]:.2f} dB)" if i < len(peak_ang_d) else "-"
        if peak_labels_d and i < len(peak_ang_d):
            raw_1 = raw_1 + f" [{peak_labels_d[peak_ang_d[i]]}]"

        interp_1 = (f"({quad_fits_d[i][0]:.2f}±{quad_fits_d[i][4]:.2f}°, {quad_fits_d[i][1]:.2f} dB)"
                    if i < len(quad_fits_d) else "-")
        data[f"Peak {i+1}"] = [raw_5, interp_5, raw_1, interp_1]
    
    table = pd.DataFrame(data, index=["5° raw", "5° interp", "1° raw", "1° interp"])
    return table



def plot_3d_transmission(angles, freq, trans_curves, scan, TRANS_SCALE=TransScale.DB):
    """
    Plot Transmission vs Frequency vs Angle in a 3D surface/mesh.
    
    angles: 1D array of angles (deg)
    freq: 1D array of frequencies (GHz)
    trans_curves: 2D array, rows=angles, columns=frequencies
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Make meshgrid for surface plot
    FREQ, ANG = np.meshgrid(freq, angles)
    
    # Transmission values
    Z = trans_curves if TRANS_SCALE == TransScale.DB else dB_to_linear(trans_curves)
    
    # Surface plot
    surf = ax.plot_surface(FREQ, ANG, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Angle (deg)")
    ax.set_zlabel("Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21| (linear)")
    ax.set_title(f"{scan}: Transmission vs Frequency vs Angle (3D)")
    
    fig.colorbar(surf, ax=ax, shrink=0.7, label="Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21|")
    plt.tight_layout()
    plt.show()




# -----------------------------
# Measure peak widths and label as ordinary/extraordinary
# -----------------------------
def measure_peak_widths_freq(freq, trans_curves, angles, peak_angles, TRANS_SCALE=TransScale.DB):
    """
    Measure peak widths along frequency for each peak angle.

    Parameters:
    -----------
    freq : np.array
        Frequencies (GHz)
    trans_curves : 2D np.array
        Transmission curves, rows=angles, cols=frequencies
    angles : np.array
        Angles corresponding to trans_curves rows
    peak_angles : np.array
        Peak angles detected from average transmission vs angle
    TRANS_SCALE : TransScale
        DB or linear scaling

    Returns:
    --------
    peak_labels : dict
        Dictionary mapping angle -> 'O' or 'E'
    peak_widths_dict : dict
        Dictionary mapping angle -> width in GHz
    """
    peak_widths_dict = {}

    # Convert to linear if needed for width measurement
    trans_curves_lin = trans_curves if TRANS_SCALE == TransScale.LINEAR else dB_to_linear(trans_curves)

    for pa in peak_angles:
        # Find index of angle
        idx = np.argmin(np.abs(angles - pa))
        y = trans_curves_lin[idx]

        # Detect frequency peak (assuming 1 main peak per curve)
        f_peaks, _ = find_peaks(y, prominence=0.1)
        if len(f_peaks) == 0:
            # fallback: take max
            peak_idx = np.argmax(y)
        else:
            # closest peak to max value
            peak_idx = f_peaks[np.argmax(y[f_peaks])]

        # Measure width at half-max
        results = peak_widths(y, [peak_idx], rel_height=0.5)
        width_GHz = results[0][0] * (freq[1] - freq[0])
        peak_widths_dict[pa] = width_GHz

    # Assign labels: two largest widths -> extraordinary, two smallest -> ordinary
    width_sorted_idx = np.argsort(list(peak_widths_dict.values()))
    peak_labels = {}
    angles_sorted = list(peak_widths_dict.keys())
    for i, idx in enumerate(width_sorted_idx):
        angle = angles_sorted[idx]
        label = 'E' if i < 2 else 'O'
        peak_labels[angle] = label

    return peak_labels, peak_widths_dict



# =====================
# MAIN
# =====================

def main(scan, material, date, SUBFOLD, band,
         INTERPPEAKS=True,
         TRANS_SCALE=TransScale.DB,
         trimfunnydata=False,
         ANNOTE_RAW=True,
         ANNOTE_INTERP=True):

    base_dir = Path.cwd()
    freq_common = np.linspace(band.lf, band.uf, 1000)
    scanfold = f"{date}_{scan}"

    # ======================
    # Reference files
    # ======================
    ref_dir = base_dir / "Data" / band.name / scanfold
    ref_paths = sorted(ref_dir.glob("*.npz"))
    if not ref_paths:
        raise RuntimeError("No reference files found")

    ref_stack = []
    for p in ref_paths:
        df = cleandata(resample(importfile(p), freq_common))
        ref_stack.append(df["data"].values)

    dfref = pd.DataFrame({
        "freq": freq_common,
        "data": np.mean(np.vstack(ref_stack), axis=0)
    })

    # ======================
    # Sample files
    # ======================
    sample_dir = base_dir / "Data" / band.name / scanfold / SUBFOLD
    sample_paths = sorted(sample_dir.glob("*.npz"))

    # ======================
    # Process 5° scans first
    # ======================
    angles_5, yavg_5, peak_ang_5, ypeaks_5, quad_fits_5, trans_curves_sorted = process_scan(
        sample_paths, dfref, freq_common, TRANS_SCALE, INTERPPEAKS, detail=False, trimfunnydata=trimfunnydata
    )

    # ======================
    # Process detailed 1° scans (if available)
    # ======================
    detailed_dir = base_dir / "Data" / band.name / scanfold / "detailed" / SUBFOLD
    detailed_paths = []
    angles_d, trans_curves_d, peak_ang_d, ypeaks_d, quad_fits_d = ([], [], [], [], [])
    if detailed_dir.exists():
        detailed_paths = sorted(detailed_dir.glob("*.npz"))
        if detailed_paths:
            angles_d, yavg_d, peak_ang_d, ypeaks_d, quad_fits_d, trans_curves_d = process_scan(
                detailed_paths, dfref, freq_common, TRANS_SCALE,
                INTERPPEAKS=INTERPPEAKS, detail=True, trimfunnydata=trimfunnydata
            )
    
    # ======================
    # Identify ordinary / extraordinary axes using peak widths vs frequency
    # ======================
    peak_labels_5, peak_widths_5 = measure_peak_widths_freq(
        freq_common, trans_curves_sorted, angles_5, peak_ang_5, TRANS_SCALE
    )
    
    if detailed_paths and len(peak_ang_d) > 0:
        peak_labels_d, peak_widths_d = measure_peak_widths_freq(
            freq_common, trans_curves_d, angles_d, peak_ang_d, TRANS_SCALE
        )

    
    # ======================
    # Transmission vs Frequency Plot
    # ======================
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"{scan}: Transmission vs Frequency")

    # Plot all 5° spectra (faint background)
    cmap_5 = plt.cm.viridis
    norm_5 = plt.Normalize(vmin=angles_5.min(), vmax=angles_5.max())
    for angle, trans_dB in zip(angles_5, trans_curves_sorted):
        yplot = trans_dB if TRANS_SCALE == TransScale.DB else dB_to_linear(trans_dB)
        color = cmap_5(norm_5(angle))
        ax.plot(freq_common, yplot, ls="-", color=color, lw=1.5, alpha=0.2, zorder=1)

    # Plot detailed 1° spectra (faint background)
    if detailed_paths:
        cmap_d = plt.cm.plasma
        norm_d = plt.Normalize(vmin=angles_d.min(), vmax=angles_d.max())
        for angle, trans_dB in zip(angles_d, trans_curves_d):
            yplot = trans_dB if TRANS_SCALE == TransScale.DB else dB_to_linear(trans_dB)
            color = cmap_d(norm_d(angle))
            ax.plot(freq_common, yplot, ls="-", color=color, lw=1.2, alpha=0.2, zorder=1)

    # Highlight peak-angle spectra
    if detailed_paths and len(peak_ang_d) > 0:
        peak_angles_to_plot = peak_ang_d
        trans_curves_peak = trans_curves_d
        angles_peak = angles_d
        peak_color = "r"
        peak_label_text = "Peak-angle spectra (detailed 1°)"
    else:
        peak_angles_to_plot = peak_ang_5
        trans_curves_peak = trans_curves_sorted
        angles_peak = angles_5
        peak_color = "r"
        peak_label_text = "Peak-angle spectra (5°)"

    peak_legend_handles = []
    for pa in peak_angles_to_plot:
        idx = np.where(np.isclose(angles_peak, pa))[0][0]
        yplot = trans_curves_peak[idx]
        yplot = yplot if TRANS_SCALE == TransScale.DB else dB_to_linear(yplot)
        ax.plot(freq_common, yplot, ls=":", color=peak_color, lw=2.5, alpha=1.0, zorder=3)
        peak_legend_handles.append(Line2D([0], [0], color="White", lw=2.5, label=f"Peak spectrum: {pa:.2f}°"))

    # Legend
    opacity_handles = [
        Line2D([0], [0], color="k", ls="-", lw=1.5, alpha=0.2, label="Non-peak spectra"),
        Line2D([0], [0], color=peak_color, ls=":", lw=2.5, alpha=1.0, label=peak_label_text)
    ]
    legend_handles = opacity_handles + peak_legend_handles
    ax.legend(handles=legend_handles, title="Transmission spectra", fontsize=8, title_fontsize=9, loc="best")

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21| (linear)")
    plt.grid(True)

    # Colorbar for 5° spectra
    sm = plt.cm.ScalarMappable(cmap=cmap_5, norm=norm_5)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Angle (deg)")

    plt.tight_layout()
    plt.show()

    # ======================
    # Average Transmission vs Angle
    # ======================
    plt.figure(figsize=(7, 5))
    plt.title(f"{scan}: Average Transmission vs Angle")

    plt.plot(angles_5, yavg_5, "-r", ms=3, label="5 degree")
    plt.scatter(peak_ang_5, ypeaks_5, c="r", marker="x", label="5 degree - Raw peaks")
    if ANNOTE_RAW:
        for x, y_db in zip(peak_ang_5, ypeaks_5):
            plt.annotate(f"({x:.1f}°, {y_db:.2f} dB)", xy=(x, y_db), xytext=(5,5),
                         textcoords="offset points", fontsize=8, color="red")

    if INTERPPEAKS:
        for xp, yp, xline, yline, sigma_x in quad_fits_5:
            yline_plot = yline if TRANS_SCALE == TransScale.DB else dB_to_linear(yline)
            plt.plot(xline, yline_plot, "k", alpha=0.3, lw=4)
            yp_plot = yp if TRANS_SCALE == TransScale.DB else dB_to_linear(yp)
            plt.scatter(xp, yp_plot, c="r", alpha=0.4)
            plt.errorbar(xp, yp_plot, xerr=sigma_x, fmt='none', ecolor='gray', alpha=0.6)
            if ANNOTE_INTERP:
                plt.annotate(f"({xp:.2f}±{sigma_x:.2f}°, {yp:.2f} dB)", xy=(xp, yp_plot),
                             xytext=(5,0), textcoords="offset points",
                             fontsize=8, color="black", alpha=0.5)

    # Detailed 1° scan plotting
    if detailed_paths:
        plt.scatter(angles_d, yavg_d, c="b", marker="+", label="1 degree")
        plt.scatter(peak_ang_d, ypeaks_d, c="b", marker="x", label="1 degree - Raw peaks")
        if ANNOTE_RAW:
            for x, y_db in zip(peak_ang_d, ypeaks_d):
                plt.annotate(f"({x:.1f}°, {y_db:.2f} dB)", xy=(x, y_db),
                             xytext=(5,-5), textcoords="offset points", fontsize=8, color="blue")
        if INTERPPEAKS:
            for xp, yp, xline, yline, sigma_x in quad_fits_d:
                yline_plot = yline if TRANS_SCALE == TransScale.DB else dB_to_linear(yline)
                plt.plot(xline, yline_plot, "black", alpha=0.3, lw=4)
                yp_plot = yp if TRANS_SCALE == TransScale.DB else dB_to_linear(yp)
                plt.scatter(xp, yp_plot, c="b", alpha=0.4)
                plt.errorbar(xp, yp_plot, xerr=sigma_x, fmt='none', ecolor='gray', alpha=0.6)
                if ANNOTE_INTERP:
                    plt.annotate(f"detailed interp: ({xp:.2f}±{sigma_x:.2f}°, {yp:.2f} dB)", xy=(xp, yp_plot),
                                 xytext=(5,-10), textcoords="offset points",
                                 fontsize=8, color="black", alpha=0.5)

    plt.xlabel("Angle (deg)")
    plt.ylabel("Avg Transmission (dB)" if TRANS_SCALE == TransScale.DB else "Avg |S21| (linear)")
    plt.grid()
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # ======================
    # Peak summary table
    # ======================
    if detailed_paths:
        peak_table = create_peak_table(
            peak_ang_5, ypeaks_5, quad_fits_5,
            peak_ang_d, ypeaks_d, quad_fits_d,
            peak_labels_5=peak_labels_5,
            peak_labels_d=peak_labels_d
        )
        
    else:
        peak_table = create_peak_table(
            peak_ang_5, ypeaks_5, quad_fits_5,
            peak_labels_5=peak_labels_5
        )
    print("\nPeak summary table:\n")
    print(peak_table.to_string())

    # ======================
    # 3D Transmission plot
    # ======================
    plot_3d_transmission(angles_5, freq_common, trans_curves_sorted, scan, TRANS_SCALE=TRANS_SCALE)



# =====================
# RUN
# =====================
main(
    scan="MF6_rotation",
    material="sapphire_MF6",
    date="20260121",
    SUBFOLD="MLOG",
    band=Fband,
    INTERPPEAKS=True,
    TRANS_SCALE=TransScale.DB,
    trimfunnydata=True,
    ANNOTE_RAW=True,
    ANNOTE_INTERP=False
)
