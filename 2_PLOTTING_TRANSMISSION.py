# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 07:08:50 2026

@author: matth
"""
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
 
from pathlib      import Path
from enum         import Enum
from scipy.signal import welch
from matplotlib   import cm, colors
from matplotlib.lines import Line2D





#####
# Transmission scale

class TransScale(Enum):
    DB = "dB"
    LINEAR = "linear"

def dB_to_linear(db):
    return 10 ** (db / 20)


class Band:
    def __init__(self, name, lf, uf):
        self.name = name
        self.lf = lf
        self.uf = uf

Gband = Band("Gband", 140, 220)
Fband = Band("Fband", 90, 140)
Eband = Band("Eband", 55, 95)

class schema:
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

def load_transmission_txt(file_path):
    """
    Load a transmission TXT file.

    Returns
    -------
    dict with keys:
        - freq : np.ndarray
        - angles : dict
            angles[angle_deg][axis] = transmission_array
            axis in {'O', 'E', 'X'}
        - headers : list of column headers
    """
    df = pd.read_csv(file_path, sep=r"\s+")

    freq = df.iloc[:, 0].values
    headers = list(df.columns[1:])

    angle_data = {}

    for col in headers:
        # Expected format: trans_5deg_40.0deg_O
        parts = col.split("_")
        if len(parts) < 4:
            continue

        try:
            angle = float(parts[2].replace("deg", ""))
            axis = parts[-1]  # O, E, or X
        except ValueError:
            continue

        angle_data.setdefault(angle, {})[axis] = df[col].values

    return {
        "freq": freq,
        "angles": angle_data,
        "headers": headers
    }



### loads band data
def load_band_transmissions(
        base_dir: Path,
        bands,
        material,
        dates
    ):
    """
    Load transmission data for multiple bands and dates.

    Returns
    -------
    dict:
        data[band][date] = output of load_transmission_txt()
    """
    transmission_dir = base_dir / "Transmission"
    data = {}

    for band in bands:
        data[band] = {}

        for date in dates:
            filename = f"{band}_trans_{material}_{date}.txt"
            file_path = transmission_dir / filename

            if not file_path.exists():
                print(f"WARNING: Missing file {file_path}")
                continue

            data[band][date] = load_transmission_txt(file_path)

    return data

#takes df of all band data with SO bands showing
def plot_transmission_bands_SO(data, material):
    """
    Plot all bands and all angles on one plot.

    Parameters
    ----------
    data : dict
        Output from load_band_transmissions()
    """
    plt.figure(figsize=(10, 6))
    
    # SO:UK MF band limits with centre-frequency labels
    so_uk_bands = [
        (70, 120, "tab:purple", "SO:UK MF 90 GHz Band"),
        (120, 180, "tab:red",    "SO:UK MF 150 GHz Band")
    ]
    
    so_alpha = 0.2
    
    for lf, uf, color, label in so_uk_bands:
        plt.axvspan(
            lf,
            uf,
            color=color,
            alpha=so_alpha,
            label=label
        )




    for band, band_data in data.items():
        
    
        for date, content in band_data.items():
            freq = content["freq"]
    
            n_angles = len(content["angles"])
            alpha = max(0.05, 1 / np.sqrt(n_angles))
    
            for angle in sorted(content["angles"]):
                trans = content["angles"][angle]
                plt.plot(
                    freq,
                    trans,
                    color=color,
                    alpha=alpha,
                    lw=1
                )

        plt.plot([], [], color=color, label=band)


    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission")
    plt.title(str(material)+": Transmission vs Frequency (All Angles)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# plots the transmisison bands with the bands and angle shown 
def plot_transmission_bands_basic(data, material, TRANS_SCALE=TransScale.DB):
    fig, ax = plt.subplots(figsize=(10, 6))

    # -----------------------------
    # Define data bands (background)
    # -----------------------------
    data_bands = [
        (Eband.lf, Eband.uf, "tab:blue",  "Eband"),
        (Fband.lf, Fband.uf, "tab:orange","Fband"),
        (Gband.lf, Gband.uf, "tab:green", "Gband")
    ]

    for lf, uf, color, label in data_bands:
        plt.axvspan(
            lf,
            uf,
            color=color,
            alpha=0.1,
            label=label
        )

    # -----------------------------
    # Collect ALL angles (global)
    # -----------------------------
    all_angles = sorted({
        angle
        for band_data in data.values()
        for content in band_data.values()
        for angle in content["angles"]
    })

    if len(all_angles) == 0:
        print("No angle data found.")
        return

    norm = colors.Normalize(
        vmin=min(all_angles),
        vmax=max(all_angles)
    )
    cmap = cm.viridis  # single global colormap// different from the band colours for contrast.

    # -----------------------------
    # Plot data
    # -----------------------------
    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]

            n_angles = len(content["angles"])
            alpha = max(0.05, 1 / np.sqrt(n_angles))

            for angle in sorted(content["angles"]):
                trans = content["angles"][angle]
                
                # Transmission values
                trans = trans if TRANS_SCALE == TransScale.DB else dB_to_linear(trans)
                
                plt.plot(
                    freq,
                    trans,
                    color=cmap(norm(angle)),
                    alpha=alpha,
                    lw=1
                )

    # -----------------------------
    # Legend + colorbar
    # -----------------------------
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21| (linear)")
    plt.title(f"{material}: Transmission vs Frequency (All Angles)")
    plt.grid(True)

    # Clean legend (bands only)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Angle colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Angle (deg)")

    plt.tight_layout()
    plt.show()



def plot_3d_transmission(data, material,  TRANS_SCALE=TransScale.DB):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Track min/max for Z-axis limits and colormap consistency
    z_min, z_max = float('inf'), float('-inf')
    last_surf = None
    cmap = cm.viridis

    # 1. Loop through bands and contents
    for band_name, band_data in data.items():
        for content_name, content in band_data.items():
            
            # Sort the angles available in THIS specific content dictionary
            current_angles = sorted(content["angles"].keys())
            freq = content["freq"]
            
            if not current_angles or len(freq) == 0:
                continue

            # 2. Build the 2D Transmission Matrix (Rows=Angles, Cols=Frequencies)
            # This extracts the 1D array for each angle into a single 2D NumPy array
            try:
                trans_matrix = np.array([content["angles"][a] for a in current_angles])
            except Exception as e:
                print(f"Error processing {band_name}: {e}")
                continue

            # 3. Create the meshgrid for this specific band
            FREQ, ANG = np.meshgrid(freq, current_angles)
            
            # 4. Apply scale (dB or Linear)
            Z = trans_matrix if TRANS_SCALE == TransScale.DB else dB_to_linear(trans_matrix)
            
            # Update global Z limits for plot formatting
            z_min = min(z_min, Z.min())
            z_max = max(z_max, Z.max())

            # 5. Plot the surface
            # alpha=0.7 allows you to see overlapping bands if they exist
            # edgecolor='none' prevents the plot from looking too dark due to wireframes
            surf = ax.plot_surface(FREQ, ANG, Z, cmap=cmap, edgecolor='k', alpha=0.7)
            last_surf = surf

    # 6. Global formatting (Outside the loops)
    if last_surf is not None:
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Angle (deg)")
        ax.set_zlabel("Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21|")
        ax.set_title(f"{material}: 3D Transmission Profile (All Bands)")
        
        # Add a single colorbar for the whole figure
        fig.colorbar(last_surf, ax=ax, shrink=0.5, aspect=10, 
                     label="Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21|")
        
        # Ensure the Z-axis is scaled correctly for all data
        ax.set_zlim(z_min, z_max)
    else:
        print("No data found to plot.")

    plt.tight_layout()
    plt.show()

    

def plot_transmission_bands(data, material, TRANS_SCALE=TransScale.DB):
    """
    Main transmission plot with:
    - E/F/G band shading
    - O/E axes highlighted
    - X (off-axis) spectra faint + angle-colored
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{material}: Transmission vs Frequency (All Bands)")

    # Background bands
    data_bands = [
        (Eband.lf, Eband.uf, "tab:blue", "E-band"),
        (Fband.lf, Fband.uf, "tab:orange", "F-band"),
        (Gband.lf, Gband.uf, "tab:green", "G-band")
    ]

    for lf, uf, color, label in data_bands:
        ax.axvspan(lf, uf, color=color, alpha=0.1, label=label)

    # Collect all angles (for colorbar)
    all_angles = sorted({
        angle
        for band_data in data.values()
        for content in band_data.values()
        for angle in content["angles"]
    })

    if not all_angles:
        print("No angle data found.")
        return

    norm = colors.Normalize(vmin=min(all_angles), vmax=max(all_angles))
    cmap = cm.viridis

    peak_legend_handles = []
    seen_axes = set()

    # Plot data
    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]

            for angle in sorted(content["angles"]):
                for axis_type, trans in content["angles"][angle].items():

                    trans = trans if TRANS_SCALE == TransScale.DB else dB_to_linear(trans)

                    if axis_type in ['O', 'E']:
                        color = "blue" if axis_type == 'O' else "red"
                        label = "Ordinary (O)" if axis_type == 'O' else "Extraordinary (E)"

                        ax.plot(
                            freq, trans,
                            ls=":", lw=2.5, color=color,
                            alpha=1.0, zorder=3
                        )

                        if axis_type not in seen_axes:
                            peak_legend_handles.append(
                                Line2D(
                                    [0], [0], color=color, ls=":",
                                    lw=2.5, label=f"{label} Axis"
                                )
                            )
                            seen_axes.add(axis_type)

                    else:
                        ax.plot(
                            freq, trans,
                            color=cmap(norm(angle)),
                            lw=1, alpha=0.2, zorder=1
                        )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    band_dict = dict(zip(labels, handles))

    opacity_handle = Line2D(
        [0], [0], color="gray", lw=1, alpha=0.5,
        label="Off-axis spectra (X)"
    )

    final_handles = list(band_dict.values()) + [opacity_handle] + peak_legend_handles
    ax.legend(
        handles=final_handles,
        title="Transmission Spectra",
        fontsize=8,
        title_fontsize=9,
        loc="upper right"
    )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21| (linear)")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Rotation Angle (deg)")

    plt.tight_layout()
    plt.show()



def main(material, TRANS_SCALE=TransScale.DB):
    BASE_DIR = Path.cwd() / "Data"
 
    bands = ["Eband", "Fband", "Gband"] # bands we looked at
    dates = ["20260119", "20260120", "20260121", "20260122", "20260123"]  # dates we took data

    data = load_band_transmissions(
        base_dir=BASE_DIR,
        bands=bands,
        material=material,
        dates=dates
    )
    
    
    plot_transmission_bands(data, material, TRANS_SCALE)
    
    #plot_transmission_bands_SO(data, material)
    
    #plot_3d_transmission(data, material, TRANS_SCALE)
    
    #plot_transmission_bands_basic(data, material, TRANS_SCALE)

if __name__ == "__main__":
    
    possible_materials = ["sapphire_MF6", "sapphire_MF7","sapphire_MF1",]
    #[ "sapphire_MF1", "sapphire_MF2","sapphire_MF3", "sapphire_MF3_50mmshift","sapphire_MF3_100mmshift", "sapphire_MF4","sapphire_MF5","sapphire_MF6","sapphire_MF7","sapphire_MF8"]
    # "alumina_AF1", "alumina_AF1_edge",
    
    
    for i in possible_materials:
        main(material = i, TRANS_SCALE=TransScale.DB)
        

    
    
    

    
    
    