
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
 
from pathlib      import Path
from enum         import Enum
from scipy.signal import welch
from matplotlib   import cm, colors
from matplotlib.lines import Line2D

from scipy.optimize import curve_fit

from tmm import coh_tmm    


from concurrent.futures import ProcessPoolExecutor


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
        - angles : dict0
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

#plots full band range with so MF highlighted 
def plot_transmission_bands_SO(data, material, TRANS_SCALE=TransScale.DB):
    """
    Plot all bands and all angles on one plot.
    O = blue, E = red, X = green
    Scale: dB or linear |S21|
    """
    plt.figure(figsize=(10, 6))

    # SO:UK MF band limits
    so_uk_bands = [
        (70, 120, "tab:purple", "SO:UK MF 90 GHz Band"),
        (120, 180, "tab:red",    "SO:UK MF 150 GHz Band")
    ]

    for lf, uf, color, label in so_uk_bands:
        plt.axvspan(lf, uf, color=color, alpha=0.2, label=label)

    # Polarisation colours
    pol_colors = {
        "O": "tab:blue",   # ordinary
        "E": "tab:red",    # extraordinary
        "X": "tab:green"   # off-axis
    }

    alpha = 0.5

    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]

            for angle in sorted(content["angles"]):
                pol_dict = content["angles"][angle]

                for pol, color in pol_colors.items():
                    trans_db = pol_dict.get(pol)
                    if trans_db is None:
                        continue

                    # Apply scale
                    trans = (
                        trans_db
                        if TRANS_SCALE == TransScale.DB
                        else dB_to_linear(trans_db)
                    )

                    plt.plot(
                        freq,
                        trans,
                        color=color,
                        alpha=alpha,
                        lw=1
                    )

    # Legend (one entry per polarization)
    legend_handles = [
        Line2D([0], [0], color="tab:blue",  lw=3, label="Ordinary)", alpha = 0.5),
        Line2D([0], [0], color="tab:red",   lw=3, label="Extraordinary)", alpha = 0.5),
        Line2D([0], [0], color="tab:green", lw=1, label="off-axis)", alpha = 0.1)
    ]

    plt.xlabel("Frequency (GHz)")
    plt.ylabel(
        "Transmission (dB)"
        if TRANS_SCALE == TransScale.DB
        else "|S21| (linear)"
    )
    plt.title(f"{material}: Transmission vs Frequency (All Angles)")
    plt.legend(handles=legend_handles)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# plots full band as a nice summary with different heads and angle showing
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

    pol_colors = {
        "O": "tab:blue",   # ordinary
        "E": "tab:red",    # extraordinary
        "X": "tab:green"   # off-axis
    }

    for lf, uf, color, label in data_bands:
        plt.axvspan(lf, uf, color=color, alpha=0.1, label=label)

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

    norm = colors.Normalize(vmin=min(all_angles), vmax=max(all_angles))
    cmap = cm.viridis  # angle → colour

    # -----------------------------
    # Plot data
    # -----------------------------
    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]
            n_angles = len(content["angles"])
            
            for angle in sorted(content["angles"]):
                pol_dict = content["angles"][angle]

                for pol, color in pol_colors.items():
                    trans_db = pol_dict.get(pol)
                    if trans_db is None:
                        continue

                    # Apply scale
                    trans = (
                        trans_db
                        if TRANS_SCALE == TransScale.DB
                        else dB_to_linear(trans_db)
                    )

                    # Make O and E lines more prominent
                    if pol in ["O", "E"]:
                        lw = 2
                        alpha = 1.0
                    else:  # X lines
                        lw = 1
                        alpha = max(0.05, 1 / np.sqrt(n_angles))

                    plt.plot(
                        freq,
                        trans,
                        color=color,
                        alpha=alpha,
                        lw=lw
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

# Fun 3d plot
def plot_3d_transmission(data, material, TRANS_SCALE=TransScale.DB):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    z_min, z_max = float('inf'), float('-inf')
    last_surf = None
    cmap = cm.viridis

    for band_name, band_data in data.items():
        for content_name, content in band_data.items():

            current_angles = sorted(content["angles"].keys())
            freq = content["freq"]

            if not current_angles or len(freq) == 0:
                continue

            # -----------------------------
            
            trans_matrix = []
            for a in current_angles:
                pol_dict = content["angles"][a]
                if not pol_dict:
                    continue
                # get the first numeric array in the dict
                first_key = next(iter(pol_dict))
                trans_array = pol_dict[first_key]
                trans_matrix.append(trans_array)

            if len(trans_matrix) == 0:
                continue

            trans_matrix = np.array(trans_matrix)
            current_angles = current_angles[:len(trans_matrix)]

            # -----------------------------
            # Create meshgrid
            # -----------------------------
            FREQ, ANG = np.meshgrid(freq, current_angles)
            Z = trans_matrix if TRANS_SCALE == TransScale.DB else dB_to_linear(trans_matrix)

            z_min = min(z_min, Z.min())
            z_max = max(z_max, Z.max())

            surf = ax.plot_surface(FREQ, ANG, Z, cmap=cmap, edgecolor='k', alpha=0.7)
            last_surf = surf

    if last_surf is not None:
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Angle (deg)")
        ax.set_zlabel("Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21|")
        ax.set_title(f"{material}: 3D Transmission Profile (All Bands)")
        fig.colorbar(last_surf, ax=ax, shrink=0.5, aspect=10,
                     label="Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21|")
        ax.set_zlim(z_min, z_max)
    else:
        print("No data found to plot.")

    plt.tight_layout()
    plt.show()


def is_birefringent(data):
    """Check if dataset has O/E axes (birefringent)"""
    for band_data in data.values():
        for content in band_data.values():
            for angle_data in content["angles"].values():
                if 'O' in angle_data or 'E' in angle_data:
                    return True
    return False

# -------------------------------------------------
# Material Recipe Wrappers
# -------------------------------------------------
class SapphireRecipe:
    def __init__(self, n, loss, thickness):
        self.n_recipe = [1, complex(n, loss), 1]
        self.th_recipe = [np.inf, thickness, np.inf]

class AluminaRecipe:
    def __init__(self, n_alumina, loss_alumina, th_alumina, n_mullite, th_mullite):
        self.n_recipe = [1, n_mullite, complex(n_alumina, loss_alumina), n_mullite, 1]
        self.th_recipe = [np.inf, th_mullite, th_alumina, th_mullite, np.inf]
        
        
# -------------------------------------------------
# Sapphire / Alumina Recipes
# -------------------------------------------------
# Sapphire (birefringent)
MF1_O = SapphireRecipe(n=3.05, loss=0.0005, thickness=3.7434e-3)
MF1_E = SapphireRecipe(n=3.39, loss=0.0005, thickness=3.7434e-3)

MF2_O = SapphireRecipe(n=3.05, loss=0.0005, thickness=3.7399e-3)
MF2_E = SapphireRecipe(n=3.39, loss=0.0005, thickness=3.7399e-3)

MF3_O = SapphireRecipe(n=3.05, loss=0.0005, thickness=3.7462e-3)
MF3_E = SapphireRecipe(n=3.39, loss=0.0005, thickness=3.7462e-3)

MF4_O = SapphireRecipe(n=3.05, loss=0.0005, thickness=3.7869e-3)
MF4_E = SapphireRecipe(n=3.39, loss=0.0005, thickness=3.7869e-3)

MF5_O = SapphireRecipe(n=3.05, loss=0.0005, thickness=3.7475e-3)
MF5_E = SapphireRecipe(n=3.39, loss=0.0005, thickness=3.7475e-3)

MF6_O = SapphireRecipe(n=3.05, loss=0.0005, thickness=3.774894035e-3)
MF6_E = SapphireRecipe(n=3.39, loss=0.0005, thickness=3.774894035e-3)

MF7_O = SapphireRecipe(n=3.05, loss=0.0005, thickness=3.8038e-3)
MF7_E = SapphireRecipe(n=3.39, loss=0.0005, thickness=3.8038e-3)

MF8_O = SapphireRecipe(n=3.05, loss=0.0005, thickness=3.7650e-3)
MF8_E = SapphireRecipe(n=3.39, loss=0.0005, thickness=3.7650e-3)

# Alumina (single alumina layer coated with mullite)
AF1 = AluminaRecipe(
    n_alumina=3.14,
    loss_alumina=0,
    th_alumina=3e-3,
    n_mullite=2.52,
    th_mullite=0.212e-3
)

# -------------------------------------------------
# Material dictionary
# -------------------------------------------------
material_dict = {
    # Sapphire MF series
    "sapphire_MF1": {"O": MF1_O, "E": MF1_E},
    "sapphire_MF2": {"O": MF2_O, "E": MF2_E},
    "sapphire_MF3": {"O": MF3_O, "E": MF3_E},
    "sapphire_MF3_50mmshift": {"O": MF3_O, "E": MF3_E},
    "sapphire_MF3_100mmshift": {"O": MF3_O, "E": MF3_E},
    "sapphire_MF4": {"O": MF4_O, "E": MF4_E},
    "sapphire_MF5": {"O": MF5_O, "E": MF5_E},
    "sapphire_MF6": {"O": MF6_O, "E": MF6_E},
    "sapphire_MF7": {"O": MF7_O, "E": MF7_E},
    "sapphire_MF8": {"O": MF8_O, "E": MF8_E},

    # Alumina AF series
    "alumina_AF1": {"O": AF1, "E": AF1, "X": AF1},
    "alumina_AF1_edge": {"O": AF1, "E": AF1, "X": AF1}
}



# -------------------------------------------------
# TMM Calculation Wrappers (fixed TRANS_SCALE usage)
# -------------------------------------------------
def tmm_transmission(freq_GHz, n_real, n_loss, layer_obj, TRANS_SCALE):
    c = 299792458.0
    INF = np.inf
    THETA0 = 7*np.pi/180
    n_fit = [1, complex(n_real, n_loss), 1]
    d_fit = [INF, layer_obj.th_recipe[1], INF]
    T = np.empty_like(freq_GHz, dtype=float)
    for i, f in enumerate(freq_GHz):
        lam = c / (f * 1e9)
        out = coh_tmm('s', n_fit, d_fit, THETA0, lam)
        T[i] = out['T']
    if TRANS_SCALE == TransScale.DB:
        T = 10*np.log10(T)
    return T


def tmm_transmission_fixed_thickness(freq_GHz, n_real, n_loss, layer_obj, th_m, th_a, TRANS_SCALE):
    c = 299792458.0
    INF = np.inf
    THETA0 = 7*np.pi/180
    n_list = [1, layer_obj.n_recipe[1], complex(n_real, n_loss), layer_obj.n_recipe[3], 1]
    d_list = [INF, th_m, th_a, th_m, INF]
    T = np.empty_like(freq_GHz, dtype=float)
    for i, f in enumerate(freq_GHz):
        lam = c / (f * 1e9)
        out = coh_tmm('s', n_list, d_list, THETA0, lam)
        T[i] = out['T']
    if TRANS_SCALE == TransScale.DB:
        T = 10*np.log10(T)
    return T


def tmm_transmission_thickness_2layer(freq_GHz, th_m, th_a, layer_obj, TRANS_SCALE):
    c = 299792458.0
    INF = np.inf
    THETA0 = 7*np.pi/180
    n_list = layer_obj.n_recipe
    d_list = [INF, th_m, th_a, th_m, INF]
    T = np.empty_like(freq_GHz, dtype=float)
    for i, f in enumerate(freq_GHz):
        lam = c / (f * 1e9)
        out = coh_tmm('s', n_list, d_list, THETA0, lam)
        T[i] = out['T']
    if TRANS_SCALE == TransScale.DB:
        T = 10*np.log10(T)
    return T


# -------------------------------------------------
# Parallel fitting per angle (fixed TRANS_SCALE usage)
# -------------------------------------------------
def fit_thickness_global(data, material, TRANS_SCALE=TransScale.DB):
    axis_type = 'X'
    layer_obj = material_dict[material][axis_type]
    th_m0, th_a0 = layer_obj.th_recipe[1], layer_obj.th_recipe[2]
    p0 = [th_m0, th_a0]
    bounds = ([0.5*th_m0,0.5*th_a0],[1.5*th_m0,1.5*th_a0])
    freq_all, trans_all = [], []
    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]
            for t_dict in content["angles"].values():
                if axis_type not in t_dict:
                    continue
                t_dat = t_dict[axis_type]
                if TRANS_SCALE==TransScale.LINEAR:
                    t_dat=dB_to_linear(t_dat)
                freq_all.append(freq)
                trans_all.append(t_dat)
                
                
    if not freq_all or not trans_all:
        print(f"No data found for {material}, skipping thickness fit.")
        return {"th_m": 0, "th_a": 0, "th_m_err": 0, "th_a_err": 0}
    
    
    freq_all = np.concatenate(freq_all)
    trans_all = np.concatenate(trans_all)
    popt, pcov = curve_fit(lambda f, th_m, th_a:
                           tmm_transmission_thickness_2layer(f, th_m, th_a, layer_obj, TRANS_SCALE),
                           freq_all, trans_all, p0=p0, bounds=bounds)
    th_m, th_a = popt
    th_m_err, th_a_err = np.sqrt(np.diag(pcov))
    return {"th_m": th_m, "th_a": th_a, "th_m_err": th_m_err, "th_a_err": th_a_err}

def _fit_angle(angle, data, material, th_fit, TRANS_SCALE):
    layer_obj = material_dict[material]['X']  # uniform layer
    freq_all, trans_all = [], []

    # Gather all transmission data for this angle
    for band_data in data.values():
        for content in band_data.values():
            if angle not in content["angles"] or 'X' not in content["angles"][angle]:
                continue
            freq_all.append(content["freq"])
            t_dat = content["angles"][angle]['X']
            if TRANS_SCALE == TransScale.LINEAR:
                t_dat = dB_to_linear(t_dat)
            trans_all.append(t_dat)

    if not freq_all:
        return angle, None

    freq_all = np.concatenate(freq_all)
    trans_all = np.concatenate(trans_all)

    # Starting values from material dictionary
    n_start = layer_obj.n_recipe[2].real
    loss_start = max(layer_obj.n_recipe[2].imag, 0.)  # ensure non-negative

    # Bounds: n and loss positive
    n_bounds = (2.5, 4.0)
    loss_bounds = (0.0, 0.1)

    popt, pcov = curve_fit(
        lambda f, n, loss: tmm_transmission_fixed_thickness(
            f, n, loss, layer_obj, th_fit["th_m"], th_fit["th_a"], TRANS_SCALE
        ),
        freq_all, trans_all,
        p0=[n_start, loss_start],
        bounds=([n_bounds[0], loss_bounds[0]],
                [n_bounds[1], loss_bounds[1]])
    )

    n_fit, loss_fit = popt
    n_err, loss_err = np.sqrt(np.diag(pcov))

    # Ensure fitted loss >= 0
    loss_fit = max(loss_fit, 0.0)
    loss_err = max(loss_err, 0.0)

    return angle, {"n": n_fit, "n_err": n_err, "loss": loss_fit, "loss_err": loss_err}


def fit_n_loss_vs_angle_parallel(data, material, th_fit, TRANS_SCALE=TransScale.DB):
    all_angles = sorted({angle
                         for band_data in data.values()
                         for content in band_data.values()
                         for angle in content["angles"]
                         if 'X' in content["angles"][angle]})
    results = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_fit_angle, angle, data, material, th_fit, TRANS_SCALE) for angle in all_angles]
        for f in futures:
            angle, res = f.result()
            if res is not None:  # filter out failed fits
                results[angle] = res
    return results



################################################################################


# --- Utility functions ---
def plot_with_error(ax, x, y, yerr, color, label, marker='o', linestyle='-'):
    ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, linestyle=linestyle,
                capsize=3, label=label, alpha=0.7)

# --- Sapphire plotting ---
def plot_sapphire_wafer(data, material, fit_results, TRANS_SCALE):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax_data, ax_fit = axes

    blue_shades = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
    red_shades  = plt.cm.Reds(np.linspace(0.4, 0.9, 8))

    # -----------------------------
    # Plot DATA (black)
    # -----------------------------
    for band, band_data in data.items():
        for date, content in band_data.items():
            freq = content["freq"]

            for angle, pol_dict in content["angles"].items():
                for axis, t in pol_dict.items():
                    if axis not in ("O", "E"):
                        continue

                    y = t if TRANS_SCALE == TransScale.DB else dB_to_linear(t)

                    ax_data.plot(
                        freq,
                        y,
                        color="k",
                        alpha=0.7,
                        marker="o" if np.isclose(angle % 5, 0) else ".",
                        lw=0
                    )

    ax_data.set_title("Measured Data")
    ax_data.set_xlabel("Frequency (GHz)")
    ax_data.set_ylabel("Transmission (dB)" if TRANS_SCALE == TransScale.DB else "|S21|")
    ax_data.grid(True)

    # -----------------------------
    # Plot FITS (sorted by frequency)
    # -----------------------------
    fit_lines = []

    for axis, axis_fits in fit_results["optical"].items():
        if axis not in ("O", "E"):
            continue
    
        for angle, fit in axis_fits.items():
            # Get frequency array safely from data
            band_key = next(iter(data))
            subband_key = next(iter(data[band_key]))
            freq = data[band_key][subband_key]["freq"]
            
            fit_lines.append((np.mean(freq), axis, angle, fit))
                

    # Sort by ascending frequency
    fit_lines.sort(key=lambda x: x[0])

    for idx, (_, axis, angle, fit) in enumerate(fit_lines):
        freq = fit["freq"]
        trans = fit["model"]

        y = trans if TRANS_SCALE == TransScale.DB else dB_to_linear(trans)

        color = (
            red_shades[idx % len(red_shades)]
            if axis == "E"
            else blue_shades[idx % len(blue_shades)]
        )

        label = (
            f"{band} {axis} | "
            f"n={fit['n']:.3f}±{fit['n_err']:.3f}, "
            f"loss={fit['loss']:.2e}±{fit['loss_err']:.2e}, "
            f"t={fit['thickness']*1e3:.2f}±{fit['thickness_err']*1e3:.2f} mm"
        )

        ax_fit.plot(freq, y, color=color, lw=2, label=label)

    ax_fit.set_title("TMM Fits")
    ax_fit.set_xlabel("Frequency (GHz)")
    ax_fit.grid(True)
    ax_fit.legend(fontsize=8)

    plt.suptitle(f"{material}: Sapphire Transmission (O & E)")
    plt.tight_layout()
    plt.show()


# --- Mullite-coated Alumina plotting ---
def plot_mullite_alumina_full(data, material, fit_results, TRANS_SCALE):
    """Plot n, loss, thickness vs angle and overlay fits for Mullite-coated Alumina"""
    
    layer_obj = material_dict[material]['X']
    optical_results = fit_results['optical']['X']
    angles = np.array(sorted(optical_results.keys()))
    n_vals = np.array([optical_results[a]["n"] for a in angles])
    loss_vals = np.array([optical_results[a]["loss"] for a in angles])
    th_m_fit = fit_results["thickness"]["mullite"]["value"]
    th_m_err = fit_results["thickness"]["mullite"]["error"]
    th_a_fit = fit_results["thickness"]["alumina"]["value"]
    th_a_err = fit_results["thickness"]["alumina"]["error"]

    # --- Refractive index vs angle ---
    plt.figure(figsize=(8,5))
    plt.errorbar(angles, n_vals, yerr=[optical_results[a]["n_err"] for a in angles],
                 fmt='o', color='tab:red', label='Alumina n')
    plt.hlines(layer_obj.n_recipe[1].real, angles.min(), angles.max(), colors='tab:blue', label='Mullite n')
    plt.xlabel("Angle (deg)")
    plt.ylabel("Refractive Index n")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Loss vs angle ---
    plt.figure(figsize=(8,5))
    plt.errorbar(angles, loss_vals, yerr=[optical_results[a]["loss_err"] for a in angles],
                 fmt='o', color='tab:red', label='Alumina loss')
    plt.hlines(0, angles.min(), angles.max(), colors='tab:blue', label='Mullite loss')
    plt.xlabel("Angle (deg)")
    plt.ylabel("Loss Im[n]")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Thickness vs angle ---
    plt.figure(figsize=(8,5))
    plt.errorbar(angles, [th_a_fit]*len(angles), yerr=[th_a_err]*len(angles), fmt='o', color='tab:red', label='Alumina th')
    plt.hlines(th_m_fit, angles.min(), angles.max(), colors='tab:blue', label='Mullite th')
    plt.xlabel("Angle (deg)")
    plt.ylabel("Thickness (mm)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Overlay of measured vs fits ---
    freq_all, trans_all = [], []
    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]
            for t_dict in content["angles"].values():
                if 'X' not in t_dict:
                    continue
                freq_all.append(freq)
                trans_all.append(t_dict['X'])
    freq_all = np.concatenate(freq_all)
    trans_all = np.concatenate(trans_all)

    # All fits per angle
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    for angle in angles:
        T_fit = tmm_transmission_thickness_2layer(freq_all, th_m_fit, th_a_fit, layer_obj, TRANS_SCALE)
        plt.plot(freq_all, T_fit, color='tab:green', alpha=0.5)
    plt.scatter(freq_all, trans_all, color='gray', s=15, alpha=0.5)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission (dB)" if TRANS_SCALE==TransScale.DB else "|S21|")
    plt.title(f"{material}: All Fits Overlay")
    plt.grid(True)

    # Average fit
    plt.subplot(1,2,2)
    T_avg_fit = tmm_transmission_thickness_2layer(freq_all, th_m_fit, th_a_fit, layer_obj, TRANS_SCALE)
    plt.plot(freq_all, T_avg_fit, color='black', lw=2, label=f"Avg Fit: n={np.mean(n_vals):.3f}, loss={np.mean(loss_vals):.5f}, th_m={th_m_fit*1e3:.3f} mm, th_a={th_a_fit*1e3:.3f} mm")
    plt.scatter(freq_all, trans_all, color='gray', s=15, alpha=0.5)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission (dB)" if TRANS_SCALE==TransScale.DB else "|S21|")
    plt.title(f"{material}: Average Fit Overlay")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- Smart plot dispatcher ---
def smart_plot(data, material, fit_results, TRANS_SCALE):
    if is_birefringent(data):
        plot_sapphire_wafer(data, material, fit_results, TRANS_SCALE)
    else:
        plot_mullite_alumina_full(data, material, fit_results, TRANS_SCALE)

        
# -------------------------------------------------
# Main Processing Function (fixed plotting)
# -------------------------------------------------
def process_dataset(data, material, TRANS_SCALE=TransScale.DB):
    """
    Fit material properties (thickness, n, loss) and automatically generate summary plots.
    - Mullite-coated alumina: average over X-axis for n/loss; plot fitted th_m and th_a.
    - Sapphire: fit O/E axes individually; include n/loss errors; plot thickness, n, loss.
    """
    if not data or all(not band_data for band_data in data.values()):
        print(f"No transmission data for {material}, skipping processing.")
        return None

    
    birefringent = is_birefringent(data)
    fit_results = {"optical": {}, "thickness": {}}

    if birefringent:
        # -----------------------------
        # Fit O/E axes for sapphire
        # -----------------------------
        for axis in ['O','E']:
            layer_obj = material_dict[material][axis]
            freq_all, trans_all = [], []

            for band_data in data.values():
                for content in band_data.values():
                    t_list = [t_dict[axis] for t_dict in content["angles"].values() if axis in t_dict]
                    if not t_list:
                        continue
                    t_mean = np.mean(t_list, axis=0)
                    freq_all.append(content["freq"])
                    trans_all.append(t_mean)
                    
            if not freq_all or not trans_all:
                print(f"No data for {axis} axis in {material}, skipping fit.")
                continue

            freq_all = np.concatenate(freq_all)
            trans_all = np.concatenate(trans_all)

            n_start = layer_obj.n_recipe[1].real
            loss_start = max(layer_obj.n_recipe[1].imag, 0.)

            popt, pcov = curve_fit(
                lambda f, n, loss: tmm_transmission(f, n, loss, layer_obj, TRANS_SCALE),
                freq_all, trans_all,
                p0=[n_start, loss_start],
                bounds=([2.5, 0.0], [4.0, 0.1])
            )

            n_fit, loss_fit = popt
            n_err, loss_err = np.sqrt(np.diag(pcov))
            loss_fit = max(loss_fit, 0.0)

            # Store both value and error
            fit_results["optical"][axis] = {"n": n_fit, "n_err": n_err, "loss": loss_fit, "loss_err": loss_err}

        # Thickness is fixed for sapphire
        fit_results["thickness"]["layer"] = {"value": layer_obj.th_recipe[1], "error": 0}

    else:
        # -----------------------------
        # Fit Mullite-coated Alumina
        # -----------------------------
        th_fit = fit_thickness_global(data, material, TRANS_SCALE)
        fit_results["thickness"]["mullite"] = {"value": th_fit["th_m"], "error": th_fit["th_m_err"]}
        fit_results["thickness"]["alumina"] = {"value": th_fit["th_a"], "error": th_fit["th_a_err"]}

        optical_results = fit_n_loss_vs_angle_parallel(data, material, th_fit, TRANS_SCALE)
        fit_results["optical"]["X"] = optical_results

    # -----------------------------
    # Automatic plotting of results
    # -----------------------------
    smart_plot(data, material, fit_results, TRANS_SCALE=TRANS_SCALE)

    return fit_results


# -------------------------------------------------
# Wrapper for full fitting + plotting
# -------------------------------------------------
def tmm_fit_and_plot(material, data, TRANS_SCALE=TransScale.DB):
    """
    Calls process_dataset and returns fit results.
    TRANS_SCALE must be a TransScale enum.
    """
    fit_results = process_dataset(data, material, TRANS_SCALE)
    return fit_results



def main(material, TRANS_SCALE=TransScale.DB):
    BASE_DIR = Path.cwd() / "Data"
 
    bands = ["Eband", "Fband", "Gband"]
    dates = ["20260119", "20260120", "20260121", "20260122", "20260123"]

    data = load_band_transmissions(
        base_dir=BASE_DIR,
        bands=bands,
        material=material,
        dates=dates
    )
    
    
    ### SUMMARY PLOTS 
    ### ==================  nice plot of all band data with SO regions highlighted 
    #plot_transmission_bands_SO(data, material)
    ### ==================  NICE SUMMARY PLOT!! plot of all data together with the bands highlighted 
    #plot_transmission_bands_basic(data, material, TRANS_SCALE=TransScale.DB)
    ### ==================  Fun 3d plot
    #plot_3d_transmission(data, material,  TRANS_SCALE=TransScale.DB)
    
    ### FITTING!!
    tmm_fit_and_plot(material, data, TRANS_SCALE=TransScale.DB)



if __name__ == "__main__":
    
    possible_materials = [ "sapphire_MF1",]
    ####all materials 
    #possible_materials = [ "alumina_AF1", "alumina_AF1_edge", "sapphire_MF1", "sapphire_MF2","sapphire_MF3", "sapphire_MF3_50mmshift","sapphire_MF3_100mmshift", "sapphire_MF4","sapphire_MF5","sapphire_MF6","sapphire_MF7","sapphire_MF8"]
    #### sample selection of materials
    #possible_materials = [ "alumina_AF1", "alumina_AF1_edge", "sapphire_MF8"]
    #### just alumina
    #possible_materials = ["alumina_AF1", "alumina_AF1_edge",]
    #### just the troublesome MF3s
    #possible_materials = [ "sapphire_MF3", "sapphire_MF3_50mmshift","sapphire_MF3_100mmshift",]
    
  
    
    for i in possible_materials:
        main(material = i, TRANS_SCALE=TransScale.DB)
    