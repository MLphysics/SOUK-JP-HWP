
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
 
from pathlib      import Path
from enum         import Enum
from matplotlib   import cm, colors
from matplotlib.lines import Line2D

from scipy.optimize import curve_fit

from tmm import coh_tmm    


from concurrent.futures import ProcessPoolExecutor


# ============================================================
# dB ↔ LINEAR CONVERSIONS  (Amplitude scale)
# ============================================================
class TransScale(Enum):
    DB = "dB"
    LINEAR = "linear"

def dB_to_linear(db):
    """Amplitude transmission from dB."""
    return 10 ** (db / 20)


def linear_to_dB(lin):
    """Amplitude transmission to dB."""
    lin = np.maximum(lin, 1e-20)
    return 20 * np.log10(lin)


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
                if "X" in pol_dict:
                    trans_array = pol_dict["X"]
                elif "O" in pol_dict:
                    trans_array = pol_dict["O"]
                elif "E" in pol_dict:
                    trans_array = pol_dict["E"]
                else:
                    continue

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
    for band_data in data.values():
        for content in band_data.values():
            for pol_dict in content["angles"].values():
                if "O" in pol_dict and "E" in pol_dict:
                    if not np.allclose(pol_dict["O"], pol_dict["E"]):
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
    th_mullite= 0.212e-3
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

# ============================================================
# CACHED TMM CALL  (Speed fix)
# ============================================================

from functools import lru_cache

@lru_cache(maxsize=4096)
def _tmm_cached(lam, n_tuple, d_tuple, theta0):
    return coh_tmm('s', list(n_tuple), list(d_tuple), theta0, lam)['T']

# -------------------------------------------------
# TMM Calculation Wrappers (fixed TRANS_SCALE usage)
# -------------------------------------------------


# ============================================================
# SINGLE-LAYER TMM (SAPPHIRE)
# ============================================================

def tmm_transmission(freq_GHz, n_real, n_loss, layer_obj, TRANS_SCALE):

    c = 299792458.0
    THETA0 = 7 * np.pi / 180
    INF = np.inf

    lam_all = c / (freq_GHz * 1e9)

    n_list = [1, complex(n_real, n_loss), 1]
    d_list = [INF, layer_obj.th_recipe[1], INF]

    A = np.empty_like(freq_GHz, dtype=float)

    for i, lam in enumerate(lam_all):
        T = _tmm_cached(lam, tuple(n_list), tuple(d_list), THETA0)
        A[i] = np.sqrt(T)

    if TRANS_SCALE is TransScale.DB:
        return linear_to_dB(A)

    return A


# ============================================================
# FIXED-THICKNESS OPTICAL FIT TMM
# ============================================================

def tmm_transmission_fixed_thickness(freq_GHz, n_real, n_loss, th_fit, layer_obj, TRANS_SCALE):
    """
    Stack where alumina optical constants are fitted, thickness fixed.
    Works for both AluminaRecipe (5-layer) and SapphireRecipe (3-layer) with air layer.
    """

    c = 299792458.0
    THETA0 = 7*np.pi/180
    INF = np.inf
    lam_all = c / (freq_GHz * 1e9)

    # Determine stack type
    if isinstance(layer_obj, AluminaRecipe):
        # 5-layer: [1, mullite, alumina(fit), mullite, 1]
        n_list = [
            1,
            layer_obj.n_recipe[1],
            complex(n_real, n_loss),
            layer_obj.n_recipe[3],
            1
        ]
        d_list = [
            INF,
            th_fit["th_m"],
            th_fit["th_a"],
            th_fit["th_m"],
            INF
        ]
    elif isinstance(layer_obj, SapphireRecipe):
        # 3-layer: [1, sapphire(fit), 1]
        n_list = [1, complex(n_real, n_loss), 1]
        d_list = [INF, layer_obj.th_recipe[1], INF]
    else:
        raise TypeError(f"Unsupported recipe type: {type(layer_obj)}")

    A = np.empty_like(freq_GHz, dtype=float)
    for i, lam in enumerate(lam_all):
        T = _tmm_cached(lam, tuple(n_list), tuple(d_list), THETA0)
        A[i] = np.sqrt(T)

    if TRANS_SCALE is TransScale.DB:
        return linear_to_dB(A)
    return A




# ============================================================
# TWO-LAYER TMM (MULLITE + ALUMINA)
# ============================================================

def tmm_transmission_thickness_2layer(freq_GHz, th_m, th_a, layer_obj, TRANS_SCALE):

    c = 299792458.0
    THETA0 = 7 * np.pi / 180
    INF = np.inf

    lam_all = c / (freq_GHz * 1e9)

    n_list = layer_obj.n_recipe
    d_list = [INF, th_m, th_a, th_m, INF]

    A = np.empty_like(freq_GHz, dtype=float)

    for i, lam in enumerate(lam_all):
        T = _tmm_cached(lam, tuple(n_list), tuple(d_list), THETA0)
        A[i] = np.sqrt(T)

    if TRANS_SCALE is TransScale.DB:
        return linear_to_dB(A)

    return A



# -------------------------------------------------
# Parallel fitting per angle (fixed TRANS_SCALE usage)
# -------------------------------------------------
def fit_thickness_global(data, material, TRANS_SCALE=TransScale.DB):
    """
    Fit thicknesses of layers globally for non-birefringent materials.
    Automatically skips infinite-thickness layers (e.g., Sapphire).
    """
    # Determine axis key
    if 'X' in material_dict[material]:
        axis_type = 'X'       # Alumina case
    else:
        axis_type = next(iter(material_dict[material].keys()))  # Sapphire case ('O' or 'E')

    layer_obj = material_dict[material][axis_type]

    # Only consider layers with finite thickness
    finite_th_indices = [i for i, th in enumerate(layer_obj.th_recipe) if np.isfinite(th)]
    if len(finite_th_indices) < 2:
        print(f"No finite thickness layers to fit for {material}, skipping thickness fit.")
        return {"th_m": 0, "th_a": 0, "th_m_err": 0, "th_a_err": 0}

    # Assume layer 1 = mullite, layer 2 = alumina (AluminaRecipe)
    th_m0 = layer_obj.th_recipe[finite_th_indices[0]]
    th_a0 = layer_obj.th_recipe[finite_th_indices[1]]

    p0 = [th_m0, th_a0]
    bounds = ([0.5*th_m0, 0.5*th_a0], [1.5*th_m0, 1.5*th_a0])

    # Collect all data for axis_type
    freq_all, trans_all = [], []

    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]
            for t_dict in content["angles"].values():
                if axis_type not in t_dict:
                    continue
                t_dat = t_dict[axis_type]
                if TRANS_SCALE == TransScale.LINEAR:
                    t_dat = dB_to_linear(t_dat)
                freq_all.append(freq)
                trans_all.append(t_dat)

    if not freq_all or not trans_all:
        print(f"No data found for {material}, skipping thickness fit.")
        return {"th_m": 0, "th_a": 0, "th_m_err": 0, "th_a_err": 0}

    freq_all = np.concatenate(freq_all)
    trans_all = np.concatenate(trans_all)

    # Fit only if finite thickness layers exist
    popt, pcov = curve_fit(
        lambda f, th_m, th_a:
            tmm_transmission_thickness_2layer(f, th_m, th_a, layer_obj, TRANS_SCALE),
        freq_all, trans_all,
        p0=p0,
        bounds=bounds
    )

    th_m, th_a = popt
    th_m_err, th_a_err = np.sqrt(np.diag(pcov))

    return {"th_m": th_m, "th_a": th_a, "th_m_err": th_m_err, "th_a_err": th_a_err}


def _get_initial_guess(layer_obj):
    """
    Return initial n and loss guess from the first complex index in n_recipe.
    Works for SapphireRecipe and AluminaRecipe.
    """
    # Pick first complex refractive index in the stack
    n_candidates = [x for x in layer_obj.n_recipe if isinstance(x, complex)]
    if not n_candidates:
        raise ValueError("No complex refractive index found in n_recipe")
    n0 = n_candidates[0].real
    loss0 = max(n_candidates[0].imag, 0.0)
    return n0, loss0


def _fit_angle(angle, data, material, th_fit, TRANS_SCALE):
    """
    Fit n and loss for a given angle.
    Automatically selects the correct axis:
    - 'X' if present (Alumina)
    - first available axis (Sapphire: 'O' or 'E')
    """
    # Determine axis key
    if 'X' in material_dict[material]:
        axis_type = 'X'
    else:
        axis_type = next(iter(material_dict[material].keys()))  # 'O' or 'E'

    layer_obj = material_dict[material][axis_type]

    freq_all, trans_all = [], []

    for band_data in data.values():
        for content in band_data.values():
            if angle not in content["angles"]:
                continue
            if axis_type not in content["angles"][angle]:
                continue
            freq_all.append(content["freq"])
            t_dat = content["angles"][angle][axis_type]
            if TRANS_SCALE == TransScale.LINEAR:
                t_dat = dB_to_linear(t_dat)
            trans_all.append(t_dat)

    if not freq_all:
        return angle, None

    freq_all = np.concatenate(freq_all)
    trans_all = np.concatenate(trans_all)

    # Initial guess from first complex index
    n0, loss0 = _get_initial_guess(layer_obj)

    # Safe bounds
    bounds_lower = [2.5, 0.0]   # n, loss
    bounds_upper = [4.0, 0.1]

    # Adjust n0 if outside bounds
    n0 = np.clip(n0, bounds_lower[0]+1e-3, bounds_upper[0]-1e-3)
    loss0 = np.clip(loss0, bounds_lower[1]+1e-6, bounds_upper[1]-1e-6)

    popt, pcov = curve_fit(
        lambda f, n, loss:
            tmm_transmission_fixed_thickness(f, n, loss, th_fit, layer_obj, TRANS_SCALE),
        freq_all, trans_all,
        p0=[n0, loss0],
        bounds=(bounds_lower, bounds_upper)
    )

    n_fit, loss_fit = popt
    n_err, loss_err = np.sqrt(np.diag(pcov))

    return angle, {
        "n": n_fit,
        "n_err": n_err,
        "loss": max(loss_fit, 0.0),
        "loss_err": max(loss_err, 0.0)
    }


def _fit_angle_light(angle, angle_data, material, th_fit, TRANS_SCALE):
    """
    Lightweight fit version for serial Alumina processing.
    Automatically selects the correct axis:
    - 'X' if present (Alumina)
    - first available axis (Sapphire: 'O' or 'E')
    """
    if 'X' in material_dict[material]:
        axis_type = 'X'
    else:
        axis_type = next(iter(material_dict[material].keys()))

    layer_obj = material_dict[material][axis_type]

    freq = angle_data["freq"]
    trans = angle_data["trans"]

    # Initial guess
    n0, loss0 = _get_initial_guess(layer_obj)

    # Safe bounds
    bounds_lower = [2.5, 0.0]
    bounds_upper = [4.0, 0.1]
    n0 = np.clip(n0, bounds_lower[0]+1e-3, bounds_upper[0]-1e-3)
    loss0 = np.clip(loss0, bounds_lower[1]+1e-6, bounds_upper[1]-1e-6)

    popt, pcov = curve_fit(
        lambda f, n, loss: tmm_transmission_thickness_2layer(
            f, th_fit["th_m"], th_fit["th_a"], layer_obj, TRANS_SCALE
        ),
        freq, trans,
        p0=[n0, loss0],
        bounds=(bounds_lower, bounds_upper)
    )

    n_fit, loss_fit = popt
    n_err, loss_err = np.sqrt(np.diag(pcov))

    return angle, {
        "n": n_fit,
        "n_err": n_err,
        "loss": max(loss_fit, 0.0),
        "loss_err": max(loss_err, 0.0)
    }





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

def fit_n_loss_vs_angle_serial(data, material, th_fit, TRANS_SCALE=TransScale.DB):
    """Fit n, loss per angle in serial (faster for small datasets)."""
    all_angles = sorted({angle
                         for band_data in data.values()
                         for content in band_data.values()
                         for angle in content["angles"]
                         if 'X' in content["angles"][angle]})
    results = {}
    for angle in all_angles:
        res = _fit_angle(angle, data, material, th_fit, TRANS_SCALE)
        angle_val, fit_dict = res
        if fit_dict is not None:
            results[angle_val] = fit_dict
    return results




################################################################################


# --- Utility functions ---
def plot_with_error(ax, x, y, yerr, color, label, marker='o', linestyle='-'):
    ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, linestyle=linestyle,
                capsize=3, label=label, alpha=0.7)

# --- sapphire ---
# -----------------------------
# Sapphire plotting (fixed)
# -----------------------------
def plot_sapphire_wafer(data, material, fit_results, TRANS_SCALE):
    fig = plt.figure(figsize=(18, 8))

    ax_3d = fig.add_subplot(121, projection='3d')
    ax_fit = fig.add_subplot(122)

    # -----------------------------
    # 3D DATA
    # -----------------------------
    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]

            for angle, pol_dict in content["angles"].items():
                if "O" in pol_dict:
                    trans = pol_dict["O"]
                elif "E" in pol_dict:
                    trans = pol_dict["E"]
                else:
                    continue

                if TRANS_SCALE is TransScale.LINEAR:
                    trans = dB_to_linear(trans)

                ax_3d.plot(
                    freq,
                    [angle]*len(freq),
                    trans,
                    alpha=0.6
                )

    ax_3d.set_xlabel("Freq (GHz)")
    ax_3d.set_ylabel("Angle (deg)")
    ax_3d.set_zlabel("Transmission")

    # -----------------------------
    # FITS OVER ALL BANDS/DATES
    # -----------------------------
    for axis in ["O", "E"]:
        fit = fit_results["optical"][axis]
        color = "tab:red" if axis=="E" else "tab:blue"

        for band_name, band_data in data.items():
            for date_name, content in band_data.items():
                freq = content["freq"]
                trans_fit = tmm_transmission(
                    freq,
                    fit["n"],
                    fit["loss"],
                    material_dict[material][axis],
                    TRANS_SCALE
                )
                if TRANS_SCALE is TransScale.LINEAR:
                    trans_fit = dB_to_linear(trans_fit)

                ax_fit.plot(
                    freq,
                    trans_fit,
                    lw=2,
                    color=color,
                    label=f"{band_name}_{date_name}_{axis}: n={fit['n']:.3f}, loss={fit['loss']:.2e}"
                )

    ax_fit.set_xlabel("Freq (GHz)")
    ax_fit.set_ylabel("Transmission")
    ax_fit.legend(fontsize=8)
    ax_fit.grid(True)

    plt.tight_layout()
    plt.show()


# -----------------------------
# Alumina plotting (fixed)
# -----------------------------
def plot_mullite_alumina_full(data, material, fit_results, TRANS_SCALE):
    """Plot all bands and angles for Alumina with mullite coating."""
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
    plt.errorbar(
        angles,
        [th_a_fit*1e3]*len(angles),
        yerr=[th_a_err*1e3]*len(angles),
        fmt='o',
        color='tab:red',
        label='Alumina th'
    )
    plt.hlines(
        th_m_fit*1e3,
        angles.min(),
        angles.max(),
        colors='tab:blue',
        label='Mullite th'
    )
    plt.xlabel("Angle (deg)")
    plt.ylabel("Thickness (mm)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Overlay of measured vs fits for ALL bands/dates ---
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    for band_name, band_data in data.items():
        for date_name, content in band_data.items():
            freq = content["freq"]
            for t_dict in content["angles"].values():
                if 'X' not in t_dict:
                    continue
                t_dat = t_dict['X']
                if TRANS_SCALE == TransScale.LINEAR:
                    t_dat = dB_to_linear(t_dat)
                plt.scatter(freq, t_dat, color='gray', s=15, alpha=0.5)

    # Overlay fits for all bands
    for band_name, band_data in data.items():
        for date_name, content in band_data.items():
            freq = content["freq"]
            T_fit = tmm_transmission_thickness_2layer(freq, th_m_fit, th_a_fit, layer_obj, TRANS_SCALE)
            plt.plot(freq, T_fit, color='tab:green', lw=2, label=f"{band_name}_{date_name} fit")

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission (dB)" if TRANS_SCALE==TransScale.DB else "|S21|")
    plt.title(f"{material}: All Fits Overlay")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.subplot(1,2,2)

    # Average fit
    freq_concat = np.concatenate([content["freq"] for band_data in data.values() for content in band_data.values()])
    T_avg_fit = tmm_transmission_thickness_2layer(freq_concat, th_m_fit, th_a_fit, layer_obj, TRANS_SCALE)
    plt.plot(freq_concat, T_avg_fit, color='black', lw=2, label=f"Avg Fit: n={np.mean(n_vals):.3f}, loss={np.mean(loss_vals):.5f}, th_m={th_m_fit*1e3:.3f} mm, th_a={th_a_fit*1e3:.3f} mm")

    # Scatter measured data
    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]
            for t_dict in content["angles"].values():
                if 'X' not in t_dict:
                    continue
                t_dat = t_dict['X']
                if TRANS_SCALE == TransScale.LINEAR:
                    t_dat = dB_to_linear(t_dat)
                plt.scatter(freq, t_dat, color='gray', s=15, alpha=0.5)

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Transmission (dB)" if TRANS_SCALE==TransScale.DB else "|S21|")
    plt.title(f"{material}: Average Fit Overlay")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_mullite_overlay(freq_all, th_m_fit, th_a_fit, layer_obj, TRANS_SCALE):

    T_fit = tmm_transmission_thickness_2layer(
        freq_all,
        th_m_fit,
        th_a_fit,
        layer_obj,
        TRANS_SCALE
    )

    plt.plot(freq_all, T_fit, color='tab:green', lw=2)

    plt.xlabel("Freq (GHz)")
    plt.ylabel("Transmission")
    plt.grid(True)
    plt.show()



# --- Smart plot dispatcher ---
def smart_plot(data, material, fit_results, TRANS_SCALE=TransScale.DB):
    """
    Unified plotting routine for sapphire and alumina, keeping errors in legends.
    """
    mat_axes = material_dict.get(material, {})
    if not mat_axes:
        print(f"Material {material} not found. Skipping plot.")
        return

    is_sapphire = "O" in mat_axes and "E" in mat_axes
    is_alumina = "X" in mat_axes

    fig, ax_meas = plt.subplots(1, 2, figsize=(16, 8))
    ax_fit = ax_meas[1]
    ax_meas = ax_meas[0]

    # -----------------------------
    # Plot measured data
    # -----------------------------
    for band_name, band_data in data.items():
        for date_name, content in band_data.items():
            freq = content["freq"]
            for angle, pol_dict in content["angles"].items():
                if is_sapphire:
                    for axis in ["O", "E"]:
                        if axis not in pol_dict:
                            continue
                        t_dat = pol_dict[axis]
                        if TRANS_SCALE == TransScale.LINEAR:
                            t_dat = dB_to_linear(t_dat)
                        ax_meas.scatter(freq, t_dat, color='gray', s=10, alpha=0.3)
                elif is_alumina:
                    if "X" not in pol_dict:
                        continue
                    t_dat = pol_dict["X"]
                    if TRANS_SCALE == TransScale.LINEAR:
                        t_dat = dB_to_linear(t_dat)
                    ax_meas.scatter(freq, t_dat, color='gray', s=10, alpha=0.3)

    ax_meas.set_xlabel("Frequency (GHz)")
    ax_meas.set_ylabel("Transmission (dB)" if TRANS_SCALE==TransScale.DB else "|S21|")
    ax_meas.set_title(f"{material}: Measured Transmission (all bands/angles)")
    ax_meas.grid(True)

    # -----------------------------
    # Overlay fits with error info
    # -----------------------------
    if is_sapphire:
        for axis in ["O", "E"]:
            fit = fit_results["optical"][axis]
            color = "tab:red" if axis=="E" else "tab:blue"
            for band_name, band_data in data.items():
                for date_name, content in band_data.items():
                    freq = content["freq"]
                    trans_fit = tmm_transmission(
                        freq,
                        fit["n"],
                        fit["loss"],
                        material_dict[material][axis],
                        TRANS_SCALE
                    )
                    if TRANS_SCALE == TransScale.LINEAR:
                        trans_fit = dB_to_linear(trans_fit)

                    ax_fit.plot(
                        freq,
                        trans_fit,
                        lw=2,
                        color=color,
                        label=f"{band_name}_{date_name}_{axis}: "
                              f"n={fit['n']:.3f}±{fit['n_err']:.3f}, "
                              f"loss={fit['loss']:.2e}±{fit['loss_err']:.2e}"
                    )
    elif is_alumina:
        layer_obj = material_dict[material]["X"]
        th_m = fit_results["thickness"]["mullite"]["value"]
        th_m_err = fit_results["thickness"]["mullite"]["error"]
        th_a = fit_results["thickness"]["alumina"]["value"]
        th_a_err = fit_results["thickness"]["alumina"]["error"]
        optical_results = fit_results["optical"]["X"]

        for band_name, band_data in data.items():
            for date_name, content in band_data.items():
                freq = content["freq"]
                T_fit = tmm_transmission_thickness_2layer(freq, th_m, th_a, layer_obj, TRANS_SCALE)
                ax_fit.plot(
                    freq,
                    T_fit,
                    lw=2,
                    color='tab:green',
                    label=f"{band_name}_{date_name} fit: "
                          f"th_m={th_m*1e3:.3f}±{th_m_err*1e3:.3f} mm, "
                          f"th_a={th_a*1e3:.3f}±{th_a_err*1e3:.3f} mm"
                )

        # Add angle-dependent n/loss info in legend (optional)
        # Take mean ± std across angles for a summary
        n_vals = [v["n"] for v in optical_results.values()]
        n_errs = [v["n_err"] for v in optical_results.values()]
        loss_vals = [v["loss"] for v in optical_results.values()]
        loss_errs = [v["loss_err"] for v in optical_results.values()]
        ax_fit.plot([], [], color='white',
                    label=f"Alumina n: {np.mean(n_vals):.3f}±{np.mean(n_errs):.3f}, "
                          f"loss: {np.mean(loss_vals):.3f}±{np.mean(loss_errs):.3f}")

    ax_fit.set_xlabel("Frequency (GHz)")
    ax_fit.set_ylabel("Transmission (dB)" if TRANS_SCALE==TransScale.DB else "|S21|")
    ax_fit.set_title(f"{material}: TMM Fit Overlay")
    ax_fit.grid(True)
    ax_fit.legend(fontsize=8)
    plt.tight_layout()
    plt.show()



     
def extract_angle_data(data, axis, TRANS_SCALE):
    angle_data = {}

    for band_data in data.values():
        for content in band_data.values():
            freq = content["freq"]
            for angle, pol_dict in content["angles"].items():
                if axis not in pol_dict:
                    continue

                t = pol_dict[axis]
                if TRANS_SCALE == TransScale.LINEAR:
                    t = dB_to_linear(t)

                angle_data.setdefault(angle, {"freq": [], "trans": []})
                angle_data[angle]["freq"].append(freq)
                angle_data[angle]["trans"].append(t)

    for angle in angle_data:
        angle_data[angle]["freq"] = np.concatenate(angle_data[angle]["freq"])
        angle_data[angle]["trans"] = np.concatenate(angle_data[angle]["trans"])

    return angle_data



# -------------------------------------------------
# Unified dataset processing for all materials
# -------------------------------------------------
def process_dataset(data, material, TRANS_SCALE=TransScale.DB, parallel=False):
    """
    Process dataset and return fitting results for a material.
    Automatically detects material type (sapphire vs alumina).
    """
    fit_results = {"thickness": {}, "optical": {}}

    # -----------------------------
    # Determine material type
    # -----------------------------
    mat_axes = material_dict.get(material, {})
    if not mat_axes:
        raise ValueError(f"Material {material} not found in material_dict.")

    is_sapphire = "O" in mat_axes and "E" in mat_axes
    is_alumina = "X" in mat_axes

    # -----------------------------
    # Sapphire workflow (birefringent)
    # -----------------------------
    if is_sapphire:
        for axis in ["O", "E"]:
            layer_obj = material_dict[material][axis]

            freq_all, trans_all = [], []

            # Collect mean transmission across all angles
            for band_data in data.values():
                for content in band_data.values():
                    freq = content["freq"]
                    t_list = []

                    for t_dict in content["angles"].values():
                        if axis not in t_dict:
                            continue
                        t_dat = t_dict[axis]
                        if TRANS_SCALE == TransScale.LINEAR:
                            t_dat = dB_to_linear(t_dat)
                        t_list.append(t_dat)

                    if not t_list:
                        continue

                    t_mean = np.mean(t_list, axis=0)
                    freq_all.append(freq)
                    trans_all.append(t_mean)

            if not freq_all:
                continue

            freq_all = np.concatenate(freq_all)
            trans_all = np.concatenate(trans_all)

            # Initial guesses
            n0 = layer_obj.n_recipe[1].real
            loss0 = max(layer_obj.n_recipe[1].imag, 0.0)

            popt, pcov = curve_fit(
                lambda f, n, loss: tmm_transmission(f, n, loss, layer_obj, TRANS_SCALE),
                freq_all, trans_all,
                p0=[n0, loss0],
                bounds=([2.5, 0.0], [4.0, 0.1])
            )

            n_fit, loss_fit = popt
            n_err, loss_err = np.sqrt(np.diag(pcov))

            fit_results["optical"][axis] = {
                "n": n_fit,
                "n_err": n_err,
                "loss": loss_fit,
                "loss_err": loss_err,
                "thickness": layer_obj.th_recipe[1],
                "thickness_err": 0.0
            }

        # Thickness for sapphire (single layer)
        fit_results["thickness"]["layer"] = {
            "value": layer_obj.th_recipe[1],
            "error": 0.0
        }

    # -----------------------------
    # Alumina workflow (non-birefringent)
    # -----------------------------
    elif is_alumina:
        # Step 1: Fit thickness globally
        th_fit = fit_thickness_global(data, material, TRANS_SCALE)
        fit_results["thickness"] = {
            "mullite": {"value": th_fit["th_m"], "error": th_fit["th_m_err"]},
            "alumina": {"value": th_fit["th_a"], "error": th_fit["th_a_err"]}
        }

        # Step 2: Precompute angle data once
        angle_data_dict = extract_angle_data(data, "X", TRANS_SCALE)

        # Step 3: Fit n, loss per angle
        optical_results = {}
        for angle, angle_data in angle_data_dict.items():
            angle_val, fit_dict = _fit_angle_light(angle, angle_data, material, th_fit, TRANS_SCALE)
            optical_results[angle_val] = fit_dict

        fit_results["optical"]["X"] = optical_results

    else:
        raise ValueError(f"Material {material} not recognized (missing axes).")

    return fit_results







def plot_thickness_vs_angle(angles, th_m_fit, th_a_fit, th_m_err, th_a_err):

    plt.figure(figsize=(8,5))

    plt.hlines(
        th_m_fit*1e3,
        min(angles),
        max(angles),
        colors='tab:blue',
        label="Mullite"
    )

    plt.errorbar(
        angles,
        [th_a_fit*1e3]*len(angles),
        yerr=th_a_err*1e3,
        fmt='o',
        color='tab:red',
        label="Alumina"
    )

    plt.xlabel("Angle (deg)")
    plt.ylabel("Thickness (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------------------------------
# Wrapper for full fitting + plotting
# -------------------------------------------------
def tmm_fit_and_plot(material, data, TRANS_SCALE=TransScale.DB, parallel=False):
    """
    Unified TMM fitting for sapphire or alumina.
    Returns fit_results dictionary.
    """
    fit_results = process_dataset(data, material, TRANS_SCALE, parallel)
    return fit_results



def main(material, TRANS_SCALE=TransScale.DB, parallel=False):
    BASE_DIR = Path.cwd() / "Data"

    bands = ["Eband", "Fband", "Gband"]
    dates = ["20260119", "20260120", "20260121", "20260122", "20260123"]

    # -----------------------------
    # Load transmission data
    # -----------------------------
    data = load_band_transmissions(
        base_dir=BASE_DIR,
        bands=bands,
        material=material,
        dates=dates
    )

    if not data:
        print(f"No data found for {material}, skipping.")
        return

    # -----------------------------
    # Determine material type
    # -----------------------------
    mat_axes = material_dict.get(material, {})
    if not mat_axes:
        print(f"Material {material} not found in material_dict. Skipping.")
        return

    # Sapphire: look for O/E axes
    is_sapphire = "O" in mat_axes and "E" in mat_axes

    # Alumina: look for X axis
    is_alumina = "X" in mat_axes

    # -----------------------------
    # Fit data
    # -----------------------------
    if is_sapphire:
        # Sapphire: O/E axes
        fit_results = tmm_fit_and_plot(material, data, TRANS_SCALE)

    elif is_alumina:
        # Alumina: X axis
        th_fit = fit_thickness_global(data, material, TRANS_SCALE)
        optical_results = (
            fit_n_loss_vs_angle_parallel(data, material, th_fit, TRANS_SCALE)
            if parallel else
            fit_n_loss_vs_angle_serial(data, material, th_fit, TRANS_SCALE)
        )

        fit_results = {
            "thickness": {
                "mullite": {"value": th_fit["th_m"], "error": th_fit["th_m_err"]},
                "alumina": {"value": th_fit["th_a"], "error": th_fit["th_a_err"]}
            },
            "optical": {"X": optical_results}
        }

    else:
        print(f"Material {material} not recognized (no proper axes found). Skipping.")
        return

    # -----------------------------
    # Plot
    # -----------------------------
    smart_plot(data, material, fit_results, TRANS_SCALE=TRANS_SCALE)




if __name__ == "__main__":
    
    #possible_materials = [ "sapphire_MF1",]
    ####all materials 
    #possible_materials = [ "alumina_AF1", "alumina_AF1_edge", "sapphire_MF1", "sapphire_MF2","sapphire_MF3", "sapphire_MF3_50mmshift","sapphire_MF3_100mmshift", "sapphire_MF4","sapphire_MF5","sapphire_MF6","sapphire_MF7","sapphire_MF8"]
    #### sample selection of materials
    possible_materials = [ "alumina_AF1", "alumina_AF1_edge", "sapphire_MF8"]
    #### just alumina
    #possible_materials = ["alumina_AF1", "alumina_AF1_edge",]
    #### just the troublesome MF3s
    #possible_materials = [ "sapphire_MF3", "sapphire_MF3_50mmshift","sapphire_MF3_100mmshift",]
    
  
    
    for i in possible_materials:
        main(material = i, TRANS_SCALE=TransScale.DB, parallel=False)
    