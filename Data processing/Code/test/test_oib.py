import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def load_oib_lat_lon(file_paths):
    """Loads lat/lon coordinates from OIB text files and stacks them."""
    lat_list, lon_list = [], []

    for file in file_paths:
        data = np.loadtxt(file, usecols=(0, 1))  # Assuming lat is col 0, lon is col 1
        lat_list.append(data[:, 0])
        lon_list.append(data[:, 1])

    # Concatenate all data points into single arrays
    target_lat = np.concatenate(lat_list)
    target_lon = np.concatenate(lon_list)

    return target_lat, target_lon

def interpolate_data(source_lat, source_lon, source_si, target_lat, target_lon):
    """Interpolates source_si to target_lat/lon grid using SciPy's griddata."""
    source_points = np.array([source_lat.flatten(), source_lon.flatten()]).T
    source_values = source_si.flatten()

    target_points = np.array([target_lat.flatten(), target_lon.flatten()]).T

    source_interp = griddata(source_points, source_values, target_points, method='linear')

    return source_interp

def multiple_box_plot_oib(oib_sit, smos_lat, smos_lon, smos_si, cryo_si, oib_paths):
    # Load OIB target lat/lon from files
    target_lat, target_lon = load_oib_lat_lon(oib_paths)

    # Interpolate SMOS data to OIB lat/lon grid
    print(f"Interpolating SMOS data to match OIB lat/lon grid...")
    smos_interp = interpolate_data(smos_lat, smos_lon, smos_si, target_lat, target_lon)

    # Ensure all datasets have the same shape
    if smos_interp.shape != oib_sit.shape or cryo_si.shape != oib_sit.shape:
        raise ValueError(f"Interpolated SMOS, CryoSat-2, and OIB data must have the same shape! "
                         f"oib: {oib_sit.shape}, smos: {smos_interp.shape}, cryo: {cryo_si.shape}")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']

    # Flatten arrays
    oib_flat = oib_sit.ravel()
    smos_flat = smos_interp.ravel()
    cryo_flat = cryo_si.ravel()

    # Mask NaN values
    valid_mask = ~np.isnan(oib_flat) & ~np.isnan(smos_flat) & ~np.isnan(cryo_flat)
    oib_flat, smos_flat, cryo_flat = oib_flat[valid_mask], smos_flat[valid_mask], cryo_flat[valid_mask]

    # Bin the data
    binned_oib_smos_data, binned_oib_cryo_data = [], []

    for i in range(len(bins) - 1):
        mask = (oib_flat >= bins[i]) & (oib_flat < bins[i + 1])
        binned_oib_smos_data.append(smos_flat[mask])
        binned_oib_cryo_data.append(cryo_flat[mask])

    # Create subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    # Boxplots
    axes[0].boxplot(binned_oib_smos_data, labels=bin_labels, medianprops=dict(color='black'))
    axes[0].set_title("OIB vs SMOS")
    axes[0].set_xlabel("OIB SIT [m]")
    axes[0].set_ylabel("SMOS SIT [m]")

    axes[1].boxplot(binned_oib_cryo_data, labels=bin_labels, medianprops=dict(color='black'))
    axes[1].set_title("OIB vs CryoSat-2")
    axes[1].set_xlabel("OIB SIT [m]")
    axes[1].set_ylabel("CryoSat-2 SIT [m]")

    # Scatter plots
    for j in range(len(bins) - 1):
        x_positions_smos = np.random.normal(j + 1, 0.05, size=len(binned_oib_smos_data[j]))
        axes[0].scatter(x_positions_smos, binned_oib_smos_data[j], alpha=0.4, color='salmon', s=10)

        x_positions_cryo = np.random.normal(j + 1, 0.05, size=len(binned_oib_cryo_data[j]))
        axes[1].scatter(x_positions_cryo, binned_oib_cryo_data[j], alpha=0.4, color='teal', s=10)

    for ax in axes:
        ax.yaxis.set_tick_params(labelleft=True)

    plt.tight_layout()
    plt.show()
