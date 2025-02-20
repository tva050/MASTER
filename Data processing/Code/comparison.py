import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc

# ------------- Paths ------------- #
cryo_oct_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\uit_cryosat2_L3_EASE2_nh25km_2023_10_v3.nc"
cryo_nov_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\uit_cryosat2_L3_EASE2_nh25km_2023_11_v3.nc"
cryo_dec_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\uit_cryosat2_L3_EASE2_nh25km_2023_12_v3.nc"

smos_oct_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\mean_sea_ice_thickness_oct.nc" 
smos_nov_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\mean_sea_ice_thickness_nov.nc"
smos_dec_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\mean_sea_ice_thickness_dec.nc"



def CryoSat2_read_data(path):
    # Load data from netCDF file, filter out invalid values and return the filtered data
    data = nc.Dataset(path)
    
    #print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['sea_ice_thickness'][:]
    si_thickness_uncertainty = data.variables['sea_ice_thickness_uncertainty'][:]
    
    mask = ~np.isnan(si_thickness) & ~np.isnan(si_thickness_uncertainty) & ~np.isnan(lat) & ~np.isnan(lon)
    
    filtered_lat = lat[mask]
    filtered_lon = lon[mask]
    filtered_si_thickness = si_thickness[mask]
    filtered_si_thickness_uncertainty = si_thickness_uncertainty[mask]
    return filtered_lat, filtered_lon, filtered_si_thickness, filtered_si_thickness_uncertainty


def SMOS_read_data(path):
    
    data = nc.Dataset(path)
    
    #print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['mean_sea_ice_thickness'][:]
    
    return lat, lon, si_thickness


def comp_all_months2():
    # Compares the SIT for three different months from CryoSat-2 and SMOS
    cryo_oct_lat, cryo_oct_lon, cryo_oct_si_thickness, cryo_oct_si_thickness_uncertainty = CryoSat2_read_data(cryo_oct_path)
    cryo_nov_lat, cryo_nov_lon, cryo_nov_si_thickness, cryo_nov_si_thickness_uncertainty = CryoSat2_read_data(cryo_nov_path)
    cryo_dec_lat, cryo_dec_lon, cryo_dec_si_thickness, cryo_dec_si_thickness_uncertainty = CryoSat2_read_data(cryo_dec_path)
    
    smos_oct_lat, smos_oct_lon, smos_oct_si_thickness = SMOS_read_data(smos_oct_path)
    smos_nov_lat, smos_nov_lon, smos_nov_si_thickness = SMOS_read_data(smos_nov_path)
    smos_dec_lat, smos_dec_lon, smos_dec_si_thickness = SMOS_read_data(smos_dec_path)
    
    # Create figure with 6 subplots with 3 for CryoSat-2 and 3 for SMOS
    fig, ax = plt.subplots(3, 2, subplot_kw={'projection': ccrs.NorthPolarStereo()})
    plt.subplots_adjust(wspace=0.0, hspace=0.2, right=0.85)
    
    # Titles for columns
    #ax[0, 0].set_title("SMOS", fontsize=14, fontweight="bold")
    #ax[0, 1].set_title("CryoSat-2", fontsize=14, fontweight="bold")
    
    # Month labels (left side)
    """ month_labels = ["October", "November", "December"]
    for i, label in enumerate(month_labels):
        ax[i, 0].text(-4.5e6, 0, label, fontsize=14, fontweight="bold", rotation=90, ha='center', va='center', transform=ccrs.PlateCarree())
 """
    # Define scatter plots for SMOS (left) and CryoSat-2 (right)
    data_pairs = [
        (smos_oct_lon, smos_oct_lat, smos_oct_si_thickness, ax[0, 0]),
        (cryo_oct_lon, cryo_oct_lat, cryo_oct_si_thickness, ax[0, 1]),
        (smos_nov_lon, smos_nov_lat, smos_nov_si_thickness, ax[1, 0]),
        (cryo_nov_lon, cryo_nov_lat, cryo_nov_si_thickness, ax[1, 1]),
        (smos_dec_lon, smos_dec_lat, smos_dec_si_thickness, ax[2, 0]),
        (cryo_dec_lon, cryo_dec_lat, cryo_dec_si_thickness, ax[2, 1])
    ]

    # Plot data
    for lon, lat, thickness, axis in data_pairs:
        axis.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
        axis.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
        axis.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        axis.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
        sc = axis.scatter(lon, lat, c=thickness, s=1, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree())

    
    # Add a single colorbar for the entire figure
    #cbar = fig.colorbar(sc, ax=ax[:, :], orientation='vertical', label='Sea Ice Thickness (m)', pad=0.05, aspect=50)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(sc, cax=cbar_ax, label='Sea Ice Thickness (m)') 
    plt.tight_layout()
    plt.show()
    
def comp_all_months():
    # Compares the SIT for three different months from CryoSat-2 and SMOS
    cryo_oct_lat, cryo_oct_lon, cryo_oct_si_thickness, cryo_oct_si_thickness_uncertainty = CryoSat2_read_data(cryo_oct_path)
    cryo_nov_lat, cryo_nov_lon, cryo_nov_si_thickness, cryo_nov_si_thickness_uncertainty = CryoSat2_read_data(cryo_nov_path)
    cryo_dec_lat, cryo_dec_lon, cryo_dec_si_thickness, cryo_dec_si_thickness_uncertainty = CryoSat2_read_data(cryo_dec_path)
    
    smos_oct_lat, smos_oct_lon, smos_oct_si_thickness = SMOS_read_data(smos_oct_path)
    smos_nov_lat, smos_nov_lon, smos_nov_si_thickness = SMOS_read_data(smos_nov_path)
    smos_dec_lat, smos_dec_lon, smos_dec_si_thickness = SMOS_read_data(smos_dec_path)
    
    fig = plt.figure(figsize=(8 , 10)) # figsize=(width, height)
    
    # Define axis positions: [left, bottom, width, height]
    ax_positions = [
        [0.05, 0.66, 0.38, 0.30], [0.32, 0.66, 0.38, 0.30],  # October
        [0.05, 0.34, 0.38, 0.30], [0.32, 0.34, 0.38, 0.30],  # November
        [0.05, 0.02, 0.38, 0.30], [0.32, 0.02, 0.38, 0.30]   # December
    ]

    # Create axes for each subplot
    axes = [fig.add_axes(pos, projection=ccrs.NorthPolarStereo()) for pos in ax_positions]

    # Set titles
    axes[0].set_title("SMOS", fontsize=14, pad=5)
    axes[1].set_title("CryoSat-2", fontsize=14, pad=5)

    # Configure each axis
    for ax in axes:
        ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
        ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)

    # Scatter plots
    scatter_plots = []

    # October
    sc_smos = axes[0].scatter(smos_oct_lon, smos_oct_lat, c=smos_oct_si_thickness, s=1, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree())
    sc_cryo = axes[1].scatter(cryo_oct_lon, cryo_oct_lat, c=cryo_oct_si_thickness, s=1, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree())

    # November
    axes[2].scatter(smos_nov_lon, smos_nov_lat, c=smos_nov_si_thickness, s=1, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree())
    axes[3].scatter(cryo_nov_lon, cryo_nov_lat, c=cryo_nov_si_thickness, s=1, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree())

    # December
    axes[4].scatter(smos_dec_lon, smos_dec_lat, c=smos_dec_si_thickness, s=1, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree())
    axes[5].scatter(cryo_dec_lon, cryo_dec_lat, c=cryo_dec_si_thickness, s=1, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree())

    scatter_plots.append((sc_smos, sc_cryo))

    # Add month labels on the left
    month_labels = ["October", "November", "December"]
    for i in range(3):
        fig.text(0.09, ax_positions[i * 2][1] + 0.15, month_labels[i], fontsize=14, rotation=90, ha='center', va='center') # (x, y, text, fontsize, rotation, ha, va)

    # Add colorbar on the right
    cbar_ax = fig.add_axes([0.65, 0.15, 0.02, 0.7])  # (left, bottom, width, height)
    fig.colorbar(sc_smos, cax=cbar_ax, label='Sea Ice Thickness (m)')

    plt.show()
    
comp_all_months()