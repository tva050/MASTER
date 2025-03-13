import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from scipy.interpolate import interp1d

# ------------- Paths ------------- #
cryo_oct_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2022\uit_cryosat2_L3_EASE2_nh25km_2022_10_v3.nc"
cryo_nov_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2022\uit_cryosat2_L3_EASE2_nh25km_2022_11_v3.nc"
cryo_dec_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2022\uit_cryosat2_L3_EASE2_nh25km_2022_12_v3.nc"

smos_oct_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2022\oct_mean_thickness.nc" 
smos_nov_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2022\nov_mean_thickness.nc"
smos_dec_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2022\dec_mean_thickness.nc"



def CryoSat2_read_data(path):
    data = nc.Dataset(path)
    
    # print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['sea_ice_thickness'][:]
    # Check if lat and lon are 1D and need reshaping
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
        print('Reshaped lat and lon')
    # Mask invalid data
    mask = ~np.isnan(si_thickness)
    filtered_si_thickness = np.where(mask, si_thickness, np.nan)
    return lat, lon, filtered_si_thickness


def SMOS_read_data(path):
    data = nc.Dataset(path)
    
    # print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['sea_ice_thickness'][:]
    
    
    mask = ~np.isnan(si_thickness) & (si_thickness != -999.0) & (si_thickness != 0.0) 
    si_thickness = np.where(mask, si_thickness, np.nan)    
    return lat, lon, si_thickness



# extract data from the netCDF files using the functions above 
cryo_oct_lat, cryo_oct_lon, cryo_oct_si_thickness = CryoSat2_read_data(cryo_oct_path)
cryo_nov_lat, cryo_nov_lon, cryo_nov_si_thickness = CryoSat2_read_data(cryo_nov_path)
cryo_dec_lat, cryo_dec_lon, cryo_dec_si_thickness = CryoSat2_read_data(cryo_dec_path)

smos_oct_lat, smos_oct_lon, smos_oct_si_thickness = SMOS_read_data(smos_oct_path)
smos_nov_lat, smos_nov_lon, smos_nov_si_thickness = SMOS_read_data(smos_nov_path)
smos_dec_lat, smos_dec_lon, smos_dec_si_thickness = SMOS_read_data(smos_dec_path)



def comp_all_months():
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
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
        ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
        ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=4)

    # pcolormesh plots
    pcolormesh_plots = []

    # October
    sc_smos = axes[0].pcolormesh(smos_oct_lon, smos_oct_lat, smos_oct_si_thickness[0, :, :], cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)
    sc_cryo = axes[1].pcolormesh(cryo_oct_lon, cryo_oct_lat, cryo_oct_si_thickness, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)

    # November
    axes[2].pcolormesh(smos_nov_lon, smos_nov_lat, smos_nov_si_thickness[0, :, :], cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)
    axes[3].pcolormesh(cryo_nov_lon, cryo_nov_lat, cryo_nov_si_thickness, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)

    # December
    axes[4].pcolormesh(smos_dec_lon, smos_dec_lat, smos_dec_si_thickness[0, :, :], cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)
    axes[5].pcolormesh(cryo_dec_lon, cryo_dec_lat, cryo_dec_si_thickness, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)

    pcolormesh_plots.append((sc_smos, sc_cryo))

    # Add month labels on the left
    month_labels = ["October", "November", "December"]
    for i in range(3):
        fig.text(0.09, ax_positions[i * 2][1] + 0.15, month_labels[i], fontsize=14, rotation=90, ha='center', va='center') # (x, y, text, fontsize, rotation, ha, va)

    # Add colorbar on the right
    cbar_ax = fig.add_axes([0.65, 0.15, 0.02, 0.7])  # (left, bottom, width, height)
    fig.colorbar(sc_smos, cax=cbar_ax, label='Sea Ice Thickness (m)')

    plt.show()


def bar_smos():
    smos_oct_lat, smos_oct_lon, smos_oct_si_thickness = SMOS_read_data(smos_oct_path)
    smos_nov_lat, smos_nov_lon, smos_nov_si_thickness = SMOS_read_data(smos_nov_path)
    smos_dec_lat, smos_dec_lon, smos_dec_si_thickness = SMOS_read_data(smos_dec_path)
    
    # filter out all values smaller than 0
    mask = smos_oct_si_thickness > 0
    smos_oct_si_thickness = np.where(mask, smos_oct_si_thickness, np.nan)
    
    mean = np.nanmean(smos_oct_si_thickness)
    print(mean)
    std = np.nanstd(smos_oct_si_thickness)
    
    plt.bar('October', mean, yerr=std, capsize=5)
    plt.ylabel('Sea Ice Thickness (m)')
    plt.title('Comparison of Sea Ice Thickness')
    plt.show()
    


if __name__ == "__main__":
    #comp_all_months()
    bar_smos()