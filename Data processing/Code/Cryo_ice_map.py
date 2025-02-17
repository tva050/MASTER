import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap


oct_path = "C:\\Users\\trym7\\OneDrive - UiT Office 365\\skole\\MASTER\\Data processing\\Data\\uit_cryosat2_L3_EASE2_nh25km_2023_10_v3.nc"
nov_path = "C:\\Users\\trym7\\OneDrive - UiT Office 365\\skole\\MASTER\\Data processing\\Data\\uit_cryosat2_L3_EASE2_nh25km_2023_11_v3.nc"
dec_path = "C:\\Users\\trym7\\OneDrive - UiT Office 365\\skole\\MASTER\\Data processing\\Data\\uit_cryosat2_L3_EASE2_nh25km_2023_12_v3.nc"



def get_data_LARM(path):
    # Load data from netCDF file, filter out invalid values and return the filtered data
    data = nc.Dataset(path)
    
    print(data.variables.keys())
    
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


def single_figure(filtered_lat, filtered_lon, filtered_si_thickness, filtered_si_thickness_uncertainty):
    fig = plt.figure(figsize=(10, 10))
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5)
    
    sc = ax.scatter(filtered_lon, filtered_lat, c=filtered_si_thickness, s=1, cmap='viridis', vmin=0, vmax=4 ,transform=ccrs.PlateCarree())
    
    plt.colorbar(sc, label='Sea Ice Thickness (m)')
    plt.show()

def compare_months_LARM(oct_path, nov_path, dec_path):
    # Compares the SIT for three different months from the same satellite CryoSat-2
    
    oct_lat, oct_lon, oct_si_thickness, oct_si_thickness_uncertainty = get_data_LARM(oct_path)
    nov_lat, nov_lon, nov_si_thickness, nov_si_thickness_uncertainty = get_data_LARM(nov_path)
    dec_lat, dec_lon, dec_si_thickness, dec_si_thickness_uncertainty = get_data_LARM(dec_path)
    
    # Create figure with 3 subplots
    fig, ax = plt.subplots(1, 3, subplot_kw={'projection': ccrs.NorthPolarStereo()})
    
    # Set extent for North Polar Stereographic projection
    for a in ax:
        a.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
        a.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
        a.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        a.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5)
    
    # Add data to each subplot
    sc1 = ax[0].scatter(oct_lon, oct_lat, c=oct_si_thickness, s=1, cmap='viridis', vmin=0, vmax=4, transform=ccrs.PlateCarree())
    sc2 = ax[1].scatter(nov_lon, nov_lat, c=nov_si_thickness, s=1, cmap='viridis', vmin=0, vmax=4, transform=ccrs.PlateCarree())
    sc3 = ax[2].scatter(dec_lon, dec_lat, c=dec_si_thickness, s=1, cmap='viridis', vmin=0, vmax=4, transform=ccrs.PlateCarree())
    
    # Add an overall colorbar
    fig.colorbar(sc3, ax=ax, orientation='horizontal', label='Sea Ice Thickness (m)')
    
    plt.show()
    



if __name__ == "__main__":
    single_figure(*get_data_LARM(oct_path))
    #ompare_months_LARM(oct_path, nov_path, dec_path)