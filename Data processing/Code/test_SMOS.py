import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from collections import defaultdict


path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\Oct\SMOS_Icethickness_v3.3_north_20231015.nc"


def get_data(path):
    # Load data from netCDF file, filter out invalid values and return the filtered data
    data = nc.Dataset(path)
    
    print(data.variables.keys())
    
    lat          = np.array(data.variables['latitude'][:])
    lon          = np.array(data.variables['longitude'][:])
    si_thickness = np.array(data.variables['sea_ice_thickness'][:])
    
    lat = lat.flatten()
    lon = lon.flatten()
    si_thickness = si_thickness.flatten()
    
    mask = ~np.isnan(si_thickness) & ~np.isnan(lat) & ~np.isnan(lon)
    
    filtered_lat = lat[mask]
    filtered_lon = lon[mask]
    filtered_si_thickness = si_thickness[mask]
    
    # if value is -999.0 or 0.0, remove it same yields for the coordinates
    mask = (filtered_si_thickness != -999.0) & (filtered_si_thickness != 0.0)
    filtered_lat = filtered_lat[mask]
    filtered_lon = filtered_lon[mask]
    filtered_si_thickness = filtered_si_thickness[mask]
    
    # write the filtered data to an txt file 
    """ with open('filtered_data.txt', 'w') as f:
        for i in range(len(filtered_lat)):
            f.write(f'{filtered_lat[i]} {filtered_lon[i]} {filtered_si_thickness[i]}\n') """
    
    return filtered_lat, filtered_lon, filtered_si_thickness

def single_figure(filtered_lat, filtered_lon, filtered_si_thickness):
    fig = plt.figure(figsize=(10, 10))
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5)
    
    sc = ax.scatter(filtered_lon, filtered_lat, c=filtered_si_thickness, s=1, cmap='viridis', vmin=0, vmax=1 ,transform=ccrs.PlateCarree())
    
    plt.colorbar(sc, label='Sea Ice Thickness (m)')
    plt.show()
    





















path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\Oct\SMOS_Icethickness_v3.3_north_20231021.nc"

folder_path_oct = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\Oct"
folder_path_nov = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\Nov"
folder_path_dec = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\Dec"



def get_data(path):
    thickness_dict = defaultdict(list)
    latitudes = []  
    longitudes = []  
    mean_sea_ice_thickness = [] 

    for filename in os.listdir(path):
        if filename.endswith(".nc"):
            file_path = os.path.join(path, filename)

            dataset = nc.Dataset(file_path)

            lat_data = np.array(dataset.variables['latitude'][:]).flatten()
            lon_data = np.array(dataset.variables['longitude'][:]).flatten()
            sea_ice_thickness_data = np.array(dataset.variables['sea_ice_thickness'][:]).flatten()

            mask = ~np.isnan(sea_ice_thickness_data) & ~np.isnan(lat_data) & ~np.isnan(lon_data)
            lat_data, lon_data, sea_ice_thickness_data = lat_data[mask], lon_data[mask], sea_ice_thickness_data[mask]

            mask = (sea_ice_thickness_data != -999.0) & (sea_ice_thickness_data != 0.0)
            lat_data, lon_data, sea_ice_thickness_data = lat_data[mask], lon_data[mask], sea_ice_thickness_data[mask]

            for lat, lon, thickness in zip(lat_data, lon_data, sea_ice_thickness_data):
                thickness_dict[(lat, lon)].append(thickness)

    # Compute mean sea ice thickness for each lat-lon pair
    for (lat, lon), values in thickness_dict.items():
        latitudes.append(lat)
        longitudes.append(lon)
        mean_sea_ice_thickness.append(np.mean(values))

    return latitudes, longitudes, mean_sea_ice_thickness

def single_figure(latitudes, longitudes, mean_sea_ice_thickness):
    fig = plt.figure(figsize=(10, 10))
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5)
    
    sc = ax.scatter(longitudes, latitudes, c=mean_sea_ice_thickness, s=1, cmap='viridis', vmin=0, vmax=1 ,transform=ccrs.PlateCarree())
    
    plt.colorbar(sc, label='Sea Ice Thickness (m)')
    plt.show()
    
latitudes, longitudes, mean_sea_ice_thickness = get_data(folder_path_oct)
single_figure(latitudes, longitudes, mean_sea_ice_thickness)
            
    # calculate the mean sea ice thickness for each coordinate
    




