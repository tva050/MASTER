import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from collections import defaultdict

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
            