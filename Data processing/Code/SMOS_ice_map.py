import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from collections import defaultdict
import time

folder_path_oct = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2024\oct"
folder_path_nov = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2024\nov"
folder_path_dec = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2024\dec"

""" def get_data(path):
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
            lat_data = np.array(dataset.variables['latitude'][:])
            lon_data = np.array(dataset.variables['longitude'][:])
            sea_ice_thickness_data = np.array(dataset.variables['sea_ice_thickness'][:])
            
    # Compute mean sea ice thickness for each lat-lon pair
    for (lat, lon), values in thickness_dict.items():
        latitudes.append(lat)
        longitudes.append(lon)
        mean_sea_ice_thickness.append(np.mean(values))

    return latitudes, longitudes, mean_sea_ice_thickness """

def get_data(path):
    thickness_dict = defaultdict(list)
    latitudes = []  
    longitudes = []  
    mean_sea_ice_thickness = [] 
    
    for filename in os.listdir(path):
        if filename.endswith(".nc"):
            file_path = os.path.join(path, filename)
            
            dataset = nc.Dataset(file_path) 
            print(dataset.variables.keys())
            lat_data = np.array(dataset.variables['latitude'][:])
            lon_data = np.array(dataset.variables['longitude'][:])
            sea_ice_thickness_data = np.array(dataset.variables['sea_ice_thickness'][:])
            
            mask = ~np.isnan(sea_ice_thickness_data) & ~np.isnan(lat_data) & ~np.isnan(lon_data)
            lat_data, lon_data, sea_ice_thickness_data = lat_data[mask], lon_data[mask], sea_ice_thickness_data[mask]
            
            mask = (sea_ice_thickness_data != -999.0) & (sea_ice_thickness_data != 0.0)
            lat_data, lon_data, sea_ice_thickness_data = lat_data[mask], lon_data[mask], sea_ice_thickness_data[mask]
            
            for lat, lon, thickness in zip(lat_data, lon_data, sea_ice_thickness_data):
                thickness_dict[(lat, lon)].append(thickness)
    
    for (lat, lon), values in thickness_dict.items():
        latitudes.append(lat)
        longitudes.append(lon)
        mean_sea_ice_thickness.append(np.mean(values))
    
    return np.array(latitudes), np.array(longitudes), np.array(mean_sea_ice_thickness)

def single_figure(lat, lon, mean_si_thickness):
    # Convert 1D lat/lon to 2D grid if necessary
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    # Reshape mean_si_thickness to match the grid
    mean_si_thickness = mean_si_thickness.reshape(lat.shape)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()})
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
   
    mesh = ax.pcolormesh(lon, lat, mean_si_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)
    
    cbar = plt.colorbar(mesh, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')
    
    plt.title('Sea Ice Thickness')
    plt.show()


def save_to_nc(latitudes, longitudes, mean_sea_ice_thickness, output_path):
    # creates an new nc file with the mean sea ice thickness data from the SMOS data for one month
    # with an column for latitude, longitude and mean sea ice thickness
    with nc.Dataset(output_path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("lat", len(latitudes))
        dataset.createDimension("lon", len(longitudes))
        dataset.createDimension("mean_sea_ice_thickness", len(mean_sea_ice_thickness))
        
        latitudes_var = dataset.createVariable("latitude", "f4", ("lat",))
        longitudes_var = dataset.createVariable("longitude", "f4", ("lon",))
        mean_sea_ice_thickness_var = dataset.createVariable("mean_sea_ice_thickness", "f4", ("mean_sea_ice_thickness",))
        
        latitudes_var[:] = latitudes
        longitudes_var[:] = longitudes
        mean_sea_ice_thickness_var[:] = mean_sea_ice_thickness
        
        dataset.description = "SMOS data for one month"
        dataset.history = "Created " + time.ctime(time.time())
        dataset.source = "SMOS data"
        
        latitudes_var.units = "degrees_north"
        longitudes_var.units = "degrees_east"
        mean_sea_ice_thickness_var.units = "m"
        
        print(f"Data saved to {output_path}")
        

latitudes, longitudes, mean_sea_ice_thickness = get_data(folder_path_oct)
single_figure(latitudes, longitudes, mean_sea_ice_thickness)

output_file = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\mean_sea_ice_thickness_oct.nc"            
#save_to_nc(latitudes, longitudes, mean_sea_ice_thickness, output_file)