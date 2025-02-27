""" 
This script includes an comprehensive comparison of the different products used to estimate sea ice thickness.
The comparison is done by plotting the data on a map and comparing the results visually.

The script includes the products:
- CryoSat-2 L2 Trajectory Data Baseline D
- CS2 ice thickness data from AWI
- CS2 ice thickness data from CPOM
- SMOS ice thickness data
- CryoSat-2 L3 

"""
import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from scipy.interpolate import griddata


cpom_oct = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_10.map.nc"
cpom_nov = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_11.map.nc"
cpom_dec = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_12.map.nc"




def get_cpom(path):
    data = nc.Dataset(path)
    
    print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['thickness'][:]
    grid_spacing = data.variables["grid_spacing"][:]
    
    
    print(f"Lat shape: {lat.shape}")
    print(f"Lon shape: {lon.shape}")
    print(f"Thickness shape: {si_thickness.shape}")
    print(f"Grid spacing: {grid_spacing}")
    
    """ if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
        print('Reshaped lat and lon') """
    
    return lat, lon, si_thickness
    
    
""" def plot(lat, lon, si_thickness):
    size = len(lat)
    side_length = int(np.sqrt(size))
    while size % side_length != 0:
        side_length -= 1
    grid_shape = (side_length, size // side_length)
    # Reshape the data
    lat_grid = lat.reshape(grid_shape)
    lon_grid = lon.reshape(grid_shape)
    thickness_grid = si_thickness.reshape(grid_shape)
    # Create a plot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    # Add coastlines and other features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # Plot the data using pcolormesh
    mesh = ax.pcolormesh(lon_grid, lat_grid, thickness_grid, transform=ccrs.PlateCarree(), cmap='viridis')
    # Add a colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', label='Sea Ice Thickness (m)')
    # Set title
    ax.set_title('Sea Ice Thickness')
    # Show the plot
    plt.show() """
    
def plot(lat, lon, si_thickness):
    size = len(lat)
    side_length = int(np.sqrt(size))
    while size % side_length != 0:
        side_length -= 1
    grid_shape = (side_length, size // side_length)
    
    lat_grid = lat.reshape(grid_shape)
    lon_grid = lon.reshape(grid_shape)
    thickness_grid = si_thickness.reshape(grid_shape)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

    # Plot the sea ice thickness
    mesh = ax.pcolormesh(lon_grid, lat_grid, thickness_grid, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=1)
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
    
    
    cbar = plt.colorbar(mesh, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')
    plt.title('Sea Ice Thickness')
    plt.show()
    
    
lat, lon, si_thickness = get_cpom(cpom_oct)
plot(lat, lon, si_thickness)