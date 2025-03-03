import os
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from scipy.interpolate import griddata
cpom_oct = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_10.map.nc"
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
    
    return lat, lon, si_thickness
def plot(lat, lon, si_thickness):
    # Convert lat/lon to the projection's coordinate system
    proj = ccrs.NorthPolarStereo()
    x, y = proj.transform_points(ccrs.PlateCarree(), lon, lat)[:, :2].T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=proj)
    # Plot the sea ice thickness using scatter
    scatter = ax.scatter(lon, lat, c=si_thickness, s=1, transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=3.3, zorder=1)
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=4)
    
    cbar = plt.colorbar(scatter, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')
    plt.title('Sea Ice Thickness')
    plt.show()
lat, lon, si_thickness = get_cpom(cpom_oct)
plot(lat, lon, si_thickness)