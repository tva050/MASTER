import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap


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
    
    
single_figure(*get_data(path))