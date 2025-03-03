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


cpom_oct_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_10.map.nc"
cpom_nov_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_11.map.nc"
cpom_dec_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\CPOM\thk_2021_12.map.nc"


cryo_oct_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2021\ubristol_cryosat2_seaicethickness_nh25km_2021_10_v1.nc"
cryo_nov_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2021\ubristol_cryosat2_seaicethickness_nh25km_2021_11_v1.nc"
cryo_dec_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2021\ubristol_cryosat2_seaicethickness_nh25km_2021_12_v1.nc"

smos_oct

def get_cpom(path):
    data = nc.Dataset(path)
    
    print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['thickness'][:]
    grid_spacing = data.variables["grid_spacing"][:]
    si_thickness_stdev = data.variables["thk_stdev"][:]
    
    print(f"Lat shape: {lat.shape}")
    print(f"Lon shape: {lon.shape}")
    print(f"Thickness shape: {si_thickness.shape}")
    
    return lat, lon, si_thickness

def get_cryo(path):
    data = nc.Dataset(path)
    print(data.variables.keys())
    lat = data.variables['Latitude'][:]
    lon = data.variables['Longitude'][:]
    si_thickness = data.variables['Sea_Ice_Thickness'][:]
    si_thickness_un = data.variables['Sea_Ice_Thickness_Uncertainty'][:]
    
    # Check if lat and lon are 1D and need reshaping
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
        print('Reshaped lat and lon')
    # Mask invalid data
    mask = ~np.isnan(si_thickness)
    filtered_si_thickness = np.where(mask, si_thickness, np.nan)
    return lat, lon, filtered_si_thickness, si_thickness_un
    
def plot_cpom(lat, lon, si_thickness):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

    # Plot the sea ice thickness
    sc = ax.scatter(lon, lat, c=si_thickness, s=1, transform=ccrs.PlateCarree(), cmap='jet', vmin=0, vmax=3.3, zorder=1)
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
    
    
    cbar = plt.colorbar(sc, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')
    plt.title('Sea Ice Thickness')
    plt.show()


def plot_cryo(lat, lon, si_thickness):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()})
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
 
    mesh = ax.pcolormesh(lon, lat, si_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=1)
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
   
    cbar = plt.colorbar(mesh, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')
    
    plt.title('Sea Ice Thickness')
    plt.show()
    
    



if __name__ == "__main__":
    lat_cpom, lon_cpom, si_thickness_cpom = get_cpom(cpom_oct_2021)
 
    
    lat_cryo, lon_cryo, si_thickness_cryo, si_thickness_un_cryo = get_cryo(cryo_oct_2021)