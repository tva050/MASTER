import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time


folder_path_oct = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\oct"
folder_path_nov = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\nov"
folder_path_dec = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\dec"


def get_data(folder_path):
    # Initialize lists to store daily data
    si_tickness_data = []
    lat, lon = None, None  # Placeholder for coordinates

    for filename in os.listdir(folder_path):
        if filename.endswith('.nc'):
            file_path = os.path.join(folder_path, filename)
            
            with nc.Dataset(file_path, "r") as dataset:
                si_thickness = dataset.variables['sea_ice_thickness'][:]
                
                # Filter out NaN, -999.0, and 0.0 values
                mask = ~np.isnan(si_thickness) & (si_thickness != -999.0) & (si_thickness != 0.0)
                si_thickness = np.where(mask, si_thickness, np.nan)
                
                
                # Store the thickness data
                si_tickness_data.append(si_thickness)
                
                # Get lat/lon from the first file
                if lat is None or lon is None:
                    lat = dataset.variables['latitude'][:]
                    lon = dataset.variables['longitude'][:]
   
    si_thickness_data = np.array(si_tickness_data)
    monthly_mean_thickness = np.mean(si_thickness_data, axis=0)
    return lat, lon, monthly_mean_thickness


def plot_data(lon, lat, si_thickness):
    # Check if lon and lat are already 2D
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

    # Plot the sea ice thickness
    mesh = ax.pcolormesh(lon, lat, si_thickness[0, :, :], transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=1)
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
    
    
    cbar = plt.colorbar(mesh, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')
    plt.title('Sea Ice Thickness')
    plt.show()


def store_to_file(lon, lat, si_thickness, output_file):
    with nc.Dataset(output_file, "w", format="NETCDF4") as dataset:
        # Define dimensions
        dataset.createDimension("time", 1)  # Single time step
        dataset.createDimension("lat", 896)
        dataset.createDimension("lon", 608)
        
        # Create variables
        times = dataset.createVariable("time", "f4", ("time",))
        lats = dataset.createVariable("latitude", "f4", ("lat", "lon"))
        lons = dataset.createVariable("longitude", "f4", ("lat", "lon"))
        sea_ice_thickness = dataset.createVariable("sea_ice_thickness", "f4", ("time", "lat", "lon"), fill_value=np.nan)
        
        # Write data
        times[:] = [0]  # Arbitrary time value
        lats[:, :] = lat
        lons[:, :] = lon
        sea_ice_thickness[0, :, :] = si_thickness
        
        # Metadata
        dataset.description = "Monthly mean sea ice thickness datset"
        dataset.source = "Processed from daily SMOS data"
        dataset.history = "Created " + time.ctime(time.time())
        


if __name__ == "__main__":
    lat, lon, monthly_mean_thickness = get_data(folder_path_dec)
    #plot_data(lon, lat, monthly_mean_thickness)
    store_to_file(lon, lat, monthly_mean_thickness, r'C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\dec_mean_thickness.nc')