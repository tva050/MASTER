import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from netCDF4 import Dataset

folder_path_2013= r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2013"

one_smos_file = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\test data\SMOS_Icethickness_v3.3_north_20221216.nc"

def print_nc_metadata(file_path):
    """Prints metadata from a NetCDF (.nc) file."""
    # Open the NetCDF file
    with Dataset(file_path, 'r') as nc_file:
        print(nc_file.variables.keys())
        # Print global attributes
        print("Global Attributes:")
        for attr in nc_file.ncattrs():
            print(f"{attr}: {nc_file.getncattr(attr)}")
        
        print("\nVariables:")
        for var_name, var in nc_file.variables.items():
            print(f"{var_name}:")
            print(f"  Dimensions: {var.dimensions}")
            print(f"  Shape: {var.shape}")
            print(f"  Data Type: {var.dtype}")
            
            # Print variable attributes
            for attr in var.ncattrs():
                print(f"  {attr}: {var.getncattr(attr)}")

def get_data(folder_path):
    # Initialize lists to store daily data
    si_tickness_data = []
    sit_un_data = []
    lat, lon = None, None  # Placeholder for coordinates
    for filename in os.listdir(folder_path):
        if filename.endswith('.nc'):
            file_path = os.path.join(folder_path, filename)
            with nc.Dataset(file_path, "r") as dataset:
                si_thickness = dataset.variables['sea_ice_thickness'][:]
                # apply land mask, used for smos data to remove land points 
                if 'land' in dataset.variables:
                    land_mask = dataset.variables['land'][:]  # Extract first time step
                else:
                    print(f"Warning: 'land' variable not found in {filename}. Using default mask.")
                    land_mask = np.zeros_like(si_thickness)  # Default to no land mask          
                         
                si_thickness = np.where(land_mask == 1, np.nan, si_thickness) 
                
                if 'ice_thickness_uncertainty' in dataset.variables:
                    sit_un = dataset.variables['ice_thickness_uncertainty'][:]
                else:
                    print(f"Warning: 'ice_thickness_uncertainty' not found in {filename}. Using NaNs.")
                    sit_un = np.full_like(si_thickness, np.nan)  # Fill with NaNs
                    
                # Filter out NaN, -999.0, and 0.0 values
                mask_sit = ~np.isnan(si_thickness) & (si_thickness != -999.0) & (si_thickness != 0.0)
                mask_sit_un = ~np.isnan(sit_un) & (sit_un != -999.0) & (sit_un != 0.0)
                
                si_thickness = np.where(mask_sit, si_thickness, np.nan)
                sit_un = np.where(mask_sit_un, sit_un, np.nan)
                
                # Store the thickness data
                si_tickness_data.append(si_thickness)
                sit_un_data.append(sit_un)
                # Get lat/lon from the first file
                if lat is None or lon is None:
                    lat = dataset.variables['latitude'][:]
                    lon = dataset.variables['longitude'][:]
   
    si_thickness_data = np.array(si_tickness_data)
    sit_un_data = np.array(sit_un_data)
    monthly_mean_thickness = np.mean(si_thickness_data, axis=0)
    monthly_mean_thickness_un = np.mean(sit_un_data, axis=0)
    return lat, lon, monthly_mean_thickness, monthly_mean_thickness_un


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


def store_to_file(lon, lat, si_thickness, si_thickness_unc, output_file):
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
        sea_ice_thickness_unc = dataset.createVariable("sea_ice_thickness_unc", "f4", ("time", "lat", "lon"), fill_value=np.nan)
        
        # Write data
        times[:] = [0]  # Arbitrary time value
        lats[:, :] = lat
        lons[:, :] = lon
        sea_ice_thickness[0, :, :] = si_thickness
        sea_ice_thickness_unc[0, :, :] = si_thickness_unc
        
        # Metadata
        dataset.description = "Monthly mean sea ice thickness datset"
        dataset.source = "Processed from daily SMOS data"
        dataset.history = "Created " + time.ctime(time.time())
        


if __name__ == "__main__":
    print_nc_metadata(one_smos_file)
    #lat, lon, monthly_mean_thickness, monthly_mean_thickness_unc = get_data(folder_path_2013)
    # print number of nan values in monthly_mean_thickness_unc
    #print(np.sum(np.isnan(monthly_mean_thickness_unc)))
    #plot_data(lon, lat, monthly_mean_thickness)
    #print(lon.shape, lat.shape, monthly_mean_thickness.shape)
    #store_to_file(lon, lat, monthly_mean_thickness, monthly_mean_thickness_unc, r'C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2013\2013_mean_thickness.nc')