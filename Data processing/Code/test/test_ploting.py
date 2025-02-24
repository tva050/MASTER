import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Path to the folder containing the daily files
folder_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2024\oct"

# Initialize lists to store daily data
thickness_data = []
lat, lon = None, None  # Placeholder for coordinates

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.nc'):  # Assuming the files are NetCDF
        file_path = os.path.join(folder_path, filename)
        
        with nc.Dataset(file_path, 'r') as ds:
            sea_ice_thickness = ds.variables['sea_ice_thickness'][:]
            print('sea_ice_thickness shape:', sea_ice_thickness.shape)
            # Filter out NaN, -999.0, and 0.0 values
            sea_ice_thickness = np.where((sea_ice_thickness == -999.0) | 
                                         (sea_ice_thickness == 0.0), np.nan, sea_ice_thickness)
            
            # Store the thickness data
            thickness_data.append(sea_ice_thickness)
            
            # Get lat/lon from the first file
            if lat is None or lon is None:
                lat = ds.variables['latitude'][:]
                lon = ds.variables['longitude'][:]

# Convert list to a NumPy array (time, lat, lon)
thickness_data = np.array(thickness_data)

# Compute monthly mean sea ice thickness (ignoring NaNs)
monthly_mean_thickness = np.nanmean(thickness_data, axis=0)


# Store lat, lon and the monthly mean thickness in a nc file, to keep the sea ice thickness data shape
output_file = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\monthly_mean_thickness.nc"

# Create a new NetCDF file
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
    lats[:, :] = lat  # Ensure lat is 2D
    lons[:, :] = lon  # Ensure lon is 2D
    sea_ice_thickness[0, :, :] = monthly_mean_thickness  # Maintain shape (1, 896, 608)

    # Add metadata
    dataset.description = "Monthly mean sea ice thickness dataset"
    dataset.source = "Processed from daily SMOS data"


# Check if lon and lat are already 2D
if lon.ndim == 1 and lat.ndim == 1:
    lon, lat = np.meshgrid(lon, lat)
    print('2D lon and lat created')

# Create the plot
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()})
ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)

# Plot the monthly average sea ice thickness
mesh = ax.pcolormesh(lon, lat, monthly_mean_thickness[0, :, :], transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)

# Add a colorbar
cbar = plt.colorbar(mesh, orientation='vertical')
cbar.set_label('Monthly Mean Sea Ice Thickness (m)')

# Set the title and show the plot
plt.title('Monthly Mean Sea Ice Thickness')
plt.show()
