import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# Path to the folder containing the daily files
folder_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2024\dec"
# Initialize lists to store daily means
daily_means = []
# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.nc'):  # Assuming the files are NetCDF
        file_path = os.path.join(folder_path, filename)
        
        # Load the data
        with nc.Dataset(file_path, 'r') as ds:
            sea_ice_thickness = ds.variables['sea_ice_thickness'][:]
            
            # Filter out NaN, -999.0, and 0.0 values
            valid_thickness = sea_ice_thickness[(~np.isnan(sea_ice_thickness)) & 
                                                (sea_ice_thickness != -999.0) & 
                                                (sea_ice_thickness != 0.0)]
            
            # Calculate the mean thickness for the day if there are valid values
            if valid_thickness.size > 0:
                daily_mean_thickness = np.mean(valid_thickness)
                daily_means.append(daily_mean_thickness)
# Calculate the monthly mean
monthly_mean_thickness = np.mean(daily_means)
# Save the monthly mean to a new file
output_file = 'monthly_mean_sea_ice_thickness.csv'
pd.DataFrame({'Monthly Mean Thickness': [monthly_mean_thickness]}).to_csv(output_file, index=False)
# Plotting
# Use the latitude and longitude from one of the files
sample_file = os.path.join(folder_path, os.listdir(folder_path)[0])
with nc.Dataset(sample_file, 'r') as ds_sample:
    lat = ds_sample.variables['latitude'][:]
    lon = ds_sample.variables['longitude'][:]
    sea_ice_thickness = ds_sample.variables['sea_ice_thickness'][:]
# Check if lon and lat are already 2D
if lon.ndim == 1 and lat.ndim == 1:
    lon, lat = np.meshgrid(lon, lat)
# Create the plot
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()})
ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
# Plot the sea ice thickness
mesh = ax.pcolormesh(lon, lat, sea_ice_thickness[0, :, :], transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)
# Add a colorbar
cbar = plt.colorbar(mesh, orientation='vertical')
cbar.set_label('Sea Ice Thickness (m)')
# Set the title and show the plot
plt.title('Sea Ice Thickness')
plt.show()