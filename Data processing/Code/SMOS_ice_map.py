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

thickness_dict = defaultdict(list)

#print(data.variables.keys())

for filename in os.listdir(folder_path_oct):
    if filename.endswith(".nc"):
        data = os.path.join(folder_path_oct, filename)
        
        with nc.Dataset(data) as dataset:
            if 'latitude' in dataset.variables and 'longitude' in dataset.variables and 'sea_ice_thickness' in dataset.variables:
                latitude = np.array(dataset.variables['latitude'][:]).flatten()
                longitude = np.array(dataset.variables['longitude'][:]).flatten()
                sea_ice_thickness = np.array(dataset.variables['sea_ice_thickness'][:]).flatten()
                
                mask = ~np.isnan(sea_ice_thickness) & ~np.isnan(latitude) & ~np.isnan(longitude)
                latitude, longitude, sea_ice_thickness = latitude[mask], longitude[mask], sea_ice_thickness[mask]
                
                for lat, lon, thickness in zip(latitude, longitude, sea_ice_thickness):
                    thickness_dict[(lat, lon)].append(thickness)
                    
latitude, longitude, mean_thickness = [], [], []

for (lat, lon), values in thickness_dict.items():
    latitude.append(lat)
    longitude.append(lon)
    mean_thickness.append(np.mean(values))
    
    
latitudes = np.array(latitude)
longitudes = np.array(longitude)
mean_thickness = np.array(mean_thickness)

# --- Plotting ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())

# Define map extent
ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

# Add features
ax.add_feature(cfeature.LAND, color='lightgray')
ax.coastlines(resolution='50m')

# Scatter plot of mean sea ice thickness
sc = ax.scatter(longitudes, latitudes, c=mean_thickness, s=10, cmap='viridis', transform=ccrs.PlateCarree())

# Add colorbar
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7)
cbar.set_label("Mean Sea Ice Thickness (m)")

# Show plot
plt.title("Mean Sea Ice Thickness Across Multiple Files")
plt.show()