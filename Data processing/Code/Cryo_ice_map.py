import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap


oct_path = "C:\\Users\\trym7\\OneDrive - UiT Office 365\\skole\\MASTER\\Data processing\\Data\\uit_cryosat2_L3_EASE2_nh25km_2023_10_v3.nc"
nov_path = "C:\\Users\\trym7\\OneDrive - UiT Office 365\\skole\\MASTER\\Data processing\\Data\\uit_cryosat2_L3_EASE2_nh25km_2023_11_v3.nc"
dec_path = "C:\\Users\\trym7\\OneDrive - UiT Office 365\\skole\\MASTER\\Data processing\\Data\\uit_cryosat2_L3_EASE2_nh25km_2023_12_v3.nc"
# Load data
data = nc.Dataset(oct_path)

#print(data.variables.keys())

time = data.variables['time'][:]
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
si_thickness = data.variables['sea_ice_thickness'][:]
si_thickness_uncertainty = data.variables['sea_ice_thickness_uncertainty'][:]

print(f"Lat range: {lat.min()} to {lat.max()}")
print(f"Lon range: {lon.min()} to {lon.max()}")
print(f"Sea Ice Thickness range: {si_thickness.min()} to {si_thickness.max()}")

mask = ~np.isnan(si_thickness) & ~np.isnan(si_thickness_uncertainty) & ~np.isnan(lat) & ~np.isnan(lon)

filtered_lat = lat[mask]
filtered_lon = lon[mask]
filtered_si_thickness = si_thickness[mask]
filtered_si_thickness_uncertainty = si_thickness_uncertainty[mask]

# Create figure and map
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5)
sc = ax.scatter(filtered_lon, filtered_lat, c=filtered_si_thickness, s=1, cmap='viridis', vmin=0, vmax=4 ,transform=ccrs.PlateCarree())
plt.colorbar(sc, label='Sea Ice Thickness (m)')
plt.show()