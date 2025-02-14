import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import netCDF4 as nc

# File path
oct_dir = "C:\\Users\\trym7\\OneDrive - UiT Office 365\\skole\\MASTER\\Data processing\\Data\\uit_cryosat2_L3_EASE2_nh25km_2023_10_v3.nc"

# Load data
data = nc.Dataset(oct_dir)

# Extract variables correctly
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
si_thickness = data.variables['sea_ice_thickness'][:]

# Check data range
print(f"Lat range: {lat.min()} to {lat.max()}")
print(f"Lon range: {lon.min()} to {lon.max()}")
print(f"Sea Ice Thickness range: {si_thickness.min()} to {si_thickness.max()}")

# Mask invalid values
mask = ~np.isnan(si_thickness) & ~np.isnan(lat) & ~np.isnan(lon)
filtered_lat = lat[mask]
filtered_lon = lon[mask]
filtered_si_thickness = si_thickness[mask]

# Create figure and map
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())

# Adjust extent for North Polar Stereographic projection
ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())  

# Add coastlines
ax.coastlines(resolution='50m')

# Use lat/lon directly with PlateCarree projection
sc = ax.scatter(filtered_lon, filtered_lat, c=filtered_si_thickness, 
                s=10, cmap='viridis', transform=ccrs.PlateCarree())

# Add colorbar
plt.colorbar(sc, label='Sea Ice Thickness (m)')

# Show plot
plt.show()
