import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs, feature as cfeature

path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\uit_cryosat2_L3_EASE2_nh25km_2023_10_v3.nc"

data = nc.Dataset(path)

print(data.variables.keys())


lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
si_thickness = data.variables['sea_ice_thickness'][:]
# Check if lat and lon are 1D and need reshaping
if lat.ndim == 1 and lon.ndim == 1:
    lon, lat = np.meshgrid(lon, lat)
    print('Reshaped lat and lon')
# Mask invalid data
mask = ~np.isnan(si_thickness)
filtered_si_thickness = np.where(mask, si_thickness, np.nan)


# Create a map plot
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()})
# Add coastlines and gridlines
ax.coastlines()
ax.gridlines(draw_labels=True)
# Plot the data using pcolormesh
mesh = ax.pcolormesh(lon, lat, filtered_si_thickness,
                     transform=ccrs.PlateCarree(), cmap='viridis')
# Add a colorbar
cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.5, pad=0.05)
cbar.set_label('Sea Ice Thickness (m)')
# Add features
ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
ax.add_feature(cfeature.OCEAN, zorder=0)
# Show the plot
plt.title('Sea Ice Thickness')
plt.show()