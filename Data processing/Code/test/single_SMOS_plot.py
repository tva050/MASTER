import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2024\oct\SMOS_Icethickness_v3.3_north_20221015.nc"
def get_data(path):    
    data = nc.Dataset(path)
    print(data.variables.keys())
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['sea_ice_thickness'][:]
    
    # Assuming si_thickness is 3D, select the first time slice
    si_thickness = si_thickness[0, :, :]
    
    # Mask invalid values
    mask = ~np.isnan(si_thickness) & (si_thickness != -999.0) & (si_thickness != 0.0)
    filtered_si_thickness = np.where(mask, si_thickness, np.nan)
    
    return lat, lon, filtered_si_thickness
lat, lon, si_thickness = get_data(path)
# Check if lon and lat are already 2D
if lon.ndim == 1 and lat.ndim == 1:
    lon, lat = np.meshgrid(lon, lat)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
ax.set_extent([-1e6, 1e6, -1e6, 1e6], crs=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
# Plot the sea ice thickness
mesh = ax.pcolormesh(lon, lat, si_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)
cbar = plt.colorbar(mesh, orientation='vertical')
cbar.set_label('Sea Ice Thickness (m)')
plt.title('Sea Ice Thickness')
plt.show()
