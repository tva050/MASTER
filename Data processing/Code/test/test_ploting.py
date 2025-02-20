import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs, feature as cfeature


def pcolormesh_plot_cryo():
    # Load NetCDF file
    oct_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\uit_cryosat2_L3_EASE2_nh25km_2023_10_v3.nc"
    netcdf_file = nc.Dataset(oct_path)

    # Read latitude & longitude (already 2D)
    lat = netcdf_file['latitude'][:]
    lon = netcdf_file['longitude'][:]

    # **Fix: Read sea ice thickness correctly as 2D**
    si_thickness = netcdf_file['sea_ice_thickness'][:]  # Keep full data
    si_thickness_uncertainty = netcdf_file['sea_ice_thickness_uncertainty'][:]  # Keep full data

    # Print shapes to verify
    print("Lat shape:", lat.shape, "Dimensions:", netcdf_file['latitude'].dimensions)
    print("Lon shape:", lon.shape, "Dimensions:", netcdf_file['longitude'].dimensions)
    print("SI Thickness shape:", si_thickness.shape, "Dimensions:", netcdf_file['sea_ice_thickness'].dimensions)
    print("SI Thickness Uncertainty shape:", si_thickness_uncertainty.shape, "Dimensions:", netcdf_file['sea_ice_thickness_uncertainty'].dimensions)

    # **Flatten all arrays to 1D for plotting**
    lat = lat.flatten()
    lon = lon.flatten()
    si_thickness = si_thickness.flatten()
    si_thickness_uncertainty = si_thickness_uncertainty.flatten()

    # Apply mask for valid data points
    valid_mask = (~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(si_thickness) & ~np.isnan(si_thickness_uncertainty))

    # Filter valid data
    filtered_lat = lat[valid_mask]
    filtered_lon = lon[valid_mask]
    filtered_si_thickness = si_thickness[valid_mask]

    # Set limits for sea ice thickness
    si_max = 1
    si_min = 0

    # Create figure and plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)

    # Use tricontourf for smooth interpolation
    contour = ax.tricontourf(filtered_lon, filtered_lat, filtered_si_thickness, levels=20, cmap='viridis', vmin=si_min, vmax=si_max, transform=ccrs.PlateCarree())

    # Add colorbar
    cbar = plt.colorbar(contour, label='Sea Ice Thickness (m)')
    plt.show()


def pcolormesh_smos():
    smos_oct_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\mean_sea_ice_thickness_oct.nc" 
    
    data = nc.Dataset(smos_oct_path)
    
    #print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['mean_sea_ice_thickness'][:]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
    
    pcm = ax.pcolormesh(lon, lat, si_thickness, cmap='viridis', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(pcm, label='Sea Ice Thickness (m)')
    plt.show()
    
#pcolormesh_plot_cryo()
pcolormesh_smos()