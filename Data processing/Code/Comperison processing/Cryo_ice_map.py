import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from scipy.interpolate import griddata
from netCDF4 import Dataset

oct_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2012_03_v3.nc"
nov_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\uit_cryosat2_L3_EASE2_nh25km_2023_11_v3.nc"
dec_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\uit_cryosat2_L3_EASE2_nh25km_2023_12_v3.nc"


def get_data(path):
    data = nc.Dataset(path)
    #print(data.variables.keys())
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['sea_ice_thickness'][:]
    time = data.variables['time'][:]
    # Check if lat and lon are 1D and need reshaping
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
        print('Reshaped lat and lon')
    # Mask invalid data
    mask = ~np.isnan(si_thickness)
    filtered_si_thickness = np.where(mask, si_thickness, np.nan)
    print(time)
    return lat, lon, filtered_si_thickness

def print_nc_metadata(file_path):
    """Prints metadata from a NetCDF (.nc) file."""
    # Open the NetCDF file
    with Dataset(file_path, 'r') as nc_file:
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
    

def write_to_txt(filtered_lat, filtered_lon, filtered_si_thickness):
    with open('filtered_data.txt', 'w') as f:
        for i in range(len(filtered_lat)):
            f.write(f'{filtered_lat[i]} {filtered_lon[i]} {filtered_si_thickness[i]}\n')


def single_figure(lat, lon, si_thickness):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()})
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
 
    mesh = ax.pcolormesh(lon, lat, si_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=1)
    
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
   
    cbar = plt.colorbar(mesh, orientation='vertical')
    cbar.set_label('Sea Ice Thickness (m)')
    
    plt.title('Sea Ice Thickness')
    plt.show()


def zoomed_figure(lat, lon, si_thickness):
    fig = plt.figure(figsize=(10, 10))
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-1e6, 1e6, -1e6, 1e6], crs=ccrs.NorthPolarStereo())
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5)
    
    sc = ax.pcolormesh(lon, lat, si_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)
    
    plt.colorbar(sc, label='Sea Ice Thickness (m)')
    plt.show()

def compare_months_LARM(oct_path, nov_path, dec_path):
    # Compares the SIT for three different months from the same satellite CryoSat-2
    
    oct_lat, oct_lon, oct_si_thickness = get_data(oct_path)
    nov_lat, nov_lon, nov_si_thickness = get_data(nov_path)
    dec_lat, dec_lon, dec_si_thickness = get_data(dec_path)
    
    # Create figure with 3 subplots
    fig, ax = plt.subplots(1, 3, subplot_kw={'projection': ccrs.NorthPolarStereo()})
    
    month_labels = ['October', 'November', 'December']
    
    # Set extent for North Polar Stereographic projection
    for i,a in enumerate(ax):
        a.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
        a.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5)
        a.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        a.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5)
        a.set_title(month_labels[i], fontsize=14)
        
        
    # Add data to each subplot
    sc1 = ax[0].pcolormesh(oct_lon, oct_lat, oct_si_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)
    sc2 = ax[1].pcolormesh(nov_lon, nov_lat, nov_si_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)
    sc3 = ax[2].pcolormesh(dec_lon, dec_lat, dec_si_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)
    
    # Add an overall colorbar
    fig.colorbar(sc3, ax=ax, orientation='horizontal', label='Sea Ice Thickness (m)', pad=0.05, aspect=50)
    plt.show()
    
    
def bar():
    lat, lon, si_thickness = get_data(oct_path)
    cryo_thickness = si_thickness
    # making an bar plot of the mean and std of the sea ice thickness
    
    mean = np.nanmean(cryo_thickness)
    std = np.nanstd(cryo_thickness)
    
    print(mean)
    
    plt.bar('CryoSat-2', mean, yerr=std, capsize=5)
    plt.ylabel('Sea Ice Thickness (m)')
    plt.title('Mean Sea Ice Thickness Comparison')
    plt.show()

def box_plot():
    lat, lon, si_thickness = get_data(oct_path)
    cryo_si = si_thickness
    
    if cryo_si is None or cryo_si.size == 0:
        print("Error: CryoSat-2 data is empty!")
        return
    
    # Flatten the CryoSea Ice Thickness data and remove NaN values
    cryo_flat = cryo_si.flatten()
    cryo_flat = cryo_flat[~np.isnan(cryo_flat)]  # Remove NaN values
    
    # Check data (debugging)
    print("Data for CryoSat-2 (first 5 values):", cryo_flat[:5])
    
    # Create the box plot
    plt.figure(figsize=(6, 4))
    plt.boxplot(cryo_flat, patch_artist=True, boxprops=dict(facecolor="skyblue", color="blue"),
                medianprops=dict(color="red"))
    
    # Set plot labels
    plt.xlabel("Sea Ice Thickness (m)")
    plt.title("CryoSat-2 Sea Ice Thickness Distribution")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    lat, lon, si_thickness = get_data(oct_path)
    #print_nc_metadata(oct_path)
    print(si_thickness)
    #write_to_txt(filtered_lat, filtered_lon, filtered_si_thickness)
    single_figure(lat, lon, si_thickness)
    #single_figure(filtered_lat, filtered_lon, filtered_si_thickness)
    #zoomed_figure(filtered_lat, filtered_lon, filtered_si_thickness, filtered_si_thickness_uncertainty)
    #compare_months_LARM(oct_path, nov_path, dec_path)
    #bar()
    #box_plot()