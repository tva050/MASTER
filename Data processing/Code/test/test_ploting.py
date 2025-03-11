import os
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from scipy.interpolate import griddata

cryo_oct_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2021\ubristol_cryosat2_seaicethickness_nh25km_2021_10_v1.nc"
cryo_nov_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2021\ubristol_cryosat2_seaicethickness_nh25km_2021_11_v1.nc"
cryo_dec_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2021\ubristol_cryosat2_seaicethickness_nh25km_2021_12_v1.nc"

smos_oct_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\oct_mean_thickness.nc"
smos_nov_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\nov_mean_thickness.nc"
smos_dec_2021 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2021\dec_mean_thickness.nc"


def get_cryo(path):
    data = nc.Dataset(path)
    #print(data.variables.keys())
    
    lat = data.variables['Latitude'][:]
    lon = data.variables['Longitude'][:]
    si_thickness = data.variables['Sea_Ice_Thickness'][:]
    si_thickness_un = data.variables['Sea_Ice_Thickness_Uncertainty'][:]
    
    # Check if lat and lon are 1D and need reshaping
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
        print('Reshaped lat and lon')
    # Mask invalid data
    mask = ~np.isnan(si_thickness)
    filtered_si_thickness = np.where(mask, si_thickness, np.nan)
    return lat, lon, filtered_si_thickness



def get_smos(path):
    data = nc.Dataset(path)
    # print(data.variables.keys())
    
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    si_thickness = data.variables['sea_ice_thickness'][:]
    
    return lat, lon, si_thickness


def interpolate_data(source_lat, source_lon, source_si, target_lat, target_lon):
    source_points = np.array([source_lat.flatten(), source_lon.flatten()]).T
    source_values = source_si.flatten()
    
    target_points = np.array([target_lat.flatten(), target_lon.flatten()]).T
    
    source_interp = griddata(source_points, source_values, target_points, method='linear')
    return source_interp.reshape(target_lat.shape)

def box_plot_comp(cryo_interp, smos_si):
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
    
    cryo_flat = cryo_interp.flatten()
    smos_flat = smos_si.flatten()
    
    valid_mask = ~np.isnan(cryo_flat) & ~np.isnan(smos_flat)
    cryo_flat = cryo_flat[valid_mask]
    smos_flat = smos_flat[valid_mask]
    
    binned_smos_data = []
    for i in range(len(bins) - 1):
        mask = (smos_flat >= bins[i]) & (smos_flat < bins[i + 1])
        binned_smos_data.append(cryo_flat[mask])
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(binned_smos_data, labels=bin_labels)
    plt.xlabel('SMOS Sea Ice Thickness Bins (m)')
    plt.ylabel('CryoSat-2 Sea Ice Thickness (m)')
    plt.title('Comparison of Sea Ice Thickness Estimates')
    plt.grid(True)
    plt.show()
    

cryo_lat, cryo_lon, cryo_oct_si = get_cryo(cryo_oct_2021)
smos_lat, smos_lon, smos_oct_si = get_smos(smos_oct_2021)

cryo_interp = interpolate_data(cryo_lat, cryo_lon, cryo_oct_si, smos_lat, smos_lon)
box_plot_comp(cryo_interp, smos_oct_si)