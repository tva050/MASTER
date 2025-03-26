import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from  scipy.interpolate import griddata
import pandas as pd
from pyproj import Proj, Transformer, transform
import cartopy.crs as ccrs
import seaborn as sns
from mpl_toolkits.basemap import Basemap


import warnings
warnings.filterwarnings("ignore", message="facecolor will have no effect*")

oib_paths_2013 = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130321.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130322.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130323.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130324.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130326.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130327.txt"
]

# 	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130425.txt",

smos_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2013\2013_mean_thickness.nc"

cryo_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2013\uit_cryosat2_L3_EASE2_nh25km_2013_03_v3.nc"

def get_data_oib(path):
	df = pd.read_csv(path, dtype=str, low_memory=False)  # Read as strings to avoid mixed types
	df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, forcing errors to NaN

	# Extract required numerical columns
	lat = df["lat"].values
	lon = df["lon"].values
	thickness = df["thickness"].values
	thickness_un = df["thickness_unc"].values

	# Apply mask to remove invalid data
	mask = (thickness != -99999.0000) & (thickness_un != -99999.0000) 
	thickness = np.where(mask, thickness, np.nan)
	thickness_un = np.where(mask, thickness_un, np.nan)

	return lat, lon, thickness, thickness_un

def extract_all_oib(oib_paths):
	all_lat, all_lon, all_thickness, all_thickness_un = [], [], [], []
	for path in oib_paths:
		lat, lon, thickness, thickness_un = get_data_oib(path)


		all_lat.extend(lat)
		all_lon.extend(lon)
		all_thickness.extend(thickness)
		all_thickness_un.extend(thickness_un)
	return all_lat, all_lon, all_thickness, all_thickness_un


def get_smos(path):
	data = nc.Dataset(path)
	#print(data.variables.keys())
	lat = data.variables['latitude'][:]
	lon = data.variables['longitude'][:]
	si_thickness = data.variables['sea_ice_thickness'][:]
	si_thickness_un = data.variables['sea_ice_thickness_unc'][:]
 
	#mask = ~np.isnan(si_thickness_un)
	#si_thickness_un = np.where(mask, si_thickness_un, np.nan)
	
	#print(si_thickness.shape)
	return lat, lon, si_thickness, si_thickness_un


def get_cryo(path):
	data = nc.Dataset(path)
	#print(data.variables.keys())
	
	lat = data.variables['latitude'][:]
	lon = data.variables['longitude'][:]
	si_thickness = data.variables['sea_ice_thickness'][:]
	si_thickness_un = data.variables['sea_ice_thickness_uncertainty'][:]
	
	# Check if lat and lon are 1D and need reshaping
	if lat.ndim == 1 and lon.ndim == 1:
		lon, lat = np.meshgrid(lon, lat)
		print('Reshaped lat and lon')
	# Mask invalid data
	mask = ~np.isnan(si_thickness)
	filtered_si_thickness = np.where(mask, si_thickness, np.nan)
	return lat, lon, filtered_si_thickness, si_thickness_un



oib_lat, oib_lon, oib_sit, oib_sit_un = extract_all_oib(oib_paths_2013)
oib_lat, oib_lon, oib_sit, oib_sit_un = np.array(oib_lat), np.array(oib_lon), np.array(oib_sit), np.array(oib_sit_un)
smos_lat, smos_lon, smos_sit, smos_sit_un = get_smos(smos_path)
cryo_lat, cryo_lon, cryo_sit, cryo_sit_un = get_cryo(cryo_path)

print('OIB:', oib_lat.shape, oib_lon.shape, oib_sit.shape)
print('SMOS:', smos_lat.shape, smos_lon.shape, smos_sit.shape)
print('Cryo:', cryo_lat.shape, cryo_lon.shape, cryo_sit.shape)
#print("cryo un:", np.sum(np.isnan(_sit_un)))

def reprojecting(lon, lat, proj=ccrs.NorthPolarStereo()):
	transformer = proj.transform_points(ccrs.PlateCarree(), lon, lat)
	x = transformer[..., 0]
	y = transformer[..., 1]
	return x, y

x_oib, y_oib = reprojecting(oib_lon, oib_lat)
x_smos, y_smos = reprojecting(smos_lon, smos_lat)
x_cryo, y_cryo = reprojecting(cryo_lon, cryo_lat)

print('Reprojected OIB:', x_oib.shape, y_oib.shape)
print('Reprojected SMOS:', x_smos.shape, y_smos.shape)
print('Reprojected Cryo:', x_cryo.shape, y_cryo.shape)


def resample_to_cryo_grid(x_source, y_source, source_sit, source_sit_un, x_target, y_target, radiues = 12500):
	""" 
	Resample oib data onto a new grid using weighted average based on uncertainty.
 	"""
	target_tree = cKDTree(np.column_stack([x_target.ravel(), y_target.ravel()]))
	
	# initialize arrays to store resampled data and weights sums
	resampled_sit = np.full(x_target.shape, np.nan)
	weights_sum = np.full(x_target.shape, 0.0)
 
	source_sit = source_sit.ravel()
	source_sit_un = source_sit_un.ravel()
	x_source = x_source.ravel()
	y_source = y_source.ravel()
 
	for i in range(len(source_sit)):
		if np.isnan(source_sit[i]) or np.isnan(source_sit_un[i]):
			continue
		
		indices = target_tree.query_ball_point([x_source[i], y_source[i]], radiues)

		weigth = 1 / source_sit_un[i]

		for idx in indices:
			row, col = np.unravel_index(idx, x_target.shape)
			resampled_sit[row, col] = (np.nansum([resampled_sit[row, col] * weights_sum[row, col], oib_sit[i] * weigth]) / (weights_sum[row, col] + weigth))
			weights_sum[row, col] += weigth
   
	resampled_sit = np.ma.masked_invalid(resampled_sit)
	return resampled_sit

def resample_to_grid(x_source, y_source, source_sit, source_sit_un, x_target, y_target, radius=12500):
    """ 
    Resample source data onto a new grid using a weighted average based on uncertainty.
    """
    target_tree = cKDTree(np.column_stack([x_target.ravel(), y_target.ravel()]))

    # Initialize arrays to store resampled data and weight sums
    resampled_sit = np.full(x_target.shape, np.nan)
    weights_sum = np.full(x_target.shape, 0.0)

    # Flatten inputs for easier indexing
    source_sit = source_sit.ravel()
    source_sit_un = source_sit_un.ravel()
    x_source = x_source.ravel()
    y_source = y_source.ravel()

    for i in range(len(source_sit)):
        if np.isnan(source_sit[i]) or np.isnan(source_sit_un[i]):
            continue  # Skip invalid data points

        indices = np.array(target_tree.query_ball_point([x_source[i], y_source[i]], radius))
        indices = indices[indices < x_target.size]  # Ensure indices are within valid range

        if len(indices) == 0:
            continue  # Skip if no neighbors found

        # Compute weight, avoiding division by zero
        weigth = 1 / source_sit_un[i] if source_sit_un[i] > 0 else 0

        for idx in indices:
            if idx >= x_target.size:
                continue  # Skip out-of-bounds index

            row, col = np.unravel_index(idx, x_target.shape)
            
            # Compute weighted sum only if denominator is valid
            denominator = weights_sum[row, col] + weigth
            if denominator > 0:
                resampled_sit[row, col] = np.nansum([
                    resampled_sit[row, col] * weights_sum[row, col], 
                    source_sit[i] * weigth
                ]) / denominator

            weights_sum[row, col] += weigth  # Update weight sum

    resampled_sit = np.ma.masked_invalid(resampled_sit)
    return resampled_sit

resampled_oib_sit = resample_to_cryo_grid(x_oib, y_oib, oib_sit, oib_sit_un, x_cryo, y_cryo)
resampled_smos_sit = resample_to_grid(x_smos, y_smos, smos_sit, smos_sit_un, x_cryo, y_cryo)
print('Resampled SMOS:', resampled_smos_sit)

def hist_cryo_smos_oib(cryo_sit, oib_sit):
	cryo_sit = cryo_sit.flatten()
	oib_sit = oib_sit.flatten()
	
	# Create histogram
	plt.hist(oib_sit, bins=100, alpha=0.5, color='green', label='OIB gridded 25km', density=True)
	plt.hist(cryo_sit, bins=100, alpha=0.5, color='blue', label='OiB raw', density=True)
	
	# Labels and title
	plt.xlabel("Sea Ice Thickness [m]")
	plt.ylabel("PDF")
	plt.title("Histogram of Sea Ice Thickness")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()


def plot_gridded_data(x_cryo, y_cryo, gridded_oib_sit):
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

	# Plot the gridded data
	scatter = ax.pcolormesh(x_cryo, y_cryo, gridded_oib_sit, transform=ccrs.PlateCarree(), cmap='viridis', shading='auto', vmin=0, vmax=3.5)
	#scatter = ax.scatter(x_cryo, y_cryo, c=gridded_oib_sit, transform=ccrs.PlateCarree(), cmap='coolwarm', s=1)

	# Add features
	ax.coastlines()
	ax.add_feature(cfeature.LAND, facecolor="lightgray")
	ax.add_feature(cfeature.BORDERS, linestyle=":")

	# Title and color bar
	ax.set_title("Gridded Mean OiB Sea Ice Thickness")
	plt.colorbar(scatter, label="Sea Ice Thickness (m)")
	plt.show()
	
#hist_cryo_smos_oib(oib_sit, resampled_oib_sit)
plot_gridded_data(cryo_lon, cryo_lat, resampled_smos_sit)
hist_cryo_smos_oib(smos_sit, resampled_smos_sit)