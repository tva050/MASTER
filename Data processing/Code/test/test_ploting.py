import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from  scipy.interpolate import griddata
import pandas as pd
from pyproj import Proj, Transformer
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", message="facecolor will have no effect*")

oib_paths_2013 = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130321.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130322.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130323.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130324.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130425.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130326.txt"
]

oib_paths_2011 = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110316.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110317.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110318.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110322.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110323.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110325.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110326.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110328.txt"
]

smos_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2013\2013_mean_thickness.nc"

cryo_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2013\uit_cryosat2_L3_EASE2_nh25km_2013_03_v3.nc"

def get_data(path):
	df = pd.read_csv(path, dtype=str, low_memory=False)  # Read as strings to avoid mixed types
	df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, forcing errors to NaN

	# Extract required numerical columns
	lat = df["lat"].values
	lon = df["lon"].values
	thickness = df["thickness"].values

	# Apply mask to remove invalid data
	mask = (thickness != -99999.) & (thickness != 0.0)
	thickness = np.where(mask, thickness, np.nan)

	return lat, lon, thickness

def extract_all_oib(oib_paths):
	all_lat, all_lon, all_thickness = [], [], []
	for path in oib_paths:
		lat, lon, thickness = get_data(path)


		all_lat.extend(lat)
		all_lon.extend(lon)
		all_thickness.extend(thickness)
	return all_lat, all_lon, all_thickness 


def get_smos(path):
	data = nc.Dataset(path)
	#print(data.variables.keys())
	lat = data.variables['latitude'][:]
	lon = data.variables['longitude'][:]
	si_thickness = data.variables['sea_ice_thickness'][:]
	#print(si_thickness.shape)
	return lat, lon, si_thickness


def get_cryo(path):
	data = nc.Dataset(path)
	#print(data.variables.keys())
	
	lat = data.variables['latitude'][:]
	lon = data.variables['longitude'][:]
	si_thickness = data.variables['sea_ice_thickness'][:]
	#si_thickness_un = data.variables['Sea_Ice_Thickness_Uncertainty'][:]
	
	# Check if lat and lon are 1D and need reshaping
	if lat.ndim == 1 and lon.ndim == 1:
		lon, lat = np.meshgrid(lon, lat)
		print('Reshaped lat and lon')
	# Mask invalid data
	mask = ~np.isnan(si_thickness)
	filtered_si_thickness = np.where(mask, si_thickness, np.nan)
	return lat, lon, filtered_si_thickness



oib_lat, oib_lon, oib_sit = extract_all_oib(oib_paths_2013)
smos_lat, smos_lon, smos_sit = get_smos(smos_path)
cryo_lat, cryo_lon, cryo_sit = get_cryo(cryo_path)

def plot_data_oib(lon, lat, si_thickness):
	# Convert lon and lat to NumPy arrays if they are lists
	lon = np.array(lon)
	lat = np.array(lat)

	# Create a figure and an axis with a polar projection
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

	# Plot the sea ice thickness using scatter (for irregular grid)
	sc = ax.scatter(lon, lat, c=si_thickness, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)
	
	ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
	
	# Add a colorbar
	cbar = plt.colorbar(sc, orientation='vertical')
	cbar.set_label('Sea Ice Thickness (m)')
	
	plt.title('Sea Ice Thickness')
	plt.show()

def plot_cryo_oib(oib_lat, oib_lon, oib_sit, cs_lat, cs_lon, cs_sit):
	# Convert lon and lat to NumPy arrays if they are lists
	oib_lon = np.array(oib_lon)
	oib_lat = np.array(oib_lat)
	#cs_lon = np.array(cs_lon)
	#cs_lat = np.array(cs_lat)

	# Create a figure and an axis with a polar projection
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

	# Plot the sea ice thickness using scatter (for irregular grid)
	sc = ax.scatter(oib_lon, oib_lat, c=oib_sit, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)
	mesh = ax.pcolormesh(cs_lon, cs_lat, cs_sit, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=0)
	
	ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
	
	# Add a colorbar
	cbar = plt.colorbar(sc, orientation='vertical')
	cbar.set_label('Sea Ice Thickness (m)')
	
	plt.title('Sea Ice Thickness')
	plt.show()
 
def plot_smos_oib(oib_lat, oib_lon, oib_sit, smos_lat, smos_lon, smos_sit):
	# Convert lon and lat to NumPy arrays if they are lists
	oib_lon = np.array(oib_lon)
	oib_lat = np.array(oib_lat)
	#cs_lon = np.array(cs_lon)
	#cs_lat = np.array(cs_lat)

	# Create a figure and an axis with a polar projection
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

	# Plot the sea ice thickness using scatter (for irregular grid)
	sc = ax.scatter(oib_lon, oib_lat, c=oib_sit, cmap='viridis', vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)
	mesh = ax.pcolormesh(smos_lon, smos_lat, smos_sit, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=0)
	
	ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.COASTLINE, color = "gray", linewidth=0.5, zorder=4)
	
	# Add a colorbar
	cbar = plt.colorbar(sc, orientation='vertical')
	cbar.set_label('Sea Ice Thickness (m)')
	
	plt.title('Sea Ice Thickness')
	plt.show()
 
 
def hist_cryo_smos_oib(cryo_sit, smos_sit, oib_sit):
	cryo_sit = cryo_sit.flatten()
	smos_sit = smos_sit.flatten()
	
	# Create histogram
	plt.hist(oib_sit, bins=100, alpha=0.5, color='green', label='OIB Thickness')
	plt.hist(cryo_sit, bins=100, alpha=0.5, color='blue', label='CryoSat-2')
	plt.hist(smos_sit, bins=100, alpha=0.5, color='red', label='SMOS')
	
	# Labels and title
	plt.xlabel("Sea Ice Thickness [m]")
	plt.ylabel("Frequency")
	plt.title("Histogram: Comparison of CryoSat-2 and SMOS Thickness with OIB")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()

#---------------------------------------------------------------------------------#

def nearest_values(ref_lat, ref_lon, ref_sit, target_lat, target_lon, target_sit):
	"""
	Find the nearest sea ice thickness values from target dataset for reference locations.
	"""
	tree = cKDTree(list(zip(target_lat.ravel(), target_lon.ravel()))) # Create KD-tree for target dataset locations (flattened) 
	_, idxs = tree.query(list(zip(ref_lat, ref_lon))) # Find nearest target locations for reference locations 
	return target_sit.ravel()[idxs] # Return the nearest sea ice thickness values

def barplot_mean_thickness(oib_lat, oib_lon, oib_sit, cryo_lat, cryo_lon, cryo_sit, smos_lat, smos_lon, smos_sit):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
	
	# Convert lists to NumPy arrays
	oib_lat, oib_lon, oib_sit = np.array(oib_lat), np.array(oib_lon), np.array(oib_sit)
	
	# Remove NaNs
	mask = ~np.isnan(oib_sit)
	oib_lat, oib_lon, oib_sit = oib_lat[mask], oib_lon[mask], oib_sit[mask]
	
	# Assign bins
	bin_indices = np.digitize(oib_sit, bins) - 1
	
	mean_cryo, mean_smos, mean_oib = [], [], []
	
	for i in range(len(bins) - 1):
		# Get OIB data points in this bin
		bin_mask = bin_indices == i
		if np.sum(bin_mask) == 0:
			mean_oib.append(np.nan)
			mean_cryo.append(np.nan)
			mean_smos.append(np.nan)
			continue
		
		bin_oib_lat, bin_oib_lon, bin_oib_sit = oib_lat[bin_mask], oib_lon[bin_mask], oib_sit[bin_mask]
		
		# Find nearest CryoSat-2 and SMOS values
		nearest_cryo = nearest_values(bin_oib_lat, bin_oib_lon, bin_oib_sit, cryo_lat, cryo_lon, cryo_sit)
		nearest_smos = nearest_values(bin_oib_lat, bin_oib_lon, bin_oib_sit, smos_lat, smos_lon, smos_sit)
		
		# Compute mean values
		mean_oib.append(np.nanmean(bin_oib_sit))
		mean_cryo.append(np.nanmean(nearest_cryo))
		mean_smos.append(np.nanmean(nearest_smos))
	
	# Plot
	x = np.arange(len(bin_labels))
	width = 0.25
	
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.bar(x - width, mean_oib, width, label='OIB', color='green', alpha=0.7)
	ax.bar(x, mean_cryo, width, label='CryoSat-2', color='blue', alpha=0.7)
	ax.bar(x + width, mean_smos, width, label='SMOS', color='red', alpha=0.7)
	
	ax.set_xlabel('Sea Ice Thickness Bins [m]')
	ax.set_ylabel('Mean Sea Ice Thickness [m]')
	ax.set_title('Comparison of Mean Sea Ice Thickness')
	ax.set_xticks(x)
	ax.set_xticklabels(bin_labels)
	ax.legend()
	ax.grid(True, linestyle='--', alpha=0.5)
	
	plt.show()

# ---------------------------------------------------------------------------------#
def inverse_distance_weighting(lon, lat, sit, grid_lon, grid_lat, power=2):
	"""
	Perform IDW interpolation on a given dataset.
	
	Parameters:
		lon, lat: 1D arrays of data point coordinates
		sit: 1D array of sea ice thickness values
		grid_lon, grid_lat: 2D arrays defining the target grid
		power: power parameter for IDW (default = 2)
	
	Returns:
		Interpolated SIT values on the specified grid.
	"""
	grid_z = np.full(grid_lon.shape, np.nan)  # Initialize grid with NaNs

	# Stack coordinates for efficient distance calculations
	points = np.vstack((lon, lat)).T
	tree = cKDTree(points)  # KDTree for fast nearest neighbor search

	for i in range(grid_lon.shape[0]):
		for j in range(grid_lon.shape[1]):
			# Get distances and indices of nearest neighbors
			dists, indices = tree.query([grid_lon[i, j], grid_lat[i, j]], k=5)

			# Avoid division by zero
			dists = np.maximum(dists, 1e-10)
			weights = 1 / dists**power

			# Ensure indices are valid before applying them
			indices = np.array(indices, dtype=int)  # Ensure indices are integers
			valid_values = sit[indices] if indices.size > 0 else np.nan
			valid_values = sit[indices]
			valid_mask = ~np.isnan(valid_values)

			if np.any(valid_mask):  # If at least one valid neighbor exists
				weights = weights[valid_mask] / np.sum(weights[valid_mask])  # Normalize
				grid_z[i, j] = np.sum(valid_values[valid_mask] * weights)

	return grid_z
def generate_grid(lat_min=65, resolution=12.5e3):
	"""
	Generate a grid covering the region north of 65 degrees latitude.
	"""
	lon_range = np.linspace(-180, 180, int(360 * 1e3 / resolution))
	lat_range = np.linspace(lat_min, 90, int((90 - lat_min) * 1e3 / resolution))
	grid_lon, grid_lat = np.meshgrid(lon_range, lat_range)
	return grid_lon, grid_lat

def interpolate_sit(lat, lon, sit, resolution=12.5e3):
	"""
	Interpolate SIT dataset onto a regular grid using IDW.
	"""
	grid_lon, grid_lat = generate_grid(resolution=resolution)
	interpolated_sit = inverse_distance_weighting(lon, lat, sit, grid_lon, grid_lat)
	return grid_lon, grid_lat, interpolated_sit

def barplot_interpolated_sit(oib_lat, oib_lon, oib_sit, cryo_lat, cryo_lon, cryo_sit, smos_lat, smos_lon, smos_sit):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
	
	# Interpolate datasets
	
	_, _, interp_oib = interpolate_sit(oib_lat, oib_lon, oib_sit)
	_, _, interp_cryo = interpolate_sit(cryo_lat, cryo_lon, cryo_sit)
	_, _, interp_smos = interpolate_sit(smos_lat, smos_lon, smos_sit)
	
	mean_cryo, mean_smos, mean_oib = [], [], []
	
	for i in range(len(bins) - 1):
		mask_oib = (interp_oib >= bins[i]) & (interp_oib < bins[i + 1])
		mask_cryo = (interp_cryo >= bins[i]) & (interp_cryo < bins[i + 1])
		mask_smos = (interp_smos >= bins[i]) & (interp_smos < bins[i + 1])
		
		mean_oib.append(np.nanmean(interp_oib[mask_oib]))
		mean_cryo.append(np.nanmean(interp_cryo[mask_cryo]))
		mean_smos.append(np.nanmean(interp_smos[mask_smos]))
	
	# Plot
	x = np.arange(len(bin_labels))
	width = 0.25
	
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.bar(x - width, mean_oib, width, label='OIB', color='green', alpha=0.7)
	ax.bar(x, mean_cryo, width, label='CryoSat-2', color='blue', alpha=0.7)
	ax.bar(x + width, mean_smos, width, label='SMOS', color='red', alpha=0.7)
	
	ax.set_xlabel('Sea Ice Thickness Bins [m]')
	ax.set_ylabel('Mean Sea Ice Thickness [m]')
	ax.set_title('Comparison of Mean Sea Ice Thickness (Interpolated)')
	ax.set_xticks(x)
	ax.set_xticklabels(bin_labels)
	ax.legend()
	ax.grid(True, linestyle='--', alpha=0.5)
	
	plt.show()

	



#plot_cryo_oib(oib_lat, oib_lon, oib_sit, cryo_lat, cryo_lon, cryo_sit)
#plot_smos_oib(oib_lat, oib_lon, oib_sit, smos_lat, smos_lon, smos_sit[0,:,:])
#hist_cryo_smos_oib(cryo_sit, smos_sit[0,:,:], oib_sit)
""" --------------------------------------------------------------------------------- """
#barplot_mean_thickness(oib_lat, oib_lon, oib_sit, cryo_lat, cryo_lon, cryo_sit, smos_lat, smos_lon, smos_sit[0,:,:])
""" --------------------------------------------------------------------------------- """
barplot_interpolated_sit(oib_lat, oib_lon, oib_sit, cryo_lat, cryo_lon, cryo_sit, smos_lat, smos_lon, smos_sit[0,:,:])