import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
import pandas as pd
import cartopy.crs as ccrs
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import matplotlib.path as mpath


import warnings
warnings.filterwarnings("ignore", message="facecolor will have no effect*")

oib_paths_2017 = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2017\IDCSI4_20170309.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2017\IDCSI4_20170310.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2017\IDCSI4_20170311.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2017\IDCSI4_20170312.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2017\IDCSI4_20170314.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2017\IDCSI4_20170320.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2017\IDCSI4_20170323.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2017\IDCSI4_20170324.txt",
]
	

oib_paths_2015 = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2015\OIB_20150319_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2015\OIB_20150324_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2015\OIB_20150325_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2015\OIB_20150326_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2015\OIB_20150327_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2015\OIB_20150329_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2015\OIB_20150330_IDCSI2.txt",
]

oib_paths_2014 = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140312_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140313_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140314_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140315_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140317_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140318_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140319_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140321_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140324_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140325_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140326_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140328_IDCSI2.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2014\OIB_20140331_IDCSI2.txt", 
]

oib_paths_2013 = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130321.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130322.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130323.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130324.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130326.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130327.txt"
]

oib_paths_2012 = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120314.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120315.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120316.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120317.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120319.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120322.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120323.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120326.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120327.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120328.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2012\IDCSI4_20120329.txt",
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

smos_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2013\2013_march_mean_thickness.nc"

cryo_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2013_03_v3.nc"


# ------------------------------ Data Extraction ------------------------------

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

# ------------------------------ Data processing ------------------------------

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

def resample_to_cryo_grid(x_source, y_source, source_sit, source_sit_un, x_target, y_target, radius=12500):
	""" 
	Resample source data onto a new grid using weighted average based on uncertainty.
	If no valid data, leave as NaN.
	If no points are found in the radius, set to NaN.
	"""
	target_tree = cKDTree(np.column_stack([x_target.ravel(), y_target.ravel()]))  # Tree for faster lookup

	# Initialize arrays to store resampled data and weight sums
	resampled_sit = np.full(x_target.shape, np.nan)
	weights_sum = np.zeros(x_target.shape)

	# Flatten all arrays for iteration
	source_sit = source_sit.ravel()
	source_sit_un = source_sit_un.ravel()
	x_source = x_source.ravel()
	y_source = y_source.ravel()

	for i in range(len(source_sit)):
		if np.isnan(source_sit[i]) or np.isnan(source_sit_un[i]):
			continue  # Skip invalid data points
		
		indices = target_tree.query_ball_point([x_source[i], y_source[i]], radius)

		if not indices:
			continue  # If no neighbors are found, skip

		weight = 1 / source_sit_un[i] if source_sit_un[i] != 0 else 0  # Avoid division by zero

		for idx in indices:
			if idx >= x_target.size:  # Ensure valid index
				continue
			
			row, col = np.unravel_index(idx, x_target.shape)

			# Avoid operations on uninitialized values
			if np.isnan(resampled_sit[row, col]):
				resampled_sit[row, col] = 0

			# Weighted sum calculation
			resampled_sit[row, col] = (resampled_sit[row, col] * weights_sum[row, col] + source_sit[i] * weight) / (weights_sum[row, col] + weight)
			weights_sum[row, col] += weight

	# Mask invalid values (no data points found)
	resampled_sit[weights_sum == 0] = np.nan

	return np.ma.masked_invalid(resampled_sit)


resampled_oib_sit = resample_to_cryo_grid(x_oib, y_oib, oib_sit, oib_sit_un, x_cryo, y_cryo)
resampled_smos_sit = resample_to_cryo_grid(x_smos, y_smos, smos_sit, smos_sit_un, x_cryo, y_cryo)
print('Resampled OIB:', resampled_oib_sit.shape)
print('Resampled SMOS:', resampled_smos_sit.shape)


def hist_cryo_smos_oib(cryo_sit, oib_sit, smos_sit):
	cryo_sit = cryo_sit.flatten()
	oib_sit = oib_sit.flatten()
	smos_sit = smos_sit.flatten()
	
	# Create histogram
	plt.hist(oib_sit, bins=100, alpha=0.5, color='green', label='OIB', density=True)
	plt.hist(cryo_sit, bins=100, alpha=0.5, color='blue', label='OiB', density=True)
	plt.hist(smos_sit, bins=100, alpha=0.5, color='red', label='SMOS', density=True)
	
	# Labels and title
	plt.xlabel("Sea Ice Thickness [m]")
	plt.ylabel("PDF")
	plt.title("Histogram: After Resampling")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()


def plot_fligth_paths(oib_2011, oib_2012, oib_2013):
	oib_2011_lat, oib_2011_lon, oib_2011_sit, oib_2011_sit_un = extract_all_oib(oib_2011)
	oib_2012_lat, oib_2012_lon, oib_2012_sit, oib_2012_sit_un = extract_all_oib(oib_2012)
	oib_2013_lat, oib_2013_lon, oib_2013_sit, oib_2013_sit_un = extract_all_oib(oib_2013)
 
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
 
	ax.coastlines()
	ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=2)
	ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.RIVERS, edgecolor='lightgray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=4)
	ax.add_feature(cfeature.COASTLINE, color = "black", linewidth=0.1, zorder=5)
	ax.gridlines(draw_labels=True, color="dimgray", zorder = 7)
 
	sc1 = ax.scatter(oib_2011_lon, oib_2011_lat, c='#72259b', s=1, label='OIB 2011', transform=ccrs.PlateCarree(), zorder = 6)
	sc2 = ax.scatter(oib_2012_lon, oib_2012_lat, c='#2f89c5', s=1, label='OIB 2012', transform=ccrs.PlateCarree(), zorder = 6)
	sc3 = ax.scatter(oib_2013_lon, oib_2013_lat, c='#41d03b', s=1, label='OIB 2013', transform=ccrs.PlateCarree(), zorder = 6)
 
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
 
	plt.legend(markerscale = 5)
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
	
def plot_cryo_oib(oib_lat, oib_lon, oib_sit, cs_lat, cs_lon, cs_sit, title):
	# Convert lon and lat to NumPy arrays if they are lists
	oib_lon = np.array(oib_lon)
	oib_lat = np.array(oib_lat)

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
	cbar.set_label('Sea Ice Thickness [m]')
	
	plt.title(title)
	plt.show()


# ------------------------------ Data visualization ------------------------------


def pair_scatter_plot(oib_sit, smos_sit, bins=100):
	""" 
	Creates a scatter plot comparing OIB and CryoSat-2 thickness.
	Points are plotted in order of density (densest on top).
	"""
	
	# Remove NaN values
	valid_mask = ~np.isnan(oib_sit) & ~np.isnan(smos_sit)
	x = oib_sit[valid_mask]
	y = smos_sit[valid_mask]

	# Create 2D histogram to count data density
	hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

	# Find the bin index for each point
	x_idx = np.searchsorted(xedges, x) - 1
	y_idx = np.searchsorted(yedges, y) - 1

	# Ensure indices are within valid range
	x_idx = np.clip(x_idx, 0, hist.shape[0] - 1) 
	y_idx = np.clip(y_idx, 0, hist.shape[1] - 1)

	# Get density values for each point
	density = hist[x_idx, y_idx]

	# Sort points by density (lower density plotted first)
	sort_idx = np.argsort(density)
	x, y, density = x[sort_idx], y[sort_idx], density[sort_idx]

	# Create scatter plot
	plt.figure(figsize=(8, 6))
	sc = plt.scatter(x, y, c=density, cmap='plasma', alpha=0.7, s=10)

	# Colorbar to show density scale
	cbar = plt.colorbar(sc)
	cbar.set_label('Point Density (2D Histogram)')

	# Labels and title
	plt.xlabel("OIB Sea Ice Thickness (m)")
	plt.ylabel("SMOS Sea Ice Thickness (m)")
	plt.title("Pair Scatter Plot: OIB vs SMOS Thickness (Density Colored)")
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()

def scatter_oib_cryo_pair(x_cryo, y_cryo, cryo_sit, oib_sit):
	"""
	Creates a pair scatter plot with OIB SIT on the x-axis and CryoSat-2 SIT on the y-axis
	for CryoSat-2 grid points where CryoSat-2 thickness is between 0 and 1 meter.
	
	Parameters:
	- x_cryo, y_cryo: CryoSat-2 projected x, y grid coordinates.
	- cryo_sit: CryoSat-2 SIT values.
	- oib_sit: OIB SIT values, already resampled to the CryoSat-2 grid.
	"""
	# Step 1: Find CryoSat-2 points where thickness is between 0 and 1 meter
	mask_cryo = (cryo_sit >= 0) & (cryo_sit <= 1) & ~np.isnan(cryo_sit)
	
	# Step 2: Extract corresponding CryoSat-2 coordinates and thickness values
	cryo_x_filtered = x_cryo[mask_cryo]
	cryo_y_filtered = y_cryo[mask_cryo]
	cryo_sit_filtered = cryo_sit[mask_cryo]
	
	# Step 3: Extract corresponding OIB thickness values at the filtered CryoSat-2 coordinates
	oib_sit_filtered = oib_sit[mask_cryo]
	
	# Step 4: Plot the pair scatter plot
	plt.figure(figsize=(8, 6))
	plt.scatter(oib_sit_filtered, cryo_sit_filtered, alpha=0.5, c='b', edgecolors='k')
	plt.xlabel("OIB Sea Ice Thickness [m]")
	plt.ylabel("CryoSat-2 Sea Ice Thickness [m]")
	plt.title("OIB vs CryoSat-2 Thickness (0 to 1m range)")
	plt.grid(True)
	plt.show()


def boxplot(cryo_sit, smos_sit, oib_sit):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']

	oib_flatten = oib_sit.flatten()
	cryo_flatten = cryo_sit.flatten()	
	smos_flatten = smos_sit.flatten()
 
	valid_mask = ~np.isnan(oib_flatten) & ~np.isnan(cryo_flatten) & ~np.isnan(smos_flatten)
	oib_flatten, cryo_flatten, smos_flatten = oib_flatten[valid_mask], cryo_flatten[valid_mask], smos_flatten[valid_mask]
 
	binned_oib_cryo_data, binned_oib_smos_data = [], []
	
	for i in range(len(bins) - 1):
		mask = (oib_flatten >= bins[i]) & (oib_flatten < bins[i + 1])
		binned_oib_cryo_data.append(cryo_flatten[mask])
		binned_oib_smos_data.append(smos_flatten[mask])
  
	fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
	
	axes[0].boxplot(binned_oib_smos_data, labels=bin_labels, medianprops=dict(color='black'))
	axes[0].set_title("OIB vs SMOS")
	axes[0].set_xlabel("OIB SIT [m]")
	axes[0].set_ylabel("SMOS SIT [m]")

	axes[1].boxplot(binned_oib_cryo_data, labels=bin_labels, medianprops=dict(color='black'))
	axes[1].set_title("OIB vs CryoSat-2")
	axes[1].set_xlabel("OIB SIT [m]")
	axes[1].set_ylabel("CryoSat-2 SIT [m]")
 
	for j in range(len(bins) - 1):
		x_positions_smos = np.random.normal(j + 1, 0.05, size=len(binned_oib_smos_data[j]))
		axes[0].scatter(x_positions_smos, binned_oib_smos_data[j], alpha=0.4, color='salmon', s=10)
 
		x_positions_cryo = np.random.normal(j + 1, 0.05, size=len(binned_oib_cryo_data[j]))
		axes[1].scatter(x_positions_cryo, binned_oib_cryo_data[j], alpha=0.4, color='teal', s=10)
  
	for ax in axes:
		ax.yaxis.set_tick_params(labelleft=True)

	plt.tight_layout()
	plt.show()

def barplot(cryo_sit, smos_sit, oib_sit):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	#bin_centers = (bins[:-1] + bins[1:]) / 2  # Midpoints for plotting
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	valied_mask = ~np.isnan(oib_sit) & ~np.isnan(cryo_sit) & ~np.isnan(smos_sit)
	oib_sit, cryo_sit, smos_sit = oib_sit[valied_mask], cryo_sit[valied_mask], smos_sit[valied_mask]
	
	mean_cryo = []
	mean_smos = []
	
	for i in range(len(bins)-1):
		mask = (oib_sit >= bins[i]) & (oib_sit < bins[i+1])
		mean_cryo.append(np.nanmean(cryo_sit[mask]))
		mean_smos.append(np.nanmean(smos_sit[mask]))
  
	bar_width = 0.35
	x = np.arange(len(bin_labels))
 
	plt.figure()
	plt.bar(x - bar_width/2, mean_smos, width=bar_width, label='SMOS', color='blue')
	plt.bar(x + bar_width/2, mean_cryo, width=bar_width, label='CryoSat-2', color='green')
	
	plt.xlabel("OIB Sea Ice Thickness Bins [m]")
	plt.ylabel("Mean Sea Ice Thickness [m]")
	plt.title("Mean Sea Ice Thickness by OIB Bins")
	plt.xticks(x, bin_labels)
	plt.xlim(-0.5, len(bin_labels) - 0.5)
	plt.legend()
	plt.grid(axis='y', linestyle='--', alpha=0.5)
	# Show plot
	plt.tight_layout()
	plt.show()
 
def heatmap(cryo_sit, smos_sit, oib_sit):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
	satellite_products = ["CryoSat-2", "SMOS"]
	
	cryo_flat = cryo_sit.flatten()
	smos_flat = smos_sit.flatten()
	oib_flat = oib_sit.flatten()
	
	valid_mask = ~np.isnan(oib_flat) & ~np.isnan(cryo_flat) & ~np.isnan(smos_flat)
	oib_flat, cryo_flat, smos_flat = oib_flat[valid_mask], cryo_flat[valid_mask], smos_flat[valid_mask]
	
	mean_difference = np.zeros((len(satellite_products), len(bins) - 1))
	
	for i in range(len(bins) - 1):
		mask = (oib_flat >= bins[i]) & (oib_flat < bins[i + 1])
		mean_difference[0, i] = np.nanmean(cryo_flat[mask] - oib_flat[mask])
		mean_difference[1, i] = np.nanmean(smos_flat[mask] - oib_flat[mask])

	plt.figure(figsize=(10, 6))
	ax = sns.heatmap(mean_difference, annot=True, fmt=".2f", cmap='plasma', xticklabels=bin_labels, yticklabels=satellite_products, center=0)
	plt.xlabel("OIB Sea Ice Thickness Bins [m]")
	plt.ylabel("Satellite Product")
	plt.title("Mean Difference from OIB Thickness")
	plt.show()
 
def differences(cryo_sit, smos_sit, oib_sit):
	# Define bins and labels
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	# Flatten data
	cryo_flat = cryo_sit.flatten()
	smos_flat = smos_sit.flatten()
	oib_flat = oib_sit.flatten()
 
	valid_mask = ~np.isnan(oib_flat) & ~np.isnan(cryo_flat) & ~np.isnan(smos_flat)
	oib_flat, cryo_flat, smos_flat = oib_flat[valid_mask], cryo_flat[valid_mask], smos_flat[valid_mask]
 
	mean_difference_cryo = []
	mean_difference_smos = []
 
	corr_cryo = []
	corr_smos = []
 
	rmse_cryo = []
	rmse_smos = []
 
	for i in range(len(bins) - 1):
		mask = (oib_flat >= bins[i]) & (oib_flat < bins[i + 1])
		oib_bin = oib_flat[mask]
		cryo_bin = cryo_flat[mask]
		smos_bin = smos_flat[mask]
  
		oib_mean = np.nanmean(oib_flat[mask])

		mean_difference_cryo.append(np.nanmean(cryo_flat[mask]) - oib_mean)
		mean_difference_smos.append(np.nanmean(smos_flat[mask]) - oib_mean)
  
		corr_cryo.append(np.corrcoef(oib_bin, cryo_bin)[0, 1])
		corr_smos.append(np.corrcoef(oib_bin, smos_bin)[0, 1])

		rmse_cryo.append(np.sqrt(np.nanmean((cryo_flat[mask] - oib_bin) ** 2)))
		rmse_smos.append(np.sqrt(np.nanmean((smos_flat[mask] - oib_bin) ** 2)))
	
	# print results
	print("-----------------OIB and CryoSat-2-----------------")
	print(f"Mean Difference: {np.round(mean_difference_cryo, 3)}")
	print(f"Correlation:     {np.round(corr_cryo, 3)}")
	print(f"RMSE:            {np.round(rmse_cryo, 3)}")
	print(f"Total Mean: {np.round(np.nanmean(mean_difference_cryo), 3)}")
	print(f"Total RMSE: {np.round(np.nanmean(rmse_cryo), 3)}")
	print("\n------------------OiB and SMOS------------------")
	print(f"Mean Difference: {np.round(mean_difference_smos, 3)}")
	print(f"Correlation: 	 {np.round(corr_smos, 3)}")
	print(f"RMSE:            {np.round(rmse_smos, 3)}")
	print(f"Total Mean: {np.round(np.nanmean(mean_difference_smos), 3)}")
	print(f"Total RMSE: {np.round(np.nanmean(rmse_smos), 3)}")
	print("\n-------------------------------------------------")
	print("[0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1]")
 

if __name__ == "__main__":
	#plot_fligth_paths(oib_paths_2014, oib_paths_2015, oib_paths_2017)	
 
 	#plot_cryo_oib(cryo_lat, cryo_lon, resampled_oib_sit, cryo_lat, cryo_lon, cryo_sit, 'CryoSat-2 vs OiB')
	#plot_cryo_oib(cryo_lat, cryo_lon, resampled_oib_sit, cryo_lat, cryo_lon, resampled_smos_sit, 'SMOS vs OiB') 
	#pair_scatter_plot(resampled_oib_sit, cryo_sit)
	scatter_oib_cryo_pair(x_cryo, y_cryo, cryo_sit, resampled_oib_sit)
	
 	#hist_cryo_smos_oib(cryo_sit, resampled_oib_sit, resampled_smos_sit)
	#boxplot(cryo_sit, resampled_smos_sit, resampled_oib_sit)
	#barplot(cryo_sit, resampled_smos_sit, resampled_oib_sit)
	#heatmap(cryo_sit, resampled_smos_sit, resampled_oib_sit)
	#differences(cryo_sit, resampled_smos_sit, resampled_oib_sit)
 