import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from  scipy.interpolate import griddata
import pandas as pd
from pyproj import Proj, Transformer
import seaborn as sns

#oib_paths = [
#	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110316.txt",
#	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110317.txt",
#	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110318.txt",
#	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110322.txt",
#	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110323.txt",
#	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110325.txt",
#	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110326.txt",
#	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2011\IDCSI4_20110328.txt"
#]

oib_paths = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130321.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130322.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130323.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130324.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130425.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\2013\IDCSI4_20130326.txt"
]

smos_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2013\2013_mean_thickness.nc"

cryo_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2013\uit_cryosat2_L3_EASE2_nh25km_2013_03_v3.nc"


# -------------------------- Data Processing -------------------------- #


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

def latlon_to_polar(lat, lon):
	"""Convert lat/lon (degrees) to North Polar Stereographic (meters)."""
	# Define WGS84 (lat/lon) and Polar Stereographic projection 
	wgs84 = Proj(proj="latlong", datum="WGS84")
	polar_stereo = Proj(proj="stere", lat_0=90, lon_0=-45, datum="WGS84", k=1, x_0=0, y_0=0)

	# Create a Transformer object
	transformer = Transformer.from_proj(wgs84, polar_stereo, always_xy=True)

	# Convert coordinates
	x, y = transformer.transform(lon, lat)
	return x, y # x is longitude, y is latitude

def extract_all_oib(oib_paths):
	all_x, all_y, all_thickness = [], [], []
	for path in oib_paths:
		lat, lon, thickness = get_data(path)
		x, y = latlon_to_polar(lat, lon)

		all_x.extend(x)
		all_y.extend(y)
		all_thickness.extend(thickness)
	return all_x, all_y, all_thickness 

all_x, all_y, all_thickness = extract_all_oib(oib_paths)
all_x, all_y, all_thickness = np.array(all_x), np.array(all_y), np.array(all_thickness)

smos_lat, smos_lon, smos_thickness = get_smos(smos_path)
cryo_lat, cryo_lon, cryo_thickness = get_cryo(cryo_path)

def pre_interpolation_data(lat, lon, sit):
	x, y = latlon_to_polar(lat, lon)
	x, y, sit = x.flatten(), y.flatten(), sit.flatten()
	
	mask = ~np.isnan(sit)
	x, y, sit = x[mask], y[mask], sit[mask]
	
	return x, y, sit

def interpolate_to_oib_grid(target_x, target_y, source_x, source_y, source_sit):
	""" 
	Interpolates source data to target grid using linear interpolation.
	This returns the interpolated sea ice thickness values, which can be compared to OIB data.
 	"""
	# convert source to polar stereographic
	source_x, source_y, source_sit = pre_interpolation_data(source_x, source_y, source_sit)
	
	interp_sit = griddata((source_x, source_y), source_sit, (target_x, target_y), method='linear')  
	return interp_sit

smos_interp = interpolate_to_oib_grid(all_x, all_y, smos_lat, smos_lon, smos_thickness)
cryo_interp = interpolate_to_oib_grid(all_x, all_y, cryo_lat, cryo_lon, cryo_thickness)


# -------------------------- Data Visualization -------------------------- #

def plot_oib(lat, lon, thickness):
	# Create figure
	fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(10, 10))
	
	# Set map extent (meters, not degrees)
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

	# Scatter plot with all data points
	scatter = ax.scatter(all_x, all_y, c=all_thickness, cmap='viridis', vmin=0, vmax=1, zorder=1, transform=ccrs.NorthPolarStereo())

	# Add map features
	ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=4)

	# Colorbar
	cbar = plt.colorbar(scatter, orientation='vertical')
	cbar.set_label('Sea Ice Thickness (m)')

	plt.title('Sea Ice Thickness (Multiple Paths)')
	plt.show()
	 
def smos_oib(all_x, all_y, all_thickness, smos_path):
	smos_lat, smos_lon, smos_thickness = get_smos(smos_path)
	
	fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(10, 10))
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
	
	scatter = ax.scatter(all_x, all_y, c=all_thickness, cmap='viridis', vmin=0, vmax=1, zorder=1, transform=ccrs.NorthPolarStereo())
	mesh = ax.pcolormesh(smos_lon, smos_lat, smos_thickness[0, :, :], transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=0)
	
	ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=4)
	
	cbar = plt.colorbar(scatter, orientation='vertical')
	cbar.set_label('Sea Ice Thickness (m)')
	
	plt.title('Sea Ice Thickness (OIB and SMOS)')
	plt.show()

def cryo_oib(all_x, all_y, all_thickness, cryo_path):
	cryo_lat, cryo_lon, cryo_thickness = get_cryo(cryo_path)
 
	fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(10, 10))
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
 
	scatter = ax.scatter(all_x, all_y, c=all_thickness, cmap='viridis', vmin=0, vmax=1, zorder=1, transform=ccrs.NorthPolarStereo())
	mesh = ax.pcolormesh(cryo_lon, cryo_lat, cryo_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=0)
 
	ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=4)
 
	cbar = plt.colorbar(scatter, orientation='vertical')
	cbar.set_label('Sea Ice Thickness (m)')
	
	plt.title('Sea Ice Thickness (OIB and CryoSat-2)')
	plt.show()

def map_plot_used_data(all_x, all_y, all_thickness, smos_lat, smos_lon, smos_thickness):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	
	# Flatten and filter OIB data
	oib_x_flat = all_x.flatten()
	oib_y_flat = all_y.flatten()
	oib_thickness_flat = all_thickness.flatten()
	
	valid_oib_mask = ~np.isnan(oib_thickness_flat)
	oib_x_flat, oib_y_flat, oib_thickness_flat = oib_x_flat[valid_oib_mask], oib_y_flat[valid_oib_mask], oib_thickness_flat[valid_oib_mask]
	
	# Flatten and filter CryoSat-2 data
	smos_x, smos_y, smos_thickness_flat = pre_interpolation_data(smos_lat, smos_lon, smos_thickness)
	
	valid_smos_mask = ~np.isnan(smos_thickness_flat)
	smos_x, smos_y, smos_thickness_flat = smos_x[valid_smos_mask], smos_y[valid_smos_mask], smos_thickness_flat[valid_smos_mask]
	
	# Create the map plot
	fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(10, 10))
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

	# Plot OIB data points (scatter plot)
	scatter_oib = ax.scatter(oib_x_flat, oib_y_flat, c=oib_thickness_flat, cmap='viridis', vmin=0, vmax=1, s=10, alpha=0.7, transform=ccrs.NorthPolarStereo(), label="OIB Data", zorder=1)

	# Plot CryoSat-2 data points (scatter plot)
	scatter_cryo = ax.scatter(smos_x, smos_y, c=smos_thickness_flat, cmap='coolwarm', vmin=0, vmax=1, s=10, alpha=0.7, transform=ccrs.NorthPolarStereo(), label="smos Data", zorder=0)

	# Add map features
	ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=4)

	# Add colorbar
	cbar = plt.colorbar(scatter_oib, ax=ax, orientation='vertical', shrink=0.7)
	cbar.set_label('OIB Sea Ice Thickness (m)')

	# Add title and legend
	plt.legend(loc='upper right')
	plt.title('Data Points Used: OIB vs CryoSat-2')

	# Show the plot
	plt.show()
	
	
def plot_interpolated_data(oib_x, oib_y, oib_thickness, interp_x, interp_y, interp_thickness, title="Interpolated Data vs OIB"):
	""" 
	Plots the interpolated thickness data on a polar stereographic map along with OIB thickness data for comparison.
	"""
	fig, ax = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.NorthPolarStereo()})

	# Plot OIB Thickness
	sc1 = ax[0].scatter(oib_x, oib_y, c=oib_thickness, cmap='coolwarm', marker='o', s=10, transform=ccrs.PlateCarree())
	ax[0].set_title("OIB Sea Ice Thickness")
	ax[0].coastlines()
	ax[0].gridlines(draw_labels=True)
	cbar1 = plt.colorbar(sc1, ax=ax[0], orientation='vertical', shrink=0.7)
	cbar1.set_label("Sea Ice Thickness (m)")

	# Plot Interpolated Thickness
	sc2 = ax[1].scatter(interp_x, interp_y, c=interp_thickness, cmap='coolwarm', marker='o', s=10, transform=ccrs.PlateCarree())
	ax[1].set_title(title)
	ax[1].coastlines()
	ax[1].gridlines(draw_labels=True)
	cbar2 = plt.colorbar(sc2, ax=ax[1], orientation='vertical', shrink=0.7)
	cbar2.set_label("Sea Ice Thickness (m)")

	plt.show()
	
def hist_cryo_smos_oib_SIT(cryo_sit, smos_sit, oib_sit):
	""" 
	Creates a histogram to compare CryoSat-2 thickness before and after interpolation.
	"""
	# Flatten and filter CryoSat-2 data
	cryo_sit = cryo_sit.flatten()
	smos_sit = smos_sit.flatten()

 
	
	# Create histogram
	plt.figure(figsize=(8, 6))
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
 
def hist_before_after_interp(before_interp, after_interp, oib_thickness):
	""" 
	Creates a histogram to compare CryoSat-2 thickness before and after interpolation.
	"""
	before_interp = before_interp.flatten()
	after_interp = after_interp.flatten()
 
	# Create histogram
	plt.figure(figsize=(8, 6))
	plt.hist(oib_thickness, bins=100, alpha=0.5, color='green', label='OIB Thickness')
	plt.hist(before_interp, bins=100, alpha=0.5, color='blue', label='CryoSat-2')
	plt.hist(after_interp, bins=100, alpha=0.5, color='red', label='SMOS')
 
	# Labels and title
	plt.xlabel("Sea Ice Thickness (m)")
	plt.ylabel("Frequency")
	plt.title("Histogram: CS and SMOS Thickness Before and After Interpolation")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()

# -------------------------- Data Analysis -------------------------- #

def pair_scatter_plot(all_thickness, smos_interp, bins=100):
	""" 
	Creates a scatter plot comparing OIB and CryoSat-2 thickness.
	Points are plotted in order of density (densest on top).
	"""
	
	# Remove NaN values
	valid_mask = ~np.isnan(all_thickness) & ~np.isnan(smos_interp)
	x = all_thickness[valid_mask]
	y = smos_interp[valid_mask]

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
	


def barplot(cryo_interp, smos_interp, all_thickness):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	#bin_centers = (bins[:-1] + bins[1:]) / 2  # Midpoints for plotting
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	valied_mask = ~np.isnan(all_thickness) & ~np.isnan(cryo_interp) & ~np.isnan(smos_interp)
	all_thickness, cryo_interp, smos_interp = all_thickness[valied_mask], cryo_interp[valied_mask], smos_interp[valied_mask]
	
	mean_cryo = []
	mean_smos = []
	
	for i in range(len(bins)-1):
		mask = (all_thickness >= bins[i]) & (all_thickness < bins[i+1])
		mean_cryo.append(np.nanmean(cryo_interp[mask]))
		mean_smos.append(np.nanmean(smos_interp[mask]))
  
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
 
def boxplot(cryo_interp, smos_interp, all_thickness):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']

	oib_flatten = all_thickness.flatten()
	cryo_flatten = cryo_interp.flatten()	
	smos_flatten = smos_interp.flatten()
 
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
 
def heatmap(cryo_interp, smos_interp, oib_sit):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
	satellite_products = ["CryoSat-2", "SMOS"]
	
	cryo_flat = cryo_interp.flatten()
	smos_flat = smos_interp.flatten()
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
 
def differences(cryo_interp, smos_interp, oib_sit):
	# Define bins and labels
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	# Flatten data
	cryo_flat = cryo_interp.flatten()
	smos_flat = smos_interp.flatten()
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
		   
 
if __name__ == '__main__':
	""" Data Visualization """
	#plot_oib(all_x, all_y, all_thickness)
	smos_oib(all_x, all_y, all_thickness, smos_path)
	#cryo_oib(all_x, all_y, all_thickness, cryo_path)	
	#map_plot_used_data(all_x, all_y, all_thickness, cryo_lat, cryo_lon, cryo_thickness)
	#plot_interpolated_data(all_x, all_y, all_thickness, all_x, all_y, cryo_interp, title="CryoSat-2 Interpolated vs OIB")
	#hist_cryo_smos_oib_SIT(cryo_thickness, smos_thickness[0, :, :], all_thickness)
	#hist_before_after_interp(cryo_interp, smos_interp, all_thickness)
	""" Data Analysis """
	#pair_scatter_plot(all_thickness, smos_interp)
	#barplot(cryo_interp, smos_interp, all_thickness)
	#boxplot(cryo_interp, smos_interp, all_thickness)
	#heatmap(cryo_interp, smos_interp, all_thickness)
	#differences(cryo_interp, smos_interp, all_thickness)