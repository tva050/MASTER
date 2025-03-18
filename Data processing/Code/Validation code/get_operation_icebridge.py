import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from  scipy.interpolate import griddata
import pandas as pd
from pyproj import Proj, transform

oib_paths = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130321.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130322.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130323.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130324.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130326.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130425.txt"
]

smos_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2013\2013_mean_thickness.nc"

cryo_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2013\uit_cryosat2_L3_EASE2_nh25km_2013_03_v3.nc"


def get_data(path):
	# Read the CSV file while automatically handling headers and mixed data types
	df = pd.read_csv(path)

	# Extract only the numerical columns we need
	lat = df["lat"].astype(float).values
	lon = df["lon"].astype(float).values
	thickness = df["thickness"].astype(float).values

	mask = (thickness != -99999.) & (thickness != 0.0)
	thickness = np.where(mask, thickness, np.nan)
	return lat, lon, thickness


def get_smos(path):
	data = nc.Dataset(path)
	#print(data.variables.keys())
	lat = data.variables['latitude'][:]
	lon = data.variables['longitude'][:]
	si_thickness = data.variables['sea_ice_thickness'][:]
	print(si_thickness.shape)
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

	# Convert coordinates
	x, y = transform(wgs84, polar_stereo, lon, lat)
	return x, y

def extract_all_oib(oib_paths):
	all_x, all_y, all_thickness = [], [], []
	for path in oib_paths:
		lat, lon, thickness = get_data(path)
		x, y = latlon_to_polar(lat, lon)

		all_x.extend(x)
		all_y.extend(y)
		all_thickness.extend(thickness)
	return all_x, all_y, all_thickness

def load_oib_lat_lon(file_paths):
    """Loads lat/lon coordinates from OIB text files and stacks them."""
    lat_list, lon_list = [], []

    for file in file_paths:
        try:
            # Skip the header and read lat/lon columns
            data = np.genfromtxt(file, delimiter=",", skip_header=1, usecols=(0, 1), dtype=float)
            
            # Check if data is empty
            if data.size == 0:
                print(f"Warning: {file} is empty or contains only headers.")
                continue
            
            lat_list.append(data[:, 0])  # Latitude
            lon_list.append(data[:, 1])  # Longitude

        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not lat_list or not lon_list:
        raise ValueError("No valid data was loaded from the files.")

    # Concatenate all data points into single arrays
    target_lat = np.concatenate(lat_list)
    target_lon = np.concatenate(lon_list)

    return target_lat, target_lon

def interpolate_data(source_lat, source_lon, source_si, target_lat, target_lon):
	"""Interpolates source_si to target_lat/lon grid using SciPy's griddata."""
	source_points = np.array([source_lat.flatten(), source_lon.flatten()]).T
	source_values = source_si.flatten()

	target_points = np.array([target_lat.flatten(), target_lon.flatten()]).T

	source_interp = griddata(source_points, source_values, target_points, method='linear')

	return source_interp

lat_oib, lon_oib, thickness_oib = extract_all_oib(oib_paths)
lat_smos, lon_smos, thickness_smos = get_smos(smos_path)
lat_cryo, lon_cryo, thickness_cryo = get_cryo(cryo_path)
print(thickness_smos.shape)
stacked_lon_oib, stacked_lat_oib = load_oib_lat_lon(oib_paths)

smos_interp = interpolate_data(lat_smos, lon_smos, thickness_smos, stacked_lat_oib, stacked_lon_oib)
cryo_interp = interpolate_data(lat_cryo, lon_cryo, thickness_cryo, stacked_lat_oib, stacked_lon_oib)
print(smos_interp.shape)
print(cryo_interp.shape)
def plot_all_data(paths):
	"""Plot sea ice thickness from multiple files on the same map."""
	all_x, all_y, all_thickness = [], [], []

	# Load and combine data from all files
	for path in paths:
		lat, lon, thickness = get_data(path)
		x, y = latlon_to_polar(lat, lon)

		all_x.extend(x)
		all_y.extend(y)
		all_thickness.extend(thickness)

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
 	
	
def smos_OIB(OIB_path, smos_path):
	all_x, all_y, all_thickness = [], [], []
	
	for path in OIB_path:
		lat, lon, thickness = get_data(path)
		x, y = latlon_to_polar(lat, lon)

		all_x.extend(x)
		all_y.extend(y)
		all_thickness.extend(thickness)
	
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

def cryo_OIB(OIB_path, cryo_path):
	all_x, all_y, all_thickness = [], [], []
	
	for path in OIB_path:
		lat, lon, thickness = get_data(path)
		x, y = latlon_to_polar(lat, lon)

		all_x.extend(x)
		all_y.extend(y)
		all_thickness.extend(thickness)
	
	cryo_lat, cryo_lon, cryo_thickness = get_cryo(cryo_path)
	
	fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(10, 10))
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
	
	scatter = ax.scatter(all_x, all_y, c=all_thickness, cmap='viridis', vmin=0, vmax=1, zorder=1, transform=ccrs.NorthPolarStereo())
	mesh = ax.pcolormesh(cryo_lon, cryo_lat, cryo_thickness, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1, zorder=0)
	#scatter_cryo = ax.scatter(cryo_lon, cryo_lat, c=cryo_thickness, cmap='viridis', vmin=0, vmax=1, zorder=0, transform=ccrs.PlateCarree())
	ax.add_feature(cfeature.LAND, color='lightgray', alpha=1, zorder=2)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5, zorder=4)
	
	cbar = plt.colorbar(scatter, orientation='vertical')
	cbar.set_label('Sea Ice Thickness (m)')
	
	plt.title('Sea Ice Thickness (OIB and CryoSat-2)')
	plt.show()
	
def mean_diff(oib_paths, smos_path, cryo_path):
	# Load OIB data
	all_x, all_y, all_thickness = [], [], []
	for path in oib_paths:
		lat, lon, thickness = get_data(path)
		x, y = latlon_to_polar(lat, lon)
		all_x.extend(x)
		all_y.extend(y)
		all_thickness.extend(thickness)
	
	oib_points = np.array(list(zip(all_x, all_y)))
	oib_thickness = np.array(all_thickness)
	
	# Load SMOS data
	smos_lat, smos_lon, smos_thickness = get_smos(smos_path)
	smos_x, smos_y = latlon_to_polar(smos_lat, smos_lon)
	smos_points = np.array(list(zip(smos_x.flatten(), smos_y.flatten())))
	smos_thickness = smos_thickness.flatten()
	
	# Load CryoSat-2 data
	cryo_lat, cryo_lon, cryo_thickness = get_cryo(cryo_path)
	cryo_x, cryo_y = latlon_to_polar(cryo_lat, cryo_lon)
	cryo_points = np.array(list(zip(cryo_x.flatten(), cryo_y.flatten())))
	cryo_thickness = cryo_thickness.flatten()
	
	# Find nearest SMOS points to OIB points
	smos_tree = cKDTree(smos_points)
	smos_distances, smos_indices = smos_tree.query(oib_points)
	nearest_smos_thickness = smos_thickness[smos_indices]
	
	# Find nearest CryoSat-2 points to OIB points
	cryo_tree = cKDTree(cryo_points)
	cryo_distances, cryo_indices = cryo_tree.query(oib_points)
	nearest_cryo_thickness = cryo_thickness[cryo_indices]
	
	# Calculate mean differences
	mean_diff_oib_smos = np.nanmean(oib_thickness - nearest_smos_thickness)
	mean_diff_oib_cryo = np.nanmean(oib_thickness - nearest_cryo_thickness)
	
	print(f"Mean difference between OIB and SMOS: {mean_diff_oib_smos:.4f} m")
	print(f"Mean difference between OIB and CryoSat-2: {mean_diff_oib_cryo:.4f} m")


def pair_scatter_plot(oib_paths, smos_path, cryo_path):
	""" 
	Pair scatter plot, i am uncertain if this is the correct way to do it.
	I am also unsure if this is displayed correctly.
	"""
	all_x, all_y, all_thickness = [], [], []
	for path in oib_paths:
		lat, lon, thickness = get_data(path)
		x, y = latlon_to_polar(lat, lon)
		all_x.extend(x)
		all_y.extend(y)
		all_thickness.extend(thickness)
	
	oib_points = np.array(list(zip(all_x, all_y)))
	oib_thickness = np.array(all_thickness)

	# Load SMOS data
	smos_lat, smos_lon, smos_thickness = get_smos(smos_path)
	smos_x, smos_y = latlon_to_polar(smos_lat, smos_lon)
	smos_points = np.array(list(zip(smos_x.flatten(), smos_y.flatten())))
	smos_thickness = smos_thickness.flatten()

	# Load CryoSat-2 data
	cryo_lat, cryo_lon, cryo_thickness = get_cryo(cryo_path)
	cryo_x, cryo_y = latlon_to_polar(cryo_lat, cryo_lon)
	cryo_points = np.array(list(zip(cryo_x.flatten(), cryo_y.flatten())))
	cryo_thickness = cryo_thickness.flatten()

	# Find nearest SMOS points to OIB points
	smos_tree = cKDTree(smos_points)
	_, smos_indices = smos_tree.query(oib_points)
	nearest_smos_thickness = smos_thickness[smos_indices]

	# Find nearest CryoSat-2 points to OIB points
	cryo_tree = cKDTree(cryo_points)
	_, cryo_indices = cryo_tree.query(oib_points)
	nearest_cryo_thickness = cryo_thickness[cryo_indices]

	# Remove NaN values
	valid_smos = ~np.isnan(oib_thickness) & ~np.isnan(nearest_smos_thickness)
	valid_cryo = ~np.isnan(oib_thickness) & ~np.isnan(nearest_cryo_thickness)

	# Define colormap
	cmap = "plasma"  # You can change to 'viridis', 'inferno', etc.

	# Density plot for OIB vs. SMOS
	plt.figure(figsize=(8, 6))
	plt.hexbin(nearest_smos_thickness[valid_smos], oib_thickness[valid_smos], gridsize=100, cmap=cmap)
	plt.colorbar(label='Point Density')
	plt.xlabel("SMOS Sea Ice Thickness (m)")
	plt.ylabel("OIB Sea Ice Thickness (m)")
	plt.title("OIB vs. SMOS Sea Ice Thickness Density")
	plt.xlim(0, 6)
	plt.grid(True)
	plt.show()

	# Density plot for OIB vs. CryoSat-2
	plt.figure(figsize=(8, 6))
	plt.hexbin(nearest_cryo_thickness[valid_cryo], oib_thickness[valid_cryo], gridsize=100, cmap=cmap)
	plt.colorbar(label='Point Density')
	plt.xlabel("Cryosat Sea Ice Thickness (m)")
	plt.ylabel("OIB Sea Ice Thickness (m)")
	plt.title("OIB vs. CryoSat-2 Sea Ice Thickness Density")
	plt.xlim(0, 6)
	plt.grid(True)
	plt.show()
		

def bar_plot(oib_paths, smos_path, cryo_path):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
	
	# Load OIB data
	all_x, all_y, all_thickness = [], [], []
	for path in oib_paths:
		lat, lon, thickness = get_data(path)
		x, y = latlon_to_polar(lat, lon)
		all_x.extend(x)
		all_y.extend(y)
		all_thickness.extend(thickness)
	
	oib_points = np.array(list(zip(all_x, all_y)))
	oib_thickness = np.array(all_thickness)
	
	# Load SMOS data
	smos_lat, smos_lon, smos_thickness = get_smos(smos_path)
	smos_x, smos_y = latlon_to_polar(smos_lat, smos_lon)
	smos_points = np.array(list(zip(smos_x.flatten(), smos_y.flatten())))
	smos_thickness = smos_thickness.flatten()
	
	# Load CryoSat-2 data
	cryo_lat, cryo_lon, cryo_thickness = get_cryo(cryo_path)
	cryo_x, cryo_y = latlon_to_polar(cryo_lat, cryo_lon)
	cryo_points = np.array(list(zip(cryo_x.flatten(), cryo_y.flatten())))
	cryo_thickness = cryo_thickness.flatten()
	
	# Find nearest SMOS points to OIB points
	smos_tree = cKDTree(smos_points)
	smos_distances, smos_indices = smos_tree.query(oib_points)
	nearest_smos_thickness = smos_thickness[smos_indices]
	
	# Find nearest CryoSat-2 points to OIB points
	cryo_tree = cKDTree(cryo_points)
	cryo_distances, cryo_indices = cryo_tree.query(oib_points)
	nearest_cryo_thickness = cryo_thickness[cryo_indices]
	
	# Initialize lists for mean thickness in each bin
	mean_smos = []
	mean_cryo = []
	
	# Compute mean thickness for each bin
	for i in range(len(bins) - 1):
		oib_mask = (oib_thickness >= bins[i]) & (oib_thickness < bins[i + 1])
		
		if np.any(oib_mask):
			mean_smos.append(np.nanmean(nearest_smos_thickness[oib_mask]))
			mean_cryo.append(np.nanmean(nearest_cryo_thickness[oib_mask]))
		else:
			mean_smos.append(np.nan)
			mean_cryo.append(np.nan)
	
	# Plot bar chart
	bar_width = 0.35
	x = np.arange(len(bin_labels))
	
	plt.figure(figsize=(10, 6))
	plt.bar(x - bar_width/2, mean_smos, width=bar_width, label='SMOS', color='blue')
	plt.bar(x + bar_width/2, mean_cryo, width=bar_width, label='CryoSat-2', color='green')
	
	# Labels and formatting
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
	
def multiple_box_plot_oib(oib_sit, smos_interp, cryo_interp):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']

	oib_sit = np.array(oib_sit)
	smos_interp = np.array(smos_interp)
	cryo_interp = np.array(cryo_interp) 

	# Flatten arrays
	oib_flat = oib_sit.flatten()
	smos_flat = smos_interp.flatten()
	cryo_flat = cryo_interp.flatten()

	# Mask NaN values
	valid_mask = ~np.isnan(oib_flat) & ~np.isnan(smos_flat) & ~np.isnan(cryo_flat)
	oib_flat, smos_flat, cryo_flat = oib_flat[valid_mask], smos_flat[valid_mask], cryo_flat[valid_mask]

	# Bin the data
	binned_oib_smos_data, binned_oib_cryo_data = [], []

	for i in range(len(bins) - 1):
		mask = (oib_flat >= bins[i]) & (oib_flat < bins[i + 1]) 
		binned_oib_smos_data.append(smos_flat[mask])
		binned_oib_cryo_data.append(cryo_flat[mask])

	# Create subplots (1 row, 2 columns)
	fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

	# Boxplots
	axes[0].boxplot(binned_oib_smos_data, labels=bin_labels, medianprops=dict(color='black'))
	axes[0].set_title("OIB vs SMOS")
	axes[0].set_xlabel("OIB SIT [m]")
	axes[0].set_ylabel("SMOS SIT [m]")

	axes[1].boxplot(binned_oib_cryo_data, labels=bin_labels, medianprops=dict(color='black'))
	axes[1].set_title("OIB vs CryoSat-2")
	axes[1].set_xlabel("OIB SIT [m]")
	axes[1].set_ylabel("CryoSat-2 SIT [m]")

	# Scatter plots
	for j in range(len(bins) - 1):
		x_positions_smos = np.random.normal(j + 1, 0.05, size=len(binned_oib_smos_data[j]))
		axes[0].scatter(x_positions_smos, binned_oib_smos_data[j], alpha=0.4, color='salmon', s=10)

		x_positions_cryo = np.random.normal(j + 1, 0.05, size=len(binned_oib_cryo_data[j]))
		axes[1].scatter(x_positions_cryo, binned_oib_cryo_data[j], alpha=0.4, color='teal', s=10)

	for ax in axes:
		ax.yaxis.set_tick_params(labelleft=True)

	plt.tight_layout()
	plt.show()

# Load data and plot
#plot_all_data(paths)
#lat, lon, thickness = get_data(oib_paths[0])


#smos_OIB(paths, smos_path)
#cryo_OIB(oib_paths, cryo_path)
#difference_calc(cryo_path, smos_path, paths)
#mean_diff(oib_paths, smos_path, cryo_path)
#pair_scatter_plot(oib_paths, smos_path, cryo_path)
#bar_plot(oib_paths, smos_path, cryo_path)
multiple_box_plot_oib(thickness_oib, smos_interp, cryo_interp)