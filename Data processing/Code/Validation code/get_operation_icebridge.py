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
	# print(data.variables.keys())
	
	lat = data.variables['latitude'][:]
	lon = data.variables['longitude'][:]
	si_thickness = data.variables['sea_ice_thickness'][:]
	
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

# Load data and plot
#plot_all_data(paths)
lat_smos, lon_smos, thickness_smos = get_smos(smos_path)
lat_cryo, lon_cryo, thickness_cryo = get_cryo(cryo_path)

#smos_OIB(paths, smos_path)
#cryo_OIB(oib_paths, cryo_path)
#difference_calc(cryo_path, smos_path, paths)
mean_diff(oib_paths, smos_path, cryo_path)