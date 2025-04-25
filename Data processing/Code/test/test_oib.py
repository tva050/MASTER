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

smos_2017_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\SMOS_monthly\SMOS_monthly_Icethickness_north_201703.nc"
smos_2015_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\SMOS_monthly\SMOS_monthly_Icethickness_north_201503.nc"
smos_2014_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\SMOS_monthly\SMOS_monthly_Icethickness_north_201403.nc"
smos_2013_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\SMOS_monthly\SMOS_monthly_Icethickness_north_201303.nc"
smos_2012_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\SMOS_monthly\SMOS_monthly_Icethickness_north_201203.nc"
smos_2011_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\SMOS_monthly\SMOS_monthly_Icethickness_north_201103.nc"

uit_2017_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2017_03_v3.nc"
uit_2015_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2015_03_v3.nc"
uit_2014_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2014_03_v3.nc"
uit_2013_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2013_03_v3.nc"
uit_2012_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2012_03_v3.nc"
uit_2011_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2011_03_v3.nc"

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
	
