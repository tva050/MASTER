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
	
	plt.title('Sea Ice Thickness RAW')
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
	cbar.set_label('Sea Ice Thickness [m]')
	
	plt.title('OiB and CS Sea Ice Thickness RAW')
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
	
	plt.title('OiB and SMOS Sea Ice Thickness RAW')
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
	plt.title("Histogram: Comparison of CryoSat-2 and SMOS Thickness with OIB RAW")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()

# ----------------------- Product processing ----------------------- #



#plot_cryo_oib(oib_lat, oib_lon, oib_sit, cryo_lat, cryo_lon, cryo_sit)
#plot_smos_oib(oib_lat, oib_lon, oib_sit, smos_lat, smos_lon, smos_sit[0,:,:])
