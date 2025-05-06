import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable



EM_bird_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\EmBird"
ALS_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\als"
ALS_24_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\als\24_1_L3"
ALS_26_1_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\als\26_1_L3"
ALS_26_2_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\als\26_2_L3"

uit_L3 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2014_03_v3.nc"
smos_L3 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\SMOS_monthly\SMOS_monthly_Icethickness_north_201403.nc"

# EM-Bird data sturcture
# 1 Year, 2 Month, 3 Day, 4 Second of the day, 5 Record number, 6 Latitude Decimal degree,
# 7 Longitude Decimal degree, 8 Distance Meter, 9 Total Thickness Meter, 10 Laser Range Meter

def read_EMbird_single(file_path):
	"""
	Read a single SMOSice file and return a DataFrame with the relevant columns.
	"""
	# Read the file
	data = pd.read_csv(file_path, sep="\t", header=None, skiprows=1)

	# Extract relevant columns
	data.columns = ["Year", "Month", "Day", "Second_of_day", "Record_number",
					"Latitude", "Longitude", "Distance", "Total_Thickness", "Laser_Range"]
	
	# Convert to datetime
	data["DateTime"] = pd.to_datetime(data[["Year", "Month", "Day"]])
	
	return data

def read_all_EMbird_files(directory):
	"""
	Read all SMOSice files in a directory and return a combined DataFrame.
	"""
	all_files = glob.glob(os.path.join(directory, "*.txt"))
	all_data = pd.DataFrame()

	for file in all_files:
		data = read_EMbird_single(file)
		all_data = pd.concat([all_data, data], ignore_index=True)

	return all_data

# ALS data (.dat file) structure
# 1 Timestamp UTC time string YYYY-MM-DDTHH:MM:SS.SSS, 2 Number of Samples, 3 Longitude Decimal degrees
# 4 Latitude Decimal degrees, 5 Freeboard Meter, 6 Freeboard Std. Dev. Meter

def read_als_single(file_path):
	try:
		# Read using whitespace as separator
		data = pd.read_csv(file_path, sep=r"\s+", engine="python", header=None, skiprows=1)

		# Check column count
		if data.shape[1] != 6:
			print(f"Unexpected column count in file: {file_path}")
			print(f"Data shape: {data.shape}")
			print(data.head())
			return pd.DataFrame()

		# Assign column names
		data.columns = ["Timestamp", "Number_of_Samples", "Longitude", "Latitude",
						"Freeboard", "Freeboard_Std_Dev"]

		# Convert to datetime
		data["DateTime"] = pd.to_datetime(data["Timestamp"], format="%Y-%m-%dT%H:%M:%S.%f")

		return data

	except Exception as e:
		print(f"Error reading file: {file_path}")
		print(e)
		return pd.DataFrame()

def read_all_als_files(directory):
	"""
	Read all ALS files in three seperate directories and return a combined DataFrame.
	"""
	all_files_24 = glob.glob(os.path.join(directory, "24_1_L3", "*.dat"))
	all_files_26_1 = glob.glob(os.path.join(directory, "26_1_L3", "*.dat"))
	all_files_26_2 = glob.glob(os.path.join(directory, "26_2_L3", "*.dat"))

	all_data = []

	for file in all_files_24 + all_files_26_1 + all_files_26_2:
		data = read_als_single(file)
		data["SourceFile"] = os.path.basename(file)  # Add file name
		all_data.append(data)

	return pd.concat(all_data, ignore_index=True)


def get_UiT(path):
	data = nc.Dataset(path)
	#print(data.variables.keys())
	
	lat = data.variables['latitude'][:]
	lon = data.variables['longitude'][:]
	sit = data.variables['sea_ice_thickness'][:]
	sit_un = data.variables['sea_ice_thickness_uncertainty'][:]
	ifb = data.variables['sea_ice_freeboard'][:]
	
	# Check if lat and lon are 1D and need reshaping
	if lat.ndim == 1 and lon.ndim == 1:
		lon, lat = np.meshgrid(lon, lat)
		print('Reshaped lat and lon')
	# Mask invalid data
	mask = ~np.isnan(sit)
	filtered_si_thickness = np.where(mask, sit, np.nan)
	return lat, lon, filtered_si_thickness, sit_un, ifb

def get_smos(path):
	data = nc.Dataset(path)
	#print(data.variables.keys())
	lat = data.variables['latitude'][:]
	lon = data.variables['longitude'][:]
	sit = data.variables['mean_ice_thickness'][:]
	sit_un = data.variables['uncertainty'][:]
	sid = data.variables["sea_ice_draft"][:]
 
	#mask = ~np.isnan(si_thickness_un)
	#si_thickness_un = np.where(mask, si_thickness_un, np.nan)
	
	#print(si_thickness.shape)
	return lat, lon, sit, sit_un, sid


plt.rcParams.update({
		'font.family':      'serif',
		'font.size':         12,
		'axes.labelsize':    12,
		'xtick.labelsize':   11,
		'ytick.labelsize':   11,
		'legend.fontsize':   12,
		'figure.titlesize':  12,
}) 
 
 
def plot_EM_bird(EM_bird):
	fig = plt.figure(figsize=(6.733, 5))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([1.7e5, 7.45e5, -1.52e6, -8.39e5], ccrs.NorthPolarStereo())

	ax.coastlines()
	ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=2)
	ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.RIVERS, edgecolor='lightgray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=4)
	ax.add_feature(cfeature.COASTLINE, color="black", linewidth=0.1, zorder=5)
	ax.gridlines(draw_labels=True, color="dimgray", zorder=7)
 
	unique_days_EM = EM_bird["DateTime"].dt.strftime('%Y-%m-%d').unique()
	cmap_EM = plt.get_cmap("plasma", len(unique_days_EM))
	for i, day in enumerate(unique_days_EM):
		group = EM_bird[EM_bird["DateTime"].dt.strftime('%Y-%m-%d') == day]
		ax.scatter(group["Longitude"], group["Latitude"], color=cmap_EM(i), s=1, transform=ccrs.PlateCarree(), label=day, zorder=6)
  
	plt.legend(markerscale=5, loc='lower left')
	plt.show()

def plot_als(als_data):
	fig = plt.figure(figsize=(6.733, 5))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([1.7e5, 7.45e5, -1.52e6, -8.39e5], ccrs.NorthPolarStereo())

	ax.coastlines()
	ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=2)
	ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.RIVERS, edgecolor='lightgray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=4)
	ax.add_feature(cfeature.COASTLINE, color="black", linewidth=0.1, zorder=5)
	ax.gridlines(draw_labels=True, color="dimgray", zorder=7)

	unique_files = als_data["SourceFile"].unique()
	cmap_ALS = plt.get_cmap("viridis", len(unique_files))
	for i, file in enumerate(unique_files):
		group = als_data[als_data["SourceFile"] == file]
		ax.scatter(group["Longitude"], group["Latitude"], color=cmap_ALS(i), s=1, transform=ccrs.PlateCarree(), label=file.split(".")[0], zorder=6)

	plt.legend(markerscale=5, loc='lower left')
	plt.show()
 
# ---------------------------- #
em_bird_data = read_all_EMbird_files(EM_bird_path)
als_data = read_all_als_files(ALS_path)

bird_lat = em_bird_data["Latitude"].values
bird_lon = em_bird_data["Longitude"].values
bird_sit = em_bird_data["Total_Thickness"].values
bird_sit_un = em_bird_data["Laser_Range"].values

uit_lat, uit_lon, uit_sit, uit_sit_un, uit_ifb = get_UiT(uit_L3)
smos_lat, smos_lon, smos_sit, smos_sit_un, smos_sid = get_smos(smos_L3)


# ---------------------------- #
	
def plot_uit_EM_bird():
	vmin, vmax = 0, 2.5
	norm = Normalize(vmin=vmin, vmax=vmax)
	cmap = plt.get_cmap("gnuplot2_r")
	sm = ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])   # dummy
	
	fig = plt.figure(figsize=(6.733, 5))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([4.7e5, 7.45e5, -1.35e6, -9.49e5], ccrs.NorthPolarStereo())

	ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=4)
	ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax.add_feature(cfeature.COASTLINE, color="black", linewidth=1, zorder=5)

	mesh = ax.pcolormesh(uit_lon, uit_lat, uit_sit, norm=norm, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2)
	sc = ax.scatter(bird_lon, bird_lat, c=bird_sit, norm=norm, cmap=cmap, s=30, transform=ccrs.PlateCarree(), zorder=3)
	
	plt.colorbar(sm, ax=ax, label="Total SIT [m]")
	plt.show()

def plot_smos_EM_bird():
	vmin, vmax = 0, 2.5
	norm = Normalize(vmin=vmin, vmax=vmax)
	cmap = plt.get_cmap("gnuplot2_r")
	sm = ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])   # dummy
	
	fig = plt.figure(figsize=(6.733, 5))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([4.7e5, 7.45e5, -1.35e6, -9.49e5], ccrs.NorthPolarStereo())

	ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=4)
	ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax.add_feature(cfeature.COASTLINE, color="black", linewidth=1, zorder=5)

	mesh = ax.pcolormesh(smos_lon, smos_lat, smos_sit, norm=norm, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2)
	sc = ax.scatter(bird_lon, bird_lat, c=bird_sit, norm=norm, cmap=cmap, s=30, transform=ccrs.PlateCarree(), zorder=3)
	
	plt.colorbar(sm, ax=ax, label="SIT [m]")
	plt.show()

def map_smos_uit_bird():
	# 1) Shared colormap & normalization
	vmin, vmax = 0, 2.5
	norm = Normalize(vmin=vmin, vmax=vmax)
	cmap = plt.get_cmap("plasma")
	sm = ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])  # dummy for colorbar

	# 2) Make 1x2 subplot, both NorthPolarStereo
	fig, axes = plt.subplots(
		1, 2,
		figsize=(6.733, 4.5),
		subplot_kw={'projection': ccrs.NorthPolarStereo()}
	)

	# Common extent
	extent = [4.7e5, 7.45e5, -1.35e6, -9.49e5]

	for ax, lon, lat, sit, title in (
		(axes[0], uit_lon,  uit_lat,  uit_sit,  "CryoSat UiT SIT"),
		(axes[1], smos_lon, smos_lat, smos_sit, "SMOS SIT"),
	):
		ax.set_extent(extent, crs=ccrs.NorthPolarStereo())

		ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=4)
		ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
		ax.add_feature(cfeature.COASTLINE, color="black", linewidth=1, zorder=5)

		# data layers
		ax.pcolormesh(
			lon, lat, sit,
			norm=norm, cmap=cmap,
			transform=ccrs.PlateCarree(),
			zorder=2
		)
		ax.scatter(
			bird_lon, bird_lat, c=bird_sit,
			norm=norm, cmap=cmap,
			s=30, transform=ccrs.PlateCarree(),
			zorder=3
		)

		#ax.set_title(title, pad=6)
		ax.gridlines(draw_labels=False, linewidth=0.5, color="dimgray", zorder=6)

	#fig.subplots_adjust(left=0.05, right=0.91, wspace=0.05)

	# 4) Add a new axes for colorbar: [left, bottom, width, height]
	#cax = fig.add_axes([0.85, 0.12, 0.02, 0.76])

	# 5) Draw the colorbar into that axes
	#cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
	#cbar.ax.yaxis.set_tick_params(direction='in')
	#cbar.set_label("SIT [m]")
	fig.subplots_adjust(left=0.02, right=0.91, wspace=0.02, bottom=0.08, top=0.95)
	cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.01)
	cbar.ax.yaxis.set_tick_params(direction='in')
	cbar.set_label("Total SIT [m]")

	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\bird_cryo_smos.png", dpi=300, bbox_inches='tight')
	plt.show()
 



if __name__ == "__main__":
	em_bird_data = read_all_EMbird_files(EM_bird_path)
	als_data = read_all_als_files(ALS_path)
 
	#uit_data = get_UiT(uit_L3)
	#smos_data = get_smos(smos_L3)

	#plot_EM_bird(em_bird_data)
	#plot_als(als_data)
 
	#plot_uit_EM_bird()
	#plot_smos_EM_bird()
	map_smos_uit_bird()