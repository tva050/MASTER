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
from scipy.spatial import cKDTree
from matplotlib.ticker import PercentFormatter
from scipy.stats import binned_statistic_2d
import matplotlib.gridspec as gridspec
import statistics as stats
import matplotlib.path as mpath


EM_bird_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\EmBird"
ALS_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\als"
ALS_24_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\als\24_1_L3"
ALS_26_1_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\als\26_1_L3"
ALS_26_2_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\als\26_2_L3"

als_cryo_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOSice\als\26_2_L3\ALS_L2_20140326T155838_162800_resampled.dat"

uit_L3 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product\uit_cryosat2_L3_EASE2_nh25km_2014_03_v3.nc"
smos_L3 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\SMOS_monthly\SMOS_monthly_Icethickness_north_201403.nc"
smos_L3_single = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2014\SMOS_Icethickness_v3.3_north_20140314.nc"

uit_L2 = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT L2 Trajectory\uit_cryosat2_L2_alongtrack_2014_03.txt"

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
	#all_files_26_2 = glob.glob(os.path.join(directory, "26_2_L3", "*.dat"))

	all_data = []

	for file in all_files_24 + all_files_26_1: #+ all_files_26_2
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
	sd = data.variables['snow_depth'][:]

	# Check if lat and lon are 1D and need reshaping
	if lat.ndim == 1 and lon.ndim == 1:
		lon, lat = np.meshgrid(lon, lat)
		print('Reshaped lat and lon')
	# Mask invalid data
	mask = ~np.isnan(sit)
	filtered_si_thickness = np.where(mask, sit, np.nan)
	return lat, lon, filtered_si_thickness, sit_un, ifb, sd

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

# UiT L2 .txt data structure:
# SAR=0_SARIN=1, Orbit_#, Segment_#, Datenumber, Latitude, Longitude, Radar_Freeboard, Surface_Height_WGS84, Sea_Surface_Height_Interp_WGS84,
# SSH_Uncertainty, Mean_Sea_Surface, SLA, Sea_Ice_Class, Lead_Class, Sea_Ice_Roughness, Sea_Ice_Concentration

def get_UiT_L2(path):
	df = pd.read_csv(path, sep=",", skipinitialspace=True, header=0)

	df = df[["Orbit_#", "Segment_#", "Datenumber",
			 "Latitude", "Longitude", "Radar_Freeboard"]]

	df["Latitude"]        = pd.to_numeric(df["Latitude"],        errors="coerce")
	df["Longitude"]       = pd.to_numeric(df["Longitude"],       errors="coerce")
	df["Radar_Freeboard"] = pd.to_numeric(df["Radar_Freeboard"], errors="coerce")

	return df

def get_smos_single(path):
	data = nc.Dataset(path)
	#print(data.variables.keys())
	lat = data.variables['latitude'][:]
	lon = data.variables['longitude'][:]
	sit = data.variables['sea_ice_thickness'][:]
	sit_un = data.variables['ice_thickness_uncertainty'][:]

	land_mask = data.variables['land'][:]
	# Apply land mask to sit and sit_un
	sit = np.where(land_mask == 1, np.nan, sit)
	sit_un = np.where(land_mask == 1, np.nan, sit_un)

	mask_sit = ~np.isnan(sit) & (sit != -999.0) & (sit != 0.0)
	mask_sit_un = ~np.isnan(sit_un) & (sit_un != -999.0) & (sit_un != 0.0)

	si_thickness = np.where(mask_sit, sit, np.nan)
	sit_un = np.where(mask_sit_un, sit_un, np.nan)

	return lat, lon, si_thickness, sit_un

def read_dict(directory):
	# Find all .dat files in the directory
	all_files = glob.glob(os.path.join(directory, "*.dat"))

	# List to collect DataFrames
	all_data = []

	for file in all_files:
		data = read_als_single(file)
		if not data.empty:
			data["SourceFile"] = os.path.basename(file)  # Optional: track source
			all_data.append(data)
		else:
			print(f"Skipped invalid or empty file: {file}")

	if not all_data:
		raise ValueError("No valid data files found in directory.")

	# Concatenate all into a single DataFrame
	combined_data = pd.concat(all_data, ignore_index=True)
	return combined_data


als_24_data = read_dict(ALS_24_path)

def resampling_als(data):
	# Resampling the als data to one second and one minute by taking the arithmetic mean
	# Resampling the data to 1 minute intervals
	data.set_index("DateTime", inplace=True)
	resample_1min = data.resample("1Min").mean(numeric_only=True)
	return resample_1min


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
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))

	# Set the extent to cover latitudes from 60°N to 90°N
	ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())

	# Define a circular boundary in axes coordinates
	theta = np.linspace(0, 2 * np.pi, 100)
	center = [0.5, 0.5]
	radius = 0.5
	circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)

	unique_days_EM = EM_bird["DateTime"].dt.strftime('%Y-%m-%d').unique()
	cmap_EM = plt.get_cmap("plasma", len(unique_days_EM))
	for i, day in enumerate(unique_days_EM):
		group = EM_bird[EM_bird["DateTime"].dt.strftime('%Y-%m-%d') == day]
		ax.scatter(group["Longitude"], group["Latitude"], color=cmap_EM(i), s=1, transform=ccrs.PlateCarree(), label=day, zorder=6)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Thesis\hem_paths_bg.png", dpi=300, bbox_inches='tight', transparent=True)
	plt.show()

def plot_als(als_data):
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))

	# Set the extent to cover latitudes from 60°N to 90°N
	ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())

	# Define a circular boundary in axes coordinates
	theta = np.linspace(0, 2 * np.pi, 100)
	center = [0.5, 0.5]
	radius = 0.5
	circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)

	#ax.coastlines()
	#ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=2)
	#ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	#ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	#ax.add_feature(cfeature.RIVERS, edgecolor='lightgray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=4)
	#ax.add_feature(cfeature.COASTLINE, color="black", linewidth=0.1, zorder=5)
	#ax.gridlines(draw_labels=True, color="dimgray", zorder=7)

	unique_files = als_data["SourceFile"].unique()
	cmap_ALS = plt.get_cmap("viridis", len(unique_files))
	for i, file in enumerate(unique_files):
		group = als_data[als_data["SourceFile"] == file]
		sc=ax.scatter(group["Longitude"], group["Latitude"], color=cmap_ALS(i), s=1, transform=ccrs.PlateCarree(), label=file.split(".")[0], zorder=6)
	#meridians = np.arange(10, 360, 20)
	#for lon in meridians:
	#	latitudes = np.linspace(60, 90, 100)  # or 0 to 90 if you want full lines
	#	longitudes = np.full_like(latitudes, lon)
	#	ax.plot(longitudes, latitudes,
	#			transform=ccrs.PlateCarree(),
	#			color='gray', linewidth=0.5, zorder=0)
	#plt.colorbar(sc,orientation='vertical', pad=0.05, shrink=0.7)
	#plt.legend(markerscale=5, loc='lower left')
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Thesis\als_paths_bg.png", dpi=300, bbox_inches='tight', transparent=True)
	plt.show()

def bird_als_paths(als_data, EM_bird_data):
	fig, axes = plt.subplots(1, 2, figsize=(6.733, 4.2), subplot_kw={'projection': ccrs.NorthPolarStereo()})
	extent = [1.7e5, 7.45e5, -1.52e6, -8.39e5]

	axes[0].set_extent(extent, crs=ccrs.NorthPolarStereo())
	axes[0].add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=4)
	axes[0].add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	axes[0].add_feature(cfeature.COASTLINE, color="black", linewidth=0.5, zorder=5)
	axes[0].gridlines(draw_labels=True, linewidth=0.5, color="dimgray", zorder=6)

	unique_days_EM = EM_bird_data["DateTime"].dt.strftime('%Y-%m-%d').unique()
	cmap_EM = plt.get_cmap("plasma", len(unique_days_EM))
	for i, day in enumerate(unique_days_EM):
		group = EM_bird_data[EM_bird_data["DateTime"].dt.strftime('%Y-%m-%d') == day]
		axes[0].scatter(group["Longitude"], group["Latitude"], color=cmap_EM(i), s=1, transform=ccrs.PlateCarree(), zorder=6)
	axes[0].text(0.03, 0.03, "HEM", transform=axes[0].transAxes, color='black')

	axes[1].set_extent(extent, crs=ccrs.NorthPolarStereo())
	axes[1].add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=4)
	axes[1].add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	axes[1].add_feature(cfeature.COASTLINE, color="black", linewidth=0.5, zorder=5)
	axes[1].gridlines(draw_labels=True, linewidth=0.5, color="dimgray", zorder=6)

	unique_files = als_data["SourceFile"].unique()
	cmap_ALS = plt.get_cmap("viridis", len(unique_files))
	for i, file in enumerate(unique_files):
		group = als_data[als_data["SourceFile"] == file]
		axes[1].scatter(group["Longitude"], group["Latitude"], color=cmap_ALS(i), s=1, transform=ccrs.PlateCarree(), zorder=6)
	axes[1].text(0.03, 0.03, "ALS", transform=axes[1].transAxes, color='black')

	plt.subplots_adjust(left=0.02, right=0.986, wspace=0.03, bottom=0.017, top=0.957)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\SMOSice validations - CryoSat and SMOS\bird_als_paths.png", dpi=300, bbox_inches='tight')
	plt.show()


def plot_L2(uit_L2):
	uit_L2 = get_UiT_L2(uit_L2)
	lon = uit_L2["Longitude"]
	lat = uit_L2["Latitude"]
	rf = uit_L2["Radar_Freeboard"]


	fig = plt.figure(figsize=(6.733, 5))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([1.7e5, 7.45e5, -1.52e6, -8.39e5], ccrs.NorthPolarStereo())

	# map features
	#ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1,   zorder=2)

	sc = ax.scatter(
		lon, lat, c=rf, cmap="plasma", s=1,
		transform=ccrs.PlateCarree(), zorder=6
	)

	# now explicitly pass it to colorbar:
	cbar = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
	cbar.set_label("Radar Freeboard [m]")
	plt.title("UiT L2 Freeboard")
	plt.show()

# ---------------------------- #

em_bird_data = read_all_EMbird_files(EM_bird_path)

bird_lat = em_bird_data["Latitude"].values
bird_lon = em_bird_data["Longitude"].values
bird_sit = em_bird_data["Total_Thickness"].values
bird_L_range = em_bird_data["Laser_Range"].values

uit_lat, uit_lon, uit_sit, uit_sit_un, uit_ifb, uit_sd = get_UiT(uit_L3)
smos_lat, smos_lon, smos_sit, smos_sit_un, smos_sid = get_smos(smos_L3)


als_all_data = read_all_als_files(ALS_path)
als_all_lat = als_all_data["Latitude"].values
als_all_lon = als_all_data["Longitude"].values
als_all_freeboard = als_all_data["Freeboard"].values


als_single_df = read_als_single(als_cryo_path)
als_lat = als_single_df["Latitude"].values
als_lon = als_single_df["Longitude"].values
als_freeboard = als_single_df["Freeboard"].values

uit_L2 = get_UiT_L2(uit_L2)
uit_L2_lat = uit_L2["Latitude"]
uit_L2_lon = uit_L2["Longitude"]
uit_L2_rf = uit_L2["Radar_Freeboard"]

smos_s_lat, smos_s_lon, smos_s_sit, smos_s_sit_un = get_smos_single(smos_L3_single)

als_24_raw_tFB = als_24_data["Freeboard"].values
als_24_raw_lat = als_24_data["Latitude"].values
als_24_raw_lon = als_24_data["Longitude"].values
als_24_raw_sit = als_24_raw_tFB * 5.5 # Converting tFB to SIT using the factor 5.5, se smosice report 

als_24_resampled = resampling_als(als_24_data)

als_24_resampled_lat = als_24_resampled["Latitude"].values
als_24_resampled_lon = als_24_resampled["Longitude"].values
als_24_resampled_tFB = als_24_resampled["Freeboard"].values
als_24_sit = als_24_resampled_tFB * 5.5 # Converting tFB to SIT using the factor 5.5, se smosice report

# ---------------------------- #
# Data handling
# ---------------------------- #

def reprojecting(lon, lat, proj=ccrs.NorthPolarStereo()):
	transformer = proj.transform_points(ccrs.PlateCarree(), lon, lat)
	x = transformer[..., 0]
	y = transformer[..., 1]
	return x, y

bird_x, bird_y = reprojecting(bird_lon, bird_lat)

uit_x, uit_y = reprojecting(uit_lon, uit_lat)
smos_x, smos_y = reprojecting(smos_lon, smos_lat)

als_x, als_y = reprojecting(als_lon, als_lat)
uit_L2_x, uit_L2_y = reprojecting(uit_L2_lon, uit_L2_lat)
smos_s_x, smos_s_y = reprojecting(smos_s_lon, smos_s_lat)

als_24_x, als_24_y = reprojecting(als_24_resampled_lon, als_24_resampled_lat)
als_24_raw_x, als_24_raw_y = reprojecting(als_24_raw_lon, als_24_raw_lat)

als_all_x, als_all_y = reprojecting(als_all_lon, als_all_lat)

def resample_to_cryo_grid(x_source, y_source, source_sit, source_sit_un, x_target, y_target, radius):
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

smos_sit = resample_to_cryo_grid(smos_x, smos_y, smos_sit, smos_sit_un, uit_x, uit_y, 12500)
bird_unc = np.ones_like(bird_sit)
bird_tot_sit = resample_to_cryo_grid(bird_x, bird_y, bird_sit, bird_unc, uit_x, uit_y, 12500)

uit_tot_sit = uit_sit + uit_sd
smos_tot_sit = smos_sit + uit_sd

# ---------------------------- #
# HEM system
# ---------------------------- #

def map_smos_uit_bird_raw():
	# 1) Shared colormap & normalization
	vmin, vmax = 0, 2.5
	norm = Normalize(vmin=vmin, vmax=vmax)
	cmap = plt.get_cmap("plasma")
	sm = ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array([])  # dummy for colorbar

	# 2) Make 1x2 subplot, both NorthPolarStereo
	fig, axes = plt.subplots(
		1, 2,
		figsize=(6.733, 3.8),
		subplot_kw={'projection': ccrs.NorthPolarStereo()}
	)

	# Common extent
	extent = [5.0e5, 7.45e5, -1.322e6, -9.934e5]

	for ax, lon, lat, sit, title in (
		(axes[0], uit_lon,  uit_lat,  uit_tot_sit,  "UiT"),
		(axes[1], uit_lon, uit_lat, smos_tot_sit, "SMOS"),
	):
		ax.set_extent(extent, crs=ccrs.NorthPolarStereo())

		ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=4)
		ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
		ax.add_feature(cfeature.COASTLINE, color="black", linewidth=1, zorder=5)

		# data layers
		ax.pcolormesh(lon, lat, sit, norm=norm, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2)
		ax.scatter(bird_lon, bird_lat, c=bird_sit, norm=norm, cmap=cmap, s=30, transform=ccrs.PlateCarree(), zorder=3)

		ax.gridlines(draw_labels=False, linewidth=0.5, color="dimgray", zorder=6)
		ax.set_title(title, pad=4)

	fig.subplots_adjust(left=0.05, right=0.954, wspace=0.0, bottom=0.038, top=0.938)
	#cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.01) #cbar = fig.colorbar(mesh, ax=ax1, orientation="vertical", pad=0.02)
	cbar = fig.colorbar(sm, ax=axes[1], orientation='vertical', pad=0.02)
	cbar.ax.yaxis.set_tick_params(direction='in')
	cbar.set_label("Total SIT [m]")


	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\SMOSice validations - CryoSat and SMOS\bird_cryo_smos_raw.png", dpi=300, bbox_inches='tight')
	plt.show()

def raw_distributions():
	grid_pts = np.column_stack([uit_x.ravel(), uit_y.ravel()])
	uit_tot_sit_flat = uit_tot_sit.ravel()
	smos_tot_sit_flat = smos_tot_sit.ravel()

	tree = cKDTree(grid_pts)

	_, idx = tree.query(np.column_stack([bird_x, bird_y]), k=1)
	unique_idx = np.unique(idx)

	uit_vals = uit_tot_sit_flat[unique_idx]
	smos_vals = smos_tot_sit_flat[unique_idx]

	w_bird = np.ones_like(bird_sit.ravel()) / len(bird_sit.ravel())
	w_uit = np.ones_like(uit_vals) / len(uit_vals)
	w_smos = np.ones_like(smos_vals) / len(smos_vals)

	fig, ax = plt.subplots(1, 2, figsize=(6.733, 3.5))
	bin_edges=np.linspace(0,3,23)

	ax[0].hist(uit_vals, weights=w_uit, histtype="step", fill=False, bins=bin_edges, color='#4ca64c', label='UiT', density=False, linewidth=1.7, zorder=1)
	ax[0].hist(bird_sit.ravel(), weights=w_bird, histtype="step", fill=False, bins=bin_edges, color='black', alpha=0.7, label='HEM', density=False, linewidth=1.7, zorder=0)
	ax[0].legend(loc ='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=2, borderaxespad=0.0, handletextpad=0.3)
	ax[0].set_xlabel("Total SIT [m]")
	ax[0].set_ylabel("Observation (%)")
	ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
	ax[0].tick_params(axis='both', direction='in')

	ax[1].hist(smos_vals, weights=w_smos, histtype="step", fill=False, bins=bin_edges, color='#4c4cff', label='SMOS', density=False, linewidth=1.7, zorder=1)
	ax[1].hist(bird_sit.ravel(), weights=w_bird, histtype="step", fill=False, bins=bin_edges, color='black', alpha=0.7, density=False, linewidth=1.7, zorder=0)
	ax[1].legend(loc ='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=1, borderaxespad=0.0, handletextpad=0.3)
	ax[1].set_xlabel("Total SIT [m]")
	ax[1].yaxis.set_major_formatter(PercentFormatter(1))
	ax[1].tick_params(axis='both', direction='in')

	plt.subplots_adjust(left=0.094, right=0.995, wspace=0.2, bottom=0.117, top=0.921)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\SMOSice validations - CryoSat and SMOS\bird_cryo_smos_hist_raw.png", dpi=300, bbox_inches='tight')
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
		figsize=(6.733, 3.8),
		subplot_kw={'projection': ccrs.NorthPolarStereo()}
	)

	# Common extent
	extent = [4.938e5, 7.45e5, -1.334e6, -9.934e5]

	for ax, lon, lat, sit, title in (
		(axes[0], uit_lon,  uit_lat,  uit_tot_sit,  "UiT"),
		(axes[1], uit_lon, uit_lat, smos_tot_sit, "SMOS"),
	):
		ax.set_extent(extent, crs=ccrs.NorthPolarStereo())

		ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=4)
		ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
		ax.add_feature(cfeature.COASTLINE, color="black", linewidth=1, zorder=5)

		# data layers
		ax.pcolormesh(lon, lat, sit, norm=norm, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2)
		ax.scatter(uit_lon, uit_lat, c=bird_tot_sit, norm=norm, cmap=cmap, s=50, transform=ccrs.PlateCarree(), zorder=3)

		ax.gridlines(draw_labels=False, linewidth=0.5, color="dimgray", zorder=6)
		ax.set_title(title, pad=4)

	fig.subplots_adjust(left=0.02, right=0.91, wspace=0.0, bottom=0.04, top=0.95)
	cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.01)
	cbar.ax.yaxis.set_tick_params(direction='in')
	cbar.set_label("Total SIT [m]")


	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\SMOSice validations - CryoSat and SMOS\bird_cryo_smos.png", dpi=300, bbox_inches='tight')
	plt.show()

def distribution():
	# 1) build a mask of exactly those grid‑cells where bird has data
	valid_mask = ~np.isnan(bird_tot_sit)

	# 2) pull out only those values
	uit_vals  = uit_tot_sit[valid_mask]
	smos_vals = smos_tot_sit[valid_mask]
	bird_vals = bird_tot_sit[valid_mask]


	w_uit  = np.ones_like(uit_vals)  / len(uit_vals)
	w_smos = np.ones_like(smos_vals) / len(smos_vals)
	w_bird = np.ones_like(bird_vals) / len(bird_vals)

	fig, ax = plt.subplots(1, 2, figsize=(6.733, 3.5))
	bin_edges=np.linspace(0,1.5,23)

	ax[0].hist(uit_vals, weights=w_uit, histtype="step", fill=False, bins=bin_edges, color='#4ca64c', label='UiT', density=False, linewidth=1.7, zorder=1)
	ax[0].hist(bird_vals, weights=w_bird, histtype="step", fill=False, bins=bin_edges, color='black', alpha=0.7, label='HEM', density=False, linewidth=1.7, zorder=0)
	ax[0].legend(loc ='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=2, borderaxespad=0.0, handletextpad=0.3)
	ax[0].set_xlabel("Total SIT [m]")
	ax[0].set_ylabel("Observation (%)")
	ax[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
	ax[0].tick_params(axis='both', direction='in')

	ax[1].hist(smos_vals, weights=w_smos, histtype="step", fill=False, bins=bin_edges, color='#4c4cff', label='SMOS', density=False, linewidth=1.7, zorder=1)
	ax[1].hist(bird_vals, weights=w_bird, histtype="step", fill=False, bins=bin_edges, color='black', alpha=0.7, density=False, linewidth=1.7, zorder=0)
	ax[1].legend(loc ='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=1, borderaxespad=0.0, handletextpad=0.3)
	ax[1].set_xlabel("Total SIT [m]")
	ax[1].yaxis.set_major_formatter(PercentFormatter(1))
	ax[1].tick_params(axis='both', direction='in')

	plt.subplots_adjust(left=0.094, right=0.995, wspace=0.2, bottom=0.117, top=0.921)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\SMOSice validations - CryoSat and SMOS\bird_cryo_smos_hist.png", dpi=300, bbox_inches='tight')
	plt.show()



def hem_stats():
	nan_mask = ~np.isnan(bird_tot_sit) #& ~np.isnan(uit_tot_sit) & ~np.isnan(smos_tot_sit)
	#bird_tot_sit_f, uit_tot_sit_f, smos_tot_sit_f = [arr[nan_mask] for arr in (bird_tot_sit, uit_tot_sit, smos_tot_sit)]
	bird_tot_sit_f = bird_tot_sit[nan_mask]
	uit_tot_sit_f = uit_tot_sit[nan_mask]
	smos_tot_sit_f = smos_tot_sit[nan_mask]
 
	range_mask = (bird_tot_sit_f >= 0) & (uit_tot_sit_f >= 0) & (smos_tot_sit_f >= 0) & (bird_tot_sit_f <= 1.2) & (uit_tot_sit_f <= 1.2) & (smos_tot_sit_f <= 1.2)
	bird_tot_sit_f, uit_tot_sit_f, smos_tot_sit_f = [arr[range_mask] for arr in (bird_tot_sit_f, uit_tot_sit_f, smos_tot_sit_f)]

	metrics = {
		'UiT': {
			'bias': np.mean(uit_tot_sit_f - bird_tot_sit_f),
			'rmse': np.sqrt(np.mean((uit_tot_sit_f - bird_tot_sit_f) ** 2)),
			'cc': np.corrcoef(uit_tot_sit_f, bird_tot_sit_f)[0, 1],
			'mean': np.mean(uit_tot_sit_f)
		},
		'SMOS': {
			'bias': np.mean(smos_tot_sit_f - bird_tot_sit_f),
			'rmse': np.sqrt(np.mean((smos_tot_sit_f - bird_tot_sit_f) ** 2)),
			'cc': np.corrcoef(smos_tot_sit_f, bird_tot_sit_f)[0, 1],
			'mean': np.mean(smos_tot_sit_f)
		}
	}
 
	products = ['UiT', 'SMOS']

	# Normalize bias and rmse across products
	for metric in ('bias', 'rmse'):
		values = np.array([metrics[p][metric] for p in products])
		mn, mx = values.min(), values.max()
		span = (mx - mn) if mx != mn else 1.0
		for p in products:
			metrics[p]['N' + metric] = (metrics[p][metric] - mn) / span

	# Compute DISO
	for p in products:
		NB = metrics[p]['Nbias']
		NR = metrics[p]['Nrmse']
		NC = metrics[p]['cc']
		metrics[p]['DISO'] = np.sqrt(NB**2 + NR**2 + (NC - 1.0)**2)

	# Print results
	for p in products:
		m = metrics[p]
		print(f"{p}: Mean={m['mean']:.3f}, Bias={m['bias']:.3f}, RMSE={m['rmse']:.3f}, "
			  f"CC={m['cc']:.3f}, DISO={m['DISO']:.3f}")

def hem_stats_bined():
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']

	# Filter NaNs and range
	#nan_mask = ~np.isnan(bird_tot_sit)
	#bird_f = bird_tot_sit[nan_mask]
	#uit_f = uit_tot_sit[nan_mask]
	#smos_f = smos_tot_sit[nan_mask]
	nan_mask = ~np.isnan(bird_tot_sit) & ~np.isnan(uit_tot_sit) & ~np.isnan(smos_tot_sit)
	bird_f, uit_f, smos_f = [arr[nan_mask] for arr in (bird_tot_sit, uit_tot_sit, smos_tot_sit)]

	range_mask = (
		(bird_f >= 0) & (bird_f <= 1.2) &
		(uit_f >= 0) & (uit_f <= 1.2) &
		(smos_f >= 0) & (smos_f <= 1.2)
	)

	bird_f, uit_f, smos_f = [arr[range_mask] for arr in (bird_f, uit_f, smos_f)]

	products = {'UiT': uit_f, 'SMOS': smos_f}
	results = {p: {} for p in products}

	for i in range(len(bins) - 1):
		lo, hi = bins[i], bins[i + 1]
		label = labels[i]
		bin_mask = (bird_f >= lo) & (bird_f < hi)

		if not np.any(bin_mask):
			for p in products:
				results[p][label] = dict(bias=np.nan, rmse=np.nan, cc=np.nan, Nbias=np.nan, Nrmse=np.nan, DISO=np.nan)
			continue

		bird_bin = bird_f[bin_mask]
		arrs = {p: products[p][bin_mask] for p in products}

		raw = {}
		for p, arr in arrs.items():
			b = np.mean(arr - bird_bin)
			r = np.sqrt(np.mean((arr - bird_bin) ** 2))
			c = np.corrcoef(arr, bird_bin)[0, 1]
			raw[p] = dict(bias=b, rmse=r, cc=c)

		# Normalize bias and rmse within this bin
		for metric in ('bias', 'rmse'):
			vals = np.array([raw[p][metric] for p in products])
			mn, mx = vals.min(), vals.max()
			span = (mx - mn) if mx != mn else 1.0
			for p in products:
				raw[p]['N' + metric] = (raw[p][metric] - mn) / span

		# Compute DISO
		for p in products:
			NB = raw[p]['Nbias']
			NR = raw[p]['Nrmse']
			NC = raw[p]['cc']
			raw[p]['DISO'] = np.sqrt(NB**2 + NR**2 + (NC - 1.0)**2)
			results[p][label] = raw[p]

	# Print results
	for prod, bins in results.items():
		print(f"\n{prod}:")
		for bin_label, m in bins.items():
			print(f"  {bin_label}: Bias={m['bias']:.3f}, RMSE={m['rmse']:.3f}, "
				  f"CC={m['cc']:.3f}, DISO={m['DISO']:.3f}")



# ---------------------------- #
# ALS
# ---------------------------- #

def find_uit_track():
	# Find the the uit track closest to the als track
	# store the corresponding lat, lon and freeboard from the uit track

	# Create a KDTree for fast nearest-neighbor search
	grid_pts = np.column_stack([uit_L2_x.ravel(), uit_L2_y.ravel()])
	tree = cKDTree(grid_pts)

	# Find the nearest neighbors in the grid for each point in the ALS data
	_, idx = tree.query(np.column_stack([als_x, als_y]), k=1)
	unique_idx = np.unique(idx)

	lon_vals = uit_L2_lon.ravel()[unique_idx]
	lat_vals = uit_L2_lat.ravel()[unique_idx]
	rf_vals  = uit_L2_rf.ravel()[unique_idx]

	return lon_vals, lat_vals, rf_vals, unique_idx

def find_snow_track_at_l2(uit_L2_x, uit_L2_y, grid_x, grid_y, grid_sd):
	# 1. Flatten the grid coordinates and snow depth
	flat_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
	flat_sd = grid_sd.ravel()

	# 2. Create KDTree from gridded coordinates
	tree = cKDTree(flat_coords)

	# 3. Query snow depth at L2 locations
	_, idx = tree.query(np.column_stack((uit_L2_x, uit_L2_y)), k=1)

	# 4. Use nearest neighbor index to extract snow depth
	sd_vals = flat_sd[idx]
	return sd_vals

uit_L2_lon_p, uit_L2_lat_p, uit_L2_rf, uit_idx = find_uit_track()
L2_sd_vals = find_snow_track_at_l2(uit_L2_x.ravel()[uit_idx], uit_L2_y.ravel()[uit_idx], uit_x, uit_y, uit_sd)



def plot_als_swath_xy(als_x, als_y, freeb):
	fig, ax = plt.subplots(figsize=(6.733, 3.7))
	sc = ax.scatter(als_x, als_y, c=freeb, cmap="plasma", s=1, zorder=2)
	sc.set_clim(vmin=0, vmax=1)
	ax.set_xlabel("X [m]")
	ax.set_ylabel("Y [m]")
	ax.set_title("ALS Freeboard")
	cbar = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
	cbar.ax.yaxis.set_tick_params(direction='in')
	cbar.set_label("Freeboard [m]")
	plt.show()



def plot_uit_als_track_raw():
	fig = plt.figure(figsize=(6.733, 3.7))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([3.557e5, 6.065e5, -1.43e6, -1.21e6], ccrs.NorthPolarStereo())

	ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=4)
	ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax.add_feature(cfeature.COASTLINE, color="black", linewidth=1, zorder=5)
	ax.gridlines(draw_labels=False, linewidth=0.5, color="dimgray", zorder=6)

	sc = ax.scatter(uit_L2_lon_p, uit_L2_lat_p, c=uit_L2_rf, cmap="plasma", s=6, transform=ccrs.PlateCarree(), zorder=3)
	al = ax.scatter(als_lon, als_lat, color="gray", s=10, transform=ccrs.PlateCarree(), zorder=2, label="ALS")

	cbar = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
	cbar.ax.yaxis.set_tick_params(direction='in')
	cbar.set_label("UiT rFB [m]")

	plt.legend(loc='upper left', bbox_to_anchor=(0, 1.08), frameon=False, ncol=1, borderaxespad=0.0, handletextpad=0.3)
	plt.subplots_adjust(top=0.945, bottom=0.04)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\SMOSice validations - CryoSat and SMOS\uit_als_track.png", dpi=300, bbox_inches='tight')
	plt.show()

uit_tFB = uit_L2_rf + L2_sd_vals

def hist():
	w_uit_tFB = np.ones_like(uit_tFB) / len(uit_tFB)
	w_als_freeboard = np.ones_like(als_freeboard) / len(als_freeboard)

	bin_edges = np.linspace(-0.15, 0.5, 25)

	fig, ax = plt.subplots(figsize=(6.733, 3.7))
	#bin_edges = np.linspace(-0.3, 0.8, 36)
	#ax.hist(uit_tFB, weights=1/len(uit_tFB)*np.ones(len(uit_tFB)), histtype="step", fill=False, bins=30, color='#4ca64c', label='UiT', density=False, linewidth=1.7, zorder=1)
	#ax.hist(als_freeboard, weights=1/len(als_freeboard)*np.ones(len(als_freeboard)), histtype="step", fill=False, bins=30, color='black', alpha=0.7, label='ALS', density=False, linewidth=1.7, zorder=0)
	ax.hist(uit_tFB, histtype="step", fill=False, bins=bin_edges, color='#4ca64c', label='UiT', density=True, linewidth=1.7, zorder=1)
	ax.hist(als_freeboard, histtype="step", fill=False, bins=bin_edges, color='black', alpha=0.7, label='ALS', density=True, linewidth=1.7, zorder=0)
	ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=2, borderaxespad=0.0, handletextpad=0.3)
	ax.set_xlabel("Total Freeboard [m]")
	ax.set_ylabel("Observation (%)")
	#ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
	ax.tick_params(axis='both', direction='in')
	plt.show()

def plot_track_hist():
	w_uit_tFB = np.ones_like(uit_tFB) / len(uit_tFB)
	w_als_freeboard = np.ones_like(als_freeboard) / len(als_freeboard)

	bin_edges = np.linspace(-0.15, 0.5, 25)

	# Manually build figure and axes
	fig = plt.figure(figsize=(6.733, 3.35))
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

	ax0 = fig.add_subplot(gs[0])
	ax1 = fig.add_subplot(gs[1], projection=ccrs.NorthPolarStereo())

	ax0.hist(uit_tFB, histtype="step", fill=False, bins=bin_edges, color='#4ca64c', label='UiT', density=True, linewidth=1.7, zorder=1)
	ax0.hist(als_freeboard, histtype="step", fill=False, bins=bin_edges, color='black', alpha=0.7, label='ALS', density=True, linewidth=1.7, zorder=0)
	ax0.legend(loc='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=2, borderaxespad=0.0, handletextpad=0.3)
	ax0.set_xlabel("Total Freeboard [m]")
	ax0.set_ylabel("Probability density")
	ax0.tick_params(axis='both', direction='in')

	ax1.set_extent([3.557e5, 6.065e5, -1.46e6, -1.179e6], crs=ccrs.NorthPolarStereo())
	ax1.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=4)
	ax1.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax1.add_feature(cfeature.COASTLINE, color="black", linewidth=0.5, zorder=5)
	ax1.gridlines(draw_labels=False, linewidth=0.5, color="dimgray", zorder=6)

	sc = ax1.scatter(uit_L2_lon_p, uit_L2_lat_p, c=uit_tFB, cmap="plasma", s=6, transform=ccrs.PlateCarree(), zorder=3) 
	ax1.scatter(als_lon, als_lat, color="gray", s=10, transform=ccrs.PlateCarree(), zorder=2, label="ALS")
	ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=1, borderaxespad=0.0, handletextpad=0.3)

	sc.set_clim(vmin=0, vmax=1.2)
	cbar = fig.colorbar(sc, ax=ax1, orientation="vertical", pad=0.02)
	cbar.ax.yaxis.set_tick_params(direction='in')
	cbar.set_label("UiT tFB [m]")

	plt.subplots_adjust(left=0.093, right=0.966, wspace=0.039, bottom=0.124, top=0.923)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\SMOSice validations - CryoSat and SMOS\uit_als_track_hist.png", dpi=300, bbox_inches='tight')
	#plt.show()

# ---------------------------- #
# ALS, smos
# ---------------------------- #


def read_dict(directory):
	# Find all .dat files in the directory
	all_files = glob.glob(os.path.join(directory, "*.dat"))

	# List to collect DataFrames
	all_data = []

	for file in all_files:
		data = read_als_single(file)
		if not data.empty:
			data["SourceFile"] = os.path.basename(file)  # Optional: track source
			all_data.append(data)
		else:
			print(f"Skipped invalid or empty file: {file}")

	if not all_data:
		raise ValueError("No valid data files found in directory.")

	# Concatenate all into a single DataFrame
	combined_data = pd.concat(all_data, ignore_index=True)
	return combined_data


als_24_data = read_dict(ALS_24_path)

def resampling_als(data):
	# Resampling the als data to one second and one minute by taking the arithmetic mean
	# Resampling the data to 1 minute intervals
	data.set_index("DateTime", inplace=True)
	resample_1min = data.resample("1Min").mean(numeric_only=True)
	return resample_1min


#als_24_resampled = resampling_als(als_24_data)
#
#als_24_resampled_lat = als_24_resampled["Latitude"].values
#als_24_resampled_lon = als_24_resampled["Longitude"].values
#als_24_resampled_tFB = als_24_resampled["Freeboard"].values
#als_24_sit = als_24_resampled_tFB * 5.5 # Converting tFB to SIT using the factor 5.5, se smosice report

def plot_als_res():
	smos_s_sit_sq = np.squeeze(smos_s_sit)
	fig = plt.figure(figsize=(6.733, 5))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([4.746e5, 7.127e5, -1.305e6, -1.025e6], ccrs.NorthPolarStereo())

	sc = ax.scatter(als_24_resampled_lon, als_24_resampled_lat, c=als_24_sit, s=20, cmap="plasma", transform=ccrs.PlateCarree(), vmin = 0, vmax = 1.3, zorder=6)
	plt.plot(als_24_resampled_lon, als_24_resampled_lat, color="gray", linewidth=1, transform=ccrs.PlateCarree(), zorder=7)
	mesh = ax.pcolormesh(smos_s_lon, smos_s_lat, smos_s_sit_sq , cmap="plasma", vmin = 0, vmax = 1.3, transform=ccrs.PlateCarree(), zorder=5)

	plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02).set_label("SIT [m]")

	plt.show()

def smos_als_track():
	smos_s_sit_sq = np.squeeze(smos_s_sit)
	smos_coords = np.column_stack([smos_s_x.ravel(), smos_s_y.ravel()])
	tree = cKDTree(smos_coords)

	# 1. Build ALS coords and filter invalid entries (e.g., NaN)
	als_coords = np.column_stack([als_24_x, als_24_y])
	valid_mask = np.all(np.isfinite(als_coords), axis=1)  # filter out rows with NaN or Inf
	als_coords_valid = als_coords[valid_mask]

	# 2. Query KDTree
	_, idx = tree.query(als_coords_valid, k=1)

	# 3. Get overlapping SMOS SIT
	smos_sit_overlap = smos_s_sit_sq.ravel()[idx]
	smos_lon = smos_s_lon.ravel()[idx]
	smos_lat = smos_s_lat.ravel()[idx]
	return smos_sit_overlap, smos_lon, smos_lat, smos_s_sit_sq

smos_als_overlap, smos_s_lon_filt, smos_s_lat_filt, smos_s_sit_sq = smos_als_track()


def smos_als_track_hist():
	w_smos_sit = np.ones_like(smos_als_overlap) / len(smos_als_overlap)
	w_als_sit = np.ones_like(als_24_sit) / len(als_24_sit)

	bin_edges = np.linspace(-0.3, 1.2, 21)

	fig = plt.figure(figsize=(6.733, 3.35))
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

	ax0 = fig.add_subplot(gs[0])
	ax1 = fig.add_subplot(gs[1], projection=ccrs.NorthPolarStereo())

	ax0.hist(smos_als_overlap, histtype="step", fill=False, bins=bin_edges, color='#4c4cff', label='SMOS', density=True, linewidth=1.7, zorder=1)
	ax0.hist(als_24_sit, histtype="step", fill=False, bins=bin_edges, color='black', alpha=0.7, label='ALS', density=True, linewidth=1.7, zorder=0)

	ax0.legend(loc='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=2, borderaxespad=0.0, handletextpad=0.3)
	ax0.set_xlabel("SIT [m]")
	ax0.set_ylabel("Probability density")
	ax0.tick_params(axis='both', direction='in')

	ax1.set_extent([4.746e5, 7.127e5, -1.305e6, -1.025e6], crs=ccrs.NorthPolarStereo())
	ax1.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=5)
	ax1.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax1.add_feature(cfeature.COASTLINE, color="black", linewidth=0.5, zorder=6)
	ax1.gridlines(draw_labels=False, linewidth=0.5, color="dimgray", zorder=7)

	mesh = ax1.pcolormesh(smos_s_lon, smos_s_lat, smos_s_sit_sq , cmap="plasma", vmin = 0, vmax = 1.2, transform=ccrs.PlateCarree(), zorder=2)
	sc = ax1.scatter(als_24_resampled_lon, als_24_resampled_lat, c=als_24_sit, s=20, cmap="plasma", transform=ccrs.PlateCarree(), vmin = 0, vmax = 1.2, zorder=3)
	ax1.plot(als_24_resampled_lon, als_24_resampled_lat, color="gray", linewidth=1, transform=ccrs.PlateCarree(), zorder=4)

	cbar = fig.colorbar(mesh, ax=ax1, orientation="vertical", pad=0.02)
	cbar.ax.yaxis.set_tick_params(direction='in')
	cbar.set_label("SIT [m]")

	plt.subplots_adjust(left=0.094, right=0.967, wspace=0.0, bottom=0.11, top=0.931)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\SMOSice validations - CryoSat and SMOS\smos_als_track_hist.png", dpi=300, bbox_inches='tight')
	#plt.show()

# Statistical metrics for ALS

def resample_to_cryo_grid_1(x_source, y_source, source_sit, source_sit_un, x_target, y_target, radius):
	# Combine target coordinates and build KDTree
	target_coords = np.column_stack((x_target, y_target))
	target_tree = cKDTree(target_coords)

	# Initialize arrays to store resampled data and weight sums
	resampled_sit = np.full(x_target.shape, np.nan)
	weights_sum = np.zeros(x_target.shape)

	# Flatten source arrays for iteration
	source_sit = source_sit.ravel()
	source_sit_un = source_sit_un.ravel()
	x_source = x_source.ravel()
	y_source = y_source.ravel()

	for i in range(len(source_sit)):
		if np.isnan(source_sit[i]) or np.isnan(source_sit_un[i]):
			continue  # Skip invalid data points

		# Find indices within the specified radius
		indices = target_tree.query_ball_point([x_source[i], y_source[i]], radius)

		if not indices:
			continue  # If no neighbors are found, skip

		weight = 1 / source_sit_un[i] if source_sit_un[i] != 0 else 0  # Avoid division by zero

		for idx in indices:
			# Initialize if NaN
			if np.isnan(resampled_sit[idx]):
				resampled_sit[idx] = 0

			# Weighted sum calculation
			resampled_sit[idx] = (resampled_sit[idx] * weights_sum[idx] + source_sit[i] * weight) / (weights_sum[idx] + weight)
			weights_sum[idx] += weight

	# Mask invalid values (no data points found)
	resampled_sit[weights_sum == 0] = np.nan

	return np.ma.masked_invalid(resampled_sit)
uit_L2_x_p, uit_L2_y_p = reprojecting(uit_L2_lon_p, uit_L2_lat_p)

#print(f"als_x shape: {als_x.shape}, als_y shape: {als_y.shape}")
als_unc = np.ones_like(als_freeboard)
als_filt = resample_to_cryo_grid_1(als_x, als_y, als_freeboard, als_unc, uit_L2_x_p, uit_L2_y_p, radius=938) #938

#print("als_fr shape:", als_filt.shape)
#print("uit_L2_rf shape:", uit_L2_rf.shape)

def als_tFB_stats():
	nan_mask = ~np.isnan(uit_tFB) & ~np.isnan(als_filt)  # Mask for valid data points
	uit_tFB_m = uit_tFB[nan_mask]
	als_fr_m = als_filt[nan_mask]

	range_mask = (uit_tFB_m >= -0.15) & (als_fr_m >= -0.15) & (uit_tFB_m <=  0.5) & (als_fr_m <=  0.5)
	uit_tFB_m = uit_tFB_m[range_mask]
	als_fr_m = als_fr_m[range_mask]
 	
	bias = np.mean(uit_tFB_m - als_fr_m)
	rmse = np.sqrt(np.mean((uit_tFB_m - als_fr_m) ** 2))
	cc = np.corrcoef(uit_tFB_m.ravel(), als_fr_m.ravel(), rowvar=False)[0, 1]
	mean_uit = np.mean(uit_tFB_m)
	mean_als = np.mean(als_fr_m)

	# Normalize using range of reference data
	ref_range = uit_tFB_m.max() - uit_tFB_m.min()
	span = ref_range #if ref_range != 0 else 1.0

	Nbias = np.abs(bias) / span
	Nrmse = rmse / span

	# DISO: deviation from ideal stats (bias=0, rmse=0, cc=1)
	DISO = np.sqrt(Nbias**2 + Nrmse**2 + (cc - 1.0)**2)

	# Print results
	print(f"UiT: Mean={mean_uit:.3f}, Bias={bias:.3f}, RMSE={rmse:.3f}, CC={cc:.3f}, DISO={DISO:.3f}")
	print(f"ALS: Mean={mean_als:.3f}")
 
def als_tFB_stats_bined():
	mask = ~np.isnan(uit_tFB) & ~np.isnan(als_filt)  # Mask for valid data points
	uit_tFB_m = uit_tFB[mask]
	als_fr_m = als_filt[mask]
 
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	print(f"{'Bin':<10} {'Bias':>8} {'RMSE':>8} {'CC':>6} {'DISO':>8} {'Mean_UiT':>10} {'Mean_ALS':>10}")
	print("-" * 60)

	for i in range(len(bins)-1):
		lo, hi = bins[i], bins[i+1]
		label = labels[i]

		# Create a mask for values falling within this bin based on UiT tFB values
		bin_mask = (uit_tFB_m >= lo) & (uit_tFB_m < hi)
		uit_bin = uit_tFB_m[bin_mask]
		als_bin = als_fr_m[bin_mask]

		if len(uit_bin) < 2 or len(als_bin) < 2:
			continue  # Skip bins with too little data

		bias = np.mean(uit_bin - als_bin)
		rmse = np.sqrt(np.mean((uit_bin - als_bin) ** 2))
		cc = np.corrcoef(uit_bin.ravel(), als_bin.ravel())[0, 1]

		mean_uit = np.mean(uit_bin)
		mean_als = np.mean(als_bin)

		ref_range = uit_bin.max() - uit_bin.min()
		span = ref_range if ref_range != 0 else 1.0

		Nbias = np.abs(bias) / span
		Nrmse = rmse / span
		DISO = np.sqrt(Nbias**2 + Nrmse**2 + (cc - 1.0)**2)

		print(f"{label:<10} {bias:8.3f} {rmse:8.3f} {cc:6.3f} {DISO:8.3f} {mean_uit:10.3f} {mean_als:10.3f}")

smos_s_x_f, smos_s_y_f = reprojecting(smos_s_lon_filt, smos_s_lat_filt)

als_24_unc = np.ones_like(als_24_sit)
als_24_sit_r = resample_to_cryo_grid_1(als_24_x, als_24_y, als_24_sit, als_24_unc, smos_s_x_f.flatten(), smos_s_y_f.flatten(), radius=6250) #938

def als_sit_stats():
	nan_mask = ~np.isnan(smos_als_overlap.flatten()) & ~np.isnan(als_24_sit_r)  # Mask for valid data points
	smos_als_overlap_m = smos_als_overlap.flatten()[nan_mask]
	als_24_sit_rm = als_24_sit_r[nan_mask]
 
	range_mask = (als_24_sit_rm >= 0) & (als_24_sit_rm <= 1) & (smos_als_overlap_m >= 0) & (smos_als_overlap_m <= 1)
	smos_als_overlap_m = smos_als_overlap_m[range_mask]
	als_24_sit_rm = als_24_sit_rm[range_mask]
	
	bias = np.mean(smos_als_overlap_m - als_24_sit_rm)
	rmse = np.sqrt(np.mean((smos_als_overlap_m - als_24_sit_rm) ** 2))
	cc = np.corrcoef(smos_als_overlap_m.ravel(), als_24_sit_rm.ravel())[0, 1]
	mean_smos = np.mean(smos_als_overlap_m)
	mean_als_24 = np.mean(als_24_sit_rm)
	

	# Normalize using range of reference data
	ref_range = smos_als_overlap_m.max() - smos_als_overlap_m.min()
	span = ref_range if ref_range != 0 else 1.0

	Nbias = np.abs(bias) / span
	Nrmse = rmse / span

	# DISO: deviation from ideal stats (bias=0, rmse=0, cc=1)
	DISO = np.sqrt(Nbias**2 + Nrmse**2 + (cc - 1.0)**2)

	print(f"SMOS: Mean={mean_smos:.3f}, Bias={bias:.3f}, RMSE={rmse:.3f}, CC={cc:.3f}, DISO={DISO:.3f}")
	print(f"ALS: Mean={mean_als_24:.3f}")


# ---------------------------- #
# ALS SIT vs HEM SIT
# ---------------------------- #

als_all_SIT = als_all_freeboard * 5.5

def als_hem_track():
	
	fig = plt.figure(figsize=(6.733, 5))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([5.0e5, 7.45e5, -1.322e6, -9.934e5], ccrs.NorthPolarStereo())
 
	sc_hem = ax.scatter(bird_lon, bird_lat, c=bird_sit, cmap="plasma", s=60, transform=ccrs.PlateCarree(), vmin=0, vmax=2.5, zorder=5)
	sc_als = ax.scatter(als_all_lon, als_all_lat, c=als_all_SIT, cmap="plasma", s=20, transform=ccrs.PlateCarree(), vmin=0, vmax=2.5, zorder=6)
 
	plt.plot(als_all_lon, als_all_lat, color="gray", linewidth=1, transform=ccrs.PlateCarree(), zorder=7)

	plt.colorbar(sc_hem, ax=ax, orientation="vertical", pad=0.02)
	plt.show()
 
def hem_als_track():
	# 1. Build KDTree from Bird coordinates
	hem_coords = np.column_stack([bird_x.ravel(), bird_y.ravel()])
	tree = cKDTree(hem_coords)

	# 2. Filter valid ALS points (no NaNs)
	als_coords = np.column_stack([als_all_x, als_all_y])
	valid_mask = np.all(np.isfinite(als_coords), axis=1) & np.isfinite(als_all_SIT)
	
	als_coords_valid = als_coords[valid_mask]
	als_sit_valid = als_all_SIT[valid_mask]

	# 3. Query: for each valid ALS point, find nearest Bird point
	_, idx = tree.query(als_coords_valid, k=1)

	# 4. Get matched Bird SIT values
	matched_bird_sit = bird_sit.ravel()[idx]  # shape matches filtered ALS points

	# 5. Return matched pairs
	return als_sit_valid, matched_bird_sit

matched_als_sit, matched_bird_sit = hem_als_track()

def hem_als_track_hist():
	w_hem_sit = np.ones_like(matched_bird_sit) / len(matched_bird_sit)
	w_als_sit = np.ones_like(matched_als_sit) / len(matched_als_sit)

	bin_edges = np.linspace(-1, 3.5, 21)
 
	fig = plt.figure(figsize=(6.733, 3.7))
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

	ax0 = fig.add_subplot(gs[0])
	ax1 = fig.add_subplot(gs[1], projection=ccrs.NorthPolarStereo())

	ax0.hist(matched_bird_sit, weights=w_hem_sit, histtype="step", fill=False, bins=bin_edges, color='black', alpha=0.7, label='HEM', density=False, linewidth=1.7, zorder=0)
	ax0.hist(matched_als_sit, weights=w_als_sit, histtype="step", fill=False, bins=bin_edges, color='blueviolet', label='ALS', density=False, linewidth=1.7, zorder=1)

	ax0.legend(loc='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=2, borderaxespad=0.0, handletextpad=0.3)
	ax0.set_xlabel("SIT [m]")
	ax0.set_ylabel("Observation (%)")
	ax0.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
	ax0.tick_params(axis='both', direction='in')

	ax1.set_extent([5.0e5, 7.45e5, -1.322e6, -9.934e5], crs=ccrs.NorthPolarStereo())
	ax1.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=5)
	ax1.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax1.add_feature(cfeature.COASTLINE, color="black", linewidth=0.5, zorder=6)
	ax1.gridlines(draw_labels=False, linewidth=0.5, color="dimgray", zorder=7)

	sc_hem = ax1.scatter(bird_lon, bird_lat, c=bird_sit, cmap="plasma", s=60, transform=ccrs.PlateCarree(), vmin=0, vmax=2.5, zorder=2)
	sc_als = ax1.scatter(als_all_lon, als_all_lat, c=als_all_SIT, cmap="plasma", s=20, transform=ccrs.PlateCarree(), vmin=0, vmax=2.5, zorder=3)
 
	ax1.plot(als_all_lon, als_all_lat, color="gray", linewidth=1, transform=ccrs.PlateCarree(), zorder=4, label="ALS path")

	cbar = fig.colorbar(sc_hem, ax=ax1, orientation="vertical", pad=0.02)
	cbar.ax.yaxis.set_tick_params(direction='in')
	cbar.set_label("SIT [m]")
 
	ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.1), frameon=False, ncol=1, borderaxespad=0.0, handletextpad=0.3)

	plt.subplots_adjust(left=0.093, right=0.964, wspace=0.0, bottom=0.105, top=0.933)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\SMOSice validations - CryoSat and SMOS\hem_als_track_hist.png", dpi=300, bbox_inches='tight')
	plt.show()

if __name__ == "__main__":
	#uit_data = get_UiT(uit_L3)
	#smos_data = get_smos(smos_L3)
	#get_smos_single(smos_L3_single)

	plot_EM_bird(em_bird_data)
	#plot_als(als_all_data)
	##bird_als_paths(als_data, em_bird_data)
	#plot_L2(uit_L2)

# ------ HEM system ------ #
	#map_smos_uit_bird_raw()
	#raw_distributions()

	#map_smos_uit_bird()
	#distribution()

# Statistical metrics
	#hem_stats()
	#hem_stats_bined()

# ------ ALS ------ #
# CS
	#plot_als_swath_xy(als_x, als_y, als_freeboard) # on hold
	#plot_uit_als_track()
	#hist()
	#plot_track_hist()

# SMOS
	#plot_als_res()
	#smos_als_track_hist()

# Statistical metrics
	#als_tFB_stats()
	#als_tFB_stats_bined()
	#als_sit_stats()

# ------ ALS SIT vs HEM SIT ------ #
	#als_hem_track()
	#hem_als_track_hist()