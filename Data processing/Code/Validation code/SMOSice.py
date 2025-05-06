import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Circle


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


plt.rcParams.update({
		'font.family':      'serif',
		'font.size':         10,
		'axes.labelsize':    10,
		'xtick.labelsize':   8,
		'ytick.labelsize':   8,
		'legend.fontsize':   10,
		'figure.titlesize':  10,
}) 
 

def plot_smosice_data(EM_bird_data, ALS_data, title="SMOSice Data"):
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	#ax.set_extent([10, 30, 75, 81], crs=ccrs.PlateCarree())
	ax.set_extent([1.7e5, 7.45e5, -1.52e6, -8.39e5], ccrs.NorthPolarStereo())

	ax.coastlines()
	ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=2)
	ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.RIVERS, edgecolor='lightgray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=4)
	ax.add_feature(cfeature.COASTLINE, color="black", linewidth=0.1, zorder=5)
	ax.gridlines(draw_labels=True, color="dimgray", zorder=7)

	unique_days_EM = EM_bird_data["DateTime"].dt.strftime('%Y-%m-%d').unique()
	cmap_EM = plt.get_cmap("plasma", len(unique_days_EM))
	for i, day in enumerate(unique_days_EM):
		group = EM_bird_data[EM_bird_data["DateTime"].dt.strftime('%Y-%m-%d') == day]
		ax.scatter(
			group["Longitude"], group["Latitude"],
			color=cmap_EM(i), s=1, transform=ccrs.PlateCarree(),
			label=day, zorder=6
		)

	#unique_files = ALS_data["SourceFile"].unique()
	#cmap_ALS = plt.get_cmap("viridis", len(unique_files))
	#for i, file in enumerate(unique_files):
	#	group = ALS_data[ALS_data["SourceFile"] == file]
	#	ax.scatter(
	#		group["Longitude"], group["Latitude"],
	#		color=cmap_ALS(i), s=1, transform=ccrs.PlateCarree(), zorder=6)

	#ax.scatter(EM_bird_data["Longitude"], EM_bird_data["Latitude"], color="blue", s=1, transform=ccrs.PlateCarree(), label="EM-Bird Data", zorder=6)
	#ax.scatter(ALS_data["Longitude"], ALS_data["Latitude"], color="red", s=1, transform=ccrs.PlateCarree(), label="ALS Data", zorder=6)
	#plt.legend(markerscale=5, loc='lower left')
	plt.show()
	
em_bird_data = read_all_EMbird_files(EM_bird_path)
als_data = read_all_als_files(ALS_path)
plot_smosice_data(em_bird_data, als_data, title="SMOSice EM-Bird Data")