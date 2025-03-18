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

	return source_interp #.reshape(target_lat.shape)

all_x, all_y, all_thickness = extract_all_oib(oib_paths)
target_lat, target_lon = load_oib_lat_lon(oib_paths)

lat_cryo, lon_cryo, cryo_si = get_cryo(cryo_path)
lat_smos, lon_smos, smos_si = get_smos(smos_path)

#cryo_interp = interpolate_data(lat_cryo, lon_cryo, cryo_si, target_lat, target_lon)
smos_interp = interpolate_data(lat_smos, lon_smos, smos_si, target_lat, target_lon)

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
	#valid_mask = ~np.isnan(oib_flat) & ~np.isnan(smos_flat) & ~np.isnan(cryo_flat)
	#oib_flat, smos_flat, cryo_flat = oib_flat[valid_mask], smos_flat[valid_mask], cryo_flat[valid_mask]

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


multiple_box_plot_oib(all_thickness, smos_interp, cryo_si)