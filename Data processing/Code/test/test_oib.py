import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from  scipy.interpolate import griddata
import pandas as pd
from pyproj import Proj, Transformer, transform
import cartopy.crs as ccrs
import seaborn as sns
from mpl_toolkits.basemap import Basemap


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

plot_data_oib(oib_lon, oib_lat, oib_sit)

def plot_cryo_oib(oib_lat, oib_lon, oib_sit, cs_lat, cs_lon, cs_sit):
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

# https://nsidc.org/data/user-resources/help-center/guide-nsidcs-polar-stereographic-projection
def reproject_to_epsg3413(lon, lat):
	"""Convert lat/lon to EPSG:3413 (meters)."""
	proj_latlon = Proj(proj="latlong", datum="WGS84")
	proj_stereo = Proj("epsg:3413")
	transformer = Transformer.from_proj(proj_latlon, proj_stereo, always_xy=True)
	x, y = transformer.transform(lon, lat)
	return x, y

x_oib, y_oib = reproject_to_epsg3413(oib_lon, oib_lat)
x_cryo, y_cryo = reproject_to_epsg3413(cryo_lon, cryo_lat)
x_smos, y_smos = reproject_to_epsg3413(smos_lon, smos_lat)

# ----------------------- Interpolation Methods ----------------------- #

def interpolating_to_cryo(source_x, source_y, sit_data, target_x, target_y):
	"""Interpolate to CryoSat-2 grid."""
	source_x = source_x.flatten() 
	source_y = source_y.flatten()
	source_sit = sit_data.flatten()
	interpolated_sit = griddata((source_x, source_y), source_sit, (target_x, target_y), method='nearest') 
	return interpolated_sit  # Return interpolated values at target locations for sea ice thickness data 

def idw_interpolation_gpt(x_source, y_source, values_source, x_target, y_target, power=2, k=5):
	"""Perform Inverse Distance Weighting (IDW) interpolation.

	Args:
		x_source, y_source: Source coordinates (e.g., OIB)
		values_source: Source values (e.g., OIB thickness).
		x_target, y_target: Target coordinates (e.g., SMOS).
		power: Power parameter for IDW (default=2).
		k: Number of nearest neighbors to use (default=5).

	Returns:
		Interpolated values at target locations.
	"""
	# Remove NaN values from source data
	mask = ~np.isnan(values_source)
	x_source, y_source, values_source = x_source[mask], y_source[mask], values_source[mask]

	# Build KDTree from source points
	tree = cKDTree(np.c_[x_source, y_source])

	# Query the k nearest neighbors for target points
	dists, idxs = tree.query(np.c_[x_target, y_target], k=k)

	# Avoid division by zero (if distance is very small, set to a small value)
	dists = np.maximum(dists, 1e-10)

	# Compute IDW weights
	weights = 1.0 / dists**power
	weights /= weights.sum(axis=1, keepdims=True)  # Normalize weights

	# Compute weighted sum of values
	interpolated_values = np.sum(weights * values_source[idxs], axis=1)

	return interpolated_values

def idw_interpolation_uit(x_known, y_known, values_known, x_target, y_target, power=2, k=2):
	"""
	Perform Inverse Distance Weighting interpolation.
	
	Parameters:
	- x_known, y_known: Coordinates of known data points.
	- values_known: Values at known data points.
	- x_target, y_target: Coordinates of target points where interpolation is desired.
	- power: Power parameter for IDW (default is 2).
	
	Returns:
	- Interpolated values at target points.
	"""
	# Create KDTree for known points
	tree = cKDTree(np.c_[x_known, y_known])
	
	# Query the tree for nearest neighbors
	distances, indices = tree.query(np.c_[x_target, y_target], k=k)
	
	# Calculate weights using inverse distance
	weights = 1 / np.where(distances == 0, np.inf, distances**power)
	
	# Handle division by zero for weights
	weights_sum = np.sum(weights, axis=1)
	weights_sum[weights_sum == 0] = np.nan  # Avoid division by zero
	
	# Calculate interpolated values
	interpolated_values = np.sum(weights * values_known[indices], axis=1) / weights_sum
	return interpolated_values 

interpolated_oib_sit = interpolating_to_cryo(np.array(x_oib), np.array(y_oib), np.array(oib_sit), x_cryo, y_cryo)
interpolated_smos_sit = interpolating_to_cryo(x_smos, y_smos, smos_sit, x_cryo, y_cryo)
print("Interpolated OiB SIT shape:", interpolated_oib_sit.shape)

""" idw_interp_oib_sit = idw_interpolation_gpt(
	np.array(x_oib), np.array(y_oib), np.array(oib_sit),
	x_cryo.flatten(), y_cryo.flatten()
) """

""" idw_interp_smos_sit = idw_interpolation_gpt(
	x_smos, y_smos, smos_sit[0,:,:],
	x_cryo.flatten(), y_cryo.flatten()
) """

""" idw_interp_oib_sit_uit = idw_interpolation_uit(np.array(x_oib), np.array(y_oib), np.array(oib_sit), x_cryo.flatten(), y_cryo.flatten())
idw_interp_smos_sit_uit = idw_interpolation_uit(x_smos.flatten(), y_smos.flatten(), smos_sit[0,:,:].flatten(), x_cryo.flatten(), y_cryo.flatten()) """

def mapping_xy(x_cryo, y_cryo, x_source, y_source):
	# Check if x_cryo and y_cryo are 2D arrays and need to be flattened
	if x_source.ndim == 2 and y_source.ndim == 2:
		x_source = x_source.flatten()
		y_source = y_source.flatten()
	else: 
		pass
	
	cryo_tree = cKDTree(np.column_stack((x_cryo.flatten(), y_cryo.flatten())))

	mapped_source_coords = []
 
	for x, y in zip(x_source, y_source):
		dist, index = cryo_tree.query([x, y], k=1)
		mapped_source_coords.append((x_cryo.flatten()[index], y_cryo.flatten()[index]))
	
	return np.array(mapped_source_coords) # Return list of (x, y) for valid points

mapped_oib_coords = mapping_xy(x_cryo, y_cryo, np.array(x_oib), np.array(y_oib))
mapped_smos_coords = mapping_xy(x_cryo, y_cryo, x_smos, y_smos)

print("Mapped OIB coords shape: ", mapped_oib_coords.shape)
print("Mapped SMOS coords shape: ", mapped_smos_coords.shape)

x_oib_mapped = mapped_oib_coords[:, 0]
y_oib_mapped = mapped_oib_coords[:, 1]

print("Mapped OIB x shape: ", x_oib_mapped.shape)
print("Mapped OIB y shape: ", y_oib_mapped.shape)

x_smos_mapped = mapped_smos_coords[:, 0]
y_smos_mapped = mapped_smos_coords[:, 1]

print("Mapped SMOS x shape: ", x_smos_mapped.shape)
print("Mapped SMOS y shape: ", y_smos_mapped.shape)

# ----------------------- Interp Histograms ----------------------- #

def histogram(interp_oib, interp_smos, cryo_sit):
	# Flatten the data
	interp_oib = interp_oib.flatten()
	interp_smos = interp_smos.flatten()
	cryo_sit = cryo_sit.flatten()

	# Create histogram
	plt.hist(interp_oib, bins=100, alpha=0.5, color='green', label='Interpolated OIB Thickness')
	plt.hist(cryo_sit, bins=100, alpha=0.5, color='blue', label='CryoSat-2')
	plt.hist(interp_smos, bins=100, alpha=0.5, color='red', label='Interpolated SMOS')
	
	# Labels and title
	plt.xlabel("Sea Ice Thickness [m]")
	plt.ylabel("Frequency")
	plt.title("Histogram: Comparison of CryoSat-2 and SMOS Thickness with Interpolated OIB")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()
 
def idw_histogram(interp_oib, interp_smos, cryo_sit):
	interp_oib = interp_oib.flatten()
	interp_smos = interp_smos.flatten()
	cryo_sit = cryo_sit.flatten()
	
	# Create histogram
	plt.hist(interp_oib, bins=100, alpha=0.5, color='green', label='OIB')
	plt.hist(cryo_sit, bins=100, alpha=0.5, color='blue', label='CryoSat-2')
	plt.hist(interp_smos, bins=100, alpha=0.5, color='red', label='SMOS')
	
	# Labels and title
	plt.xlabel("Sea Ice Thickness [m]")
	plt.ylabel("Frequency")
	plt.title("Histogram: Comparison of CryoSat-2 with IDW Interpolated OIB and SMOS")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()
	
def idw_histogram_uit(interp_oib, interp_smos, cryo_sit):
	interp_oib = interp_oib.flatten()
	interp_smos = interp_smos.flatten()
	cryo_sit = cryo_sit.flatten()
	
	# Create histogram
	plt.hist(interp_oib, bins=100, alpha=0.5, color='green', label='OIB')
	plt.hist(cryo_sit, bins=100, alpha=0.5, color='blue', label='CryoSat-2')
	plt.hist(interp_smos, bins=100, alpha=0.5, color='red', label='SMOS')
	
	# Labels and title
	plt.xlabel("Sea Ice Thickness [m]")
	plt.ylabel("Frequency")
	plt.title("Histogram: Comparison of CryoSat-2 with IDW Interpolated OIB and SMOS")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()
	
# ---------------------------------------------- # 
""" 
The following functions finds the Cryosat-2 data and smos data grid points that are below 1m, 
for so making an 25km diameter circle around the grid point and then finds the OIB data points that are within the circle. 
Than averages the OiB data points within the circle, and if the average is below 1m it is stored else it is discarded.
	- Done by using cKDTree to find the nearest neighbors.
"""

 
def filter_and_average_oib_cryo(cryo_x, cryo_y, cryo_sit, oib_x, oib_y, oib_sit, radius=12500):
	"""
	Filters CryoSat-2 and SMOS data for points below 1m SIT, finds OIB points within 25 km, 
	and averages them if the mean is also below 1m.
	
	Parameters:
		cryo_x, cryo_y: CryoSat-2 grid coordinates (meters, Polar Stereographic).
		cryo_sit: CryoSat-2 sea ice thickness data.
		smos_sit: SMOS sea ice thickness data.
		oib_x, oib_y: OIB grid coordinates (meters, Polar Stereographic).
		oib_sit: OIB sea ice thickness data.
		radius: Search radius in meters (default: 25000m or 25km).
	
	Returns:
		List of (x, y, avg_OIB_SIT) for valid points.
	"""

	# 1. Identify CryoSat-2 & SMOS points with SIT < 1m
	mask = (cryo_sit < 1) 
	cryo_filtered_x = cryo_x[mask]
	cryo_filtered_y = cryo_y[mask]

	# 2. Build a KDTree using OIB coordinates
	oib_tree = cKDTree(np.column_stack((oib_x, oib_y)))
	
	results = []

	# 3. Find OIB points within the radius for each CryoSat-2 point
	for x, y in zip(cryo_filtered_x, cryo_filtered_y):
		indices = oib_tree.query_ball_point([x, y], radius)

		# 4. Ensure indices are within valid range
		valid_indices = [i for i in indices if i < len(oib_sit)]
		if valid_indices:
			nearby_oib_sit = oib_sit[valid_indices]
			avg_oib_sit = np.mean(nearby_oib_sit)

			# 5. Store only if avg OIB SIT < 1m
			if avg_oib_sit < 1:
				results.append((x, y, avg_oib_sit))

	return results # Return list of (x, y, avg_OIB_SIT) for valid points


def filter_and_average_oib_smos(smos_x, smos_y, smos_sit, oib_x, oib_y, oib_sit, radius=12500):
    """
    Filters SMOS data for points below 1m SIT, finds OIB points within 25 km, 
    and averages them if the mean OIB SIT is also below 1m.

    Parameters:
        smos_x, smos_y: SMOS grid coordinates (meters, Polar Stereographic).
        smos_sit: SMOS sea ice thickness data.
        oib_x, oib_y: OIB grid coordinates (meters, Polar Stereographic).
        oib_sit: OIB sea ice thickness data.
        radius: Search radius in meters (default: 12500m or 25km).

    Returns:
        List of (x, y, avg_OIB_SIT) for valid SMOS points.
    """

    # 1. Identify SMOS points with SIT < 1m
    mask = (smos_sit < 1)
    smos_filtered_x = smos_x[mask]
    smos_filtered_y = smos_y[mask]

    # 2. Build a KDTree using OIB coordinates
    oib_tree = cKDTree(np.column_stack((oib_x, oib_y)))

    results = []

    # 3. Find OIB points within the radius for each SMOS point
    for x, y in zip(smos_filtered_x, smos_filtered_y):
        indices = oib_tree.query_ball_point([x, y], radius)

        # 4. Extract valid OIB SIT values
        valid_indices = [i for i in indices if i < len(oib_sit)]
        if valid_indices:
            nearby_oib_sit = oib_sit[valid_indices]
            avg_oib_sit = np.mean(nearby_oib_sit)

            # 5. Store only if avg OIB SIT < 1m
            if avg_oib_sit < 1:
                results.append((x, y, avg_oib_sit))

    return results

interpolated_oib_sit = interpolated_oib_sit.flatten()
oib_cryo_filtered = filter_and_average_oib_cryo(x_cryo, y_cryo, cryo_sit, x_oib_mapped, y_oib_mapped, interpolated_oib_sit)
oib_smos_filtered = filter_and_average_oib_smos(x_smos_mapped, y_smos_mapped, interpolated_smos_sit, x_oib_mapped, y_oib_mapped, interpolated_oib_sit)


#print(filter_and_average_oib(x_cryo, y_cryo, cryo_sit, np.array(x_oib), np.array(y_oib), interpolated_oib_sit))
 
 
#plot_cryo_oib(oib_lat, oib_lon, oib_sit, cryo_lat, cryo_lon, cryo_sit)
#plot_smos_oib(oib_lat, oib_lon, oib_sit, smos_lat, smos_lon, smos_sit[0,:,:])

#histogram(interpolated_oib_sit, interpolated_smos_sit, cryo_sit)
#idw_histogram(idw_interp_oib_sit, idw_interp_smos_sit, cryo_sit)
#idw_histogram_uit(idw_interp_oib_sit_uit, idw_interp_smos_sit_uit, cryo_sit)

#plot_data_oib(oib_lon, oib_lat, oib_sit)