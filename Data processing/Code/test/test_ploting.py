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

smos_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\2013\2013_mean_thickness.nc"

cryo_path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\2013\uit_cryosat2_L3_EASE2_nh25km_2013_03_v3.nc"

def get_data_oib(path):
	df = pd.read_csv(path, dtype=str, low_memory=False)  # Read as strings to avoid mixed types
	df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, forcing errors to NaN

	# Extract required numerical columns
	lat = df["lat"].values
	lon = df["lon"].values
	thickness = df["thickness"].values

	# Apply mask to remove invalid data
	mask = (thickness != -99999.0000) & (thickness != 0.0)
	thickness = np.where(mask, thickness, np.nan)

	return lat, lon, thickness

def extract_all_oib(oib_paths):
	all_lat, all_lon, all_thickness = [], [], []
	for path in oib_paths:
		lat, lon, thickness = get_data_oib(path)


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
	si_thickness_un = data.variables['sea_ice_thickness_uncertainty'][:]
	
	# Check if lat and lon are 1D and need reshaping
	if lat.ndim == 1 and lon.ndim == 1:
		lon, lat = np.meshgrid(lon, lat)
		print('Reshaped lat and lon')
	# Mask invalid data
	mask = ~np.isnan(si_thickness)
	filtered_si_thickness = np.where(mask, si_thickness, np.nan)
	return lat, lon, filtered_si_thickness, si_thickness_un



oib_lat, oib_lon, oib_sit = extract_all_oib(oib_paths_2013)
oib_lat, oib_lon, oib_sit = np.array(oib_lat), np.array(oib_lon), np.array(oib_sit)
smos_lat, smos_lon, smos_sit = get_smos(smos_path)
cryo_lat, cryo_lon, cryo_sit, cryo_sit_un = get_cryo(cryo_path)

print('OIB:', oib_lat.shape, oib_lon.shape, oib_sit.shape)
print('SMOS:', smos_lat.shape, smos_lon.shape, smos_sit.shape)
print('Cryo:', cryo_lat.shape, cryo_lon.shape, cryo_sit.shape)

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

def find_grid_resolution(x, y):
    """ Find the grid resolution of the input data in meters """
    # Flatten the arrays to make them 1D
    x = x.ravel()
    y = y.ravel()

    # Calculate the distance between each point and its nearest neighbor
    tree = cKDTree(np.column_stack((x, y)))  # Use np.column_stack instead of zip
    distances, _ = tree.query(np.column_stack((x, y)), k=2)

    # Take the mean of the nearest neighbor distances
    grid_resolution = np.mean(distances[:, 1])
    return grid_resolution

#grid_res_oib = find_grid_resolution(x_oib, y_oib)
#grid_res_smos = find_grid_resolution(x_smos, y_smos)	
#grid_res_cryo = find_grid_resolution(x_cryo, y_cryo)

#print('Grid resolution OIB:', grid_res_oib)
#print('Grid resolution SMOS:', grid_res_smos)
#print('Grid resolution Cryo:', grid_res_cryo)

def grid_cells(lon, lat):
    # Define the projection (e.g., North Polar Stereographic)
	projection = ccrs.NorthPolarStereo()

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())  # Adjust extent to your region

	# Plot the grid as a mesh
	mesh = ax.pcolormesh(lon, lat, np.zeros_like(lon), transform=ccrs.PlateCarree(),
	                      edgecolor='gray', linewidth=0.5, alpha=0.5, shading='auto')

	# Add features
	ax.coastlines()
	ax.add_feature(cfeature.LAND, facecolor="lightgray")
	ax.add_feature(cfeature.BORDERS, linestyle=":")

	# Title
	ax.set_title("Grid Cells Visualization (25 km)")

	plt.show()

#grid_cells(cryo_lon, cryo_lat)


# Calculate the gridded mean of OiB by averaging the the referance data within the a 12.5 km radius of each generated grid. 
#     - The grid is generated by the CryoSat-2 data, which has a 25 km resolution.
#     - Data grids is are generated at 12.5 km intervals along the trajectory of the OiB data.


def grid_mean_oib(x_oib, y_oib, oib_sit, x_cryo, y_cryo, grid_res=25000, search_radius=12500):
    """
    Compute the gridded mean of OiB data by averaging the reference data within a 12.5 km radius of each CryoSat-2 grid cell.
    
    Parameters:
    x_oib, y_oib (arrays): OiB data coordinates (stereographic projection).
    oib_sit (array): OiB sea ice thickness values.
    x_cryo, y_cryo (arrays): CryoSat-2 grid coordinates.
    grid_res (int): Resolution of the CryoSat-2 grid in meters (default 25 km).
    search_radius (int): Radius within which OiB data is averaged (default 12.5 km).

    Returns:
    gridded_oib (array): Mean OiB sea ice thickness for each CryoSat-2 grid cell.
    """

    # Flatten the CryoSat-2 grid for easier processing
    x_cryo_flat = x_cryo.ravel()  # Flatten the x-coordinates
    y_cryo_flat = y_cryo.ravel()  # Flatten the y-coordinates
	
    # Create a KDTree for the OiB data points
    tree = cKDTree(np.column_stack((x_oib, y_oib)))  # Use the coordinates of the OiB data

    # Query the tree to find OiB points within 12.5 km radius for each CryoSat-2 grid point
    indices = tree.query_ball_point(np.column_stack((x_cryo_flat, y_cryo_flat)), search_radius)

    # Initialize grid for storing mean OiB SIT
    gridded_oib = np.full(x_cryo_flat.shape, np.nan)

    # Compute mean OiB SIT within the radius
    for i, inds in enumerate(indices):
        if len(inds) > 0:
            gridded_oib[i] = np.nanmean(oib_sit[inds])  # Compute mean ignoring NaN values

    # Reshape back to match CryoSat-2 grid dimensions (361, 361)
    gridded_oib = gridded_oib.reshape(x_cryo.shape)

    return gridded_oib

def grid_mean_smos(x_oib, y_oib, oib_sit, x_cryo, y_cryo, grid_res=25000, search_radius=12500):
    # Flatten the CryoSat-2 grid for easier processing
    x_cryo_flat = x_cryo.ravel()
    y_cryo_flat = y_cryo.ravel()

    # Create a KDTree for the OiB data points
    tree = cKDTree(np.column_stack((x_oib, y_oib)))

    # Query the tree to find OiB points within 12.5 km radius for each CryoSat-2 grid point
    indices = tree.query_ball_point(np.column_stack((x_cryo_flat, y_cryo_flat)), search_radius)

    # Initialize grid for storing mean OiB SIT (use np.empty to avoid read-only issues)
    gridded_oib = np.empty(x_cryo_flat.shape)
    gridded_oib.fill(np.nan)

    # Compute mean OiB SIT within the radius
    for i, inds in enumerate(indices):
        if len(inds) > 0:
            # Ensure there are valid data points before computing the mean
            valid_data = oib_sit[inds]
            if np.any(~np.isnan(valid_data)):  # Check if there are valid (non-NaN) values
                gridded_oib[i] = np.nanmean(valid_data)  # Compute mean ignoring NaN values
            else:
                gridded_oib[i] = np.nan  # If no valid data, leave as NaN
        else:
            gridded_oib[i] = np.nan  # If no points are found in the radius, set to NaN

    # Reshape back to match CryoSat-2 grid dimensions (361, 361)
    gridded_oib = gridded_oib.reshape(x_cryo.shape)

    return gridded_oib


# Compute the gridded mean OiB SIT
gridded_oib_sit = grid_mean_oib(x_oib, y_oib, oib_sit, x_cryo, y_cryo)
gridded_smos_sit = grid_mean_smos(x_smos.flatten(), y_smos.flatten(), smos_sit.flatten(), x_cryo, y_cryo)

# Print the new grid shape to verify it matches CryoSat-2
print("Gridded OiB SIT Shape:", gridded_oib_sit.shape)  # Expected (361, 361)

def plot_gridded_data(x_cryo, y_cryo, gridded_oib_sit):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())

    # Plot the gridded data
    scatter = ax.pcolormesh(x_cryo, y_cryo, gridded_oib_sit, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)
    

    # Add features
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    # Title and color bar
    ax.set_title("Gridded Mean OiB Sea Ice Thickness")
    plt.colorbar(scatter, label="Sea Ice Thickness (m)")
    plt.show()
    
    
def hist_cryo_smos_oib(cryo_sit, gridded_data):
	cryo_sit = cryo_sit.flatten()
	gridded_data = gridded_data.flatten()
	
	# Create histogram
	plt.hist(gridded_data, bins=100, alpha=0.5, color='green', label='OIB Thickness', density=True)
	plt.hist(cryo_sit, bins=100, alpha=0.5, color='blue', label='CryoSat-2', density=True)
	
	# Labels and title
	plt.xlabel("Sea Ice Thickness [m]")
	plt.ylabel("Frequency")
	plt.title("Histogram: Comparison of CryoSat-2 and SMOS Thickness with OIB RAW")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	plt.show()

hist_cryo_smos_oib(cryo_sit, gridded_smos_sit)
plot_gridded_data(cryo_lon, cryo_lat, gridded_smos_sit)


# Plot the grid
#plot_grid(xx, yy, z)
