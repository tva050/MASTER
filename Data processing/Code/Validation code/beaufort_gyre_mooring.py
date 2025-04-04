import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.spatial import cKDTree

""" 
Plan 
 - Extract the files for corresponding mooring,....................................................✔️
	- these are represtented in the file name as "**a_**.mat", "**b_**.mat" and "**c_**.mat".......✔️
 - create an single data set for each mooring (a, b, d), which includes IDS and dates..............✔️
	- Filter out data which not covers the month 01, 02, 03, 04, 10, 11, and 12
	- These are the only valid months for Cryosat and SMOS
 - IDS: daily ice draft statistics: number, mean, std, minimum, maximum, median....................✔️
 - Extract, satellite data SMOS and Cryo...........................................................✔️
 - Reproject the coordinates to polarstereo........................................................✔️
 - Grid mooring and smos satellite to the same grid (Cryo grid, 25km)?.............................✖️
	- i dont think this is necessary, as the we going to use the mooring to look 
 	  for satellite data wihtin the search radius
 - Calculate referance monthly mean ice draft thickness from the moorings..........................✔️
	- Calculate the mean, std, min, max, and median for each month
 - Identify the satellite grid cells within the search radius of moorings
 - Satellite grid cells within the search radius is averaged 
 - Convert the derived sea ice freeboard to draft (possible earlier)...............................✔️
	- sid = sit - ifb 
 - Plot time series
	- Ranging from 0.1 - 1m draft 
	
Note:
Variables in the .mat file: ['BETA', 'ID', 'IDS', 'WLS', 'TILTS', 'OWBETA', 'BTBETA', 'WL', 'T', 'yday', 'dates', 'name']
"""

mooring_folder = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data converted"
cs_uit_folder = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product"
smos_folder = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\SMOS\All years\SMOS_monthly"

MOORING_A_LAT = 75
MOORING_A_LON = -150
MOORING_B_LAT = 78
MOORING_B_LON = -150
MOORING_D_LAT = 74
MOORING_D_LON = -140

RADIUS_RANGE = 200e3


def get_mooring_data(folder_path):
	mooring_files_A = glob.glob(os.path.join(folder_path, "*a_*.mat"))
	mooring_files_B = glob.glob(os.path.join(folder_path, "*b_*.mat"))
	mooring_files_D = glob.glob(os.path.join(folder_path, "*d_*.mat"))
	
	ids_A, ids_B, ids_D = [], [], []
	dates_A, dates_B, dates_D = [], [], []
   
	# handles mooring A
	for files in mooring_files_A:
		data = scipy.io.loadmat(files)
		dates_A.append(data["dates"])
		ids_a = data["IDS"]
		ids_A.append(ids_a[:, :6])
	
	for files in mooring_files_B:
		data = scipy.io.loadmat(files)
		dates_B.append(data["dates"])
		ids_b = data["IDS"]
		ids_B.append(ids_b[:, :6])
  
	for files in mooring_files_D:
		data = scipy.io.loadmat(files)
		dates_D.append(data["dates"])
		ids_d = data["IDS"]
		ids_D.append(ids_d[:, :6])
  
	ids_A = np.concatenate(ids_A, axis=0)
	ids_B = np.concatenate(ids_B, axis=0)
	ids_D = np.concatenate(ids_D, axis=0)
 
	dates_A = np.concatenate(dates_A, axis=0)
	dates_B = np.concatenate(dates_B, axis=0)
	dates_D = np.concatenate(dates_D, axis=0)
	
	return ids_A, ids_B, ids_D, dates_A, dates_B, dates_D

ids_A, ids_B, ids_D, dates_A, dates_B, dates_D = get_mooring_data(mooring_folder)

def filter_valid_dates(dates, ids):
	valid_months = {1, 2, 3, 4, 10, 11, 12}
	months = np.array([int(date.split("-")[1]) for date in dates])
	valid_indices = np.isin(months, list(valid_months))
	return ids[valid_indices], dates[valid_indices]
 
ids_A, dates_A = filter_valid_dates(dates_A, ids_A)
ids_B, dates_B = filter_valid_dates(dates_B, ids_B)
ids_D, dates_D = filter_valid_dates(dates_D, ids_D)

def monthly_statistics(ids, dates):
	""" 
	- Calculates monthly statistics for sea ice draft from mooring data.
	- Computes mean, standard deviation, minimum, maximum, and median for each month.
	- with the corresponding dates. in the format YYYY-MM
	"""
	dates = pd.to_datetime(dates)
	
	daily_mean = ids[:,1]
	df = pd.DataFrame({"date": dates, "mean_draft": daily_mean})
	
	monthly_stats = df.groupby(df["date"].dt.to_period("M")).agg(
		mean_draft=("mean_draft", "mean"),
		std_draft=("mean_draft", "std"),
		min_draft=("mean_draft", "min"),
		max_draft=("mean_draft", "max"),
		median_draft=("mean_draft", "median")
	).reset_index()
	monthly_stats["date"] = monthly_stats["date"].astype(str)
	
	return monthly_stats

monthly_stats_A = monthly_statistics(ids_A, dates_A)
monthly_stats_B = monthly_statistics(ids_B, dates_B)
monthly_stats_D = monthly_statistics(ids_D, dates_D)

def get_cyro(folder_path):
	""" 
	Reads CryoSat-2 data from NetCDF files in the specified folder and extracts relevant 
	information into a pandas DataFrame for easier time-series analysis.

	Parameters
	----------
	folder_path : str
		The path to the folder containing CryoSat-2 NetCDF files.

	Returns
	-------
	pd.DataFrame
		A DataFrame where each row corresponds to a unique timestamp and contains the following columns:
		- **date** (str): Date in "YYYY-MM" format.
		- **latitude** (numpy.ndarray): 2D array of latitude values.
		- **longitude** (numpy.ndarray): 2D array of longitude values.
		- **sea_ice_thickness** (numpy.ndarray): 2D array of sea ice thickness values.
		- **sea_ice_freeboard** (numpy.ndarray): 2D array of sea ice freeboard values.
		- **sea_ice_draft** (numpy.ndarray): 2D array of sea ice draft values (calculated as thickness - freeboard).

	Notes
	-----
	- The function expects NetCDF files to have `Year` and `Month` attributes.
	- Files missing date attributes are skipped with a warning message.
	- Invalid data points (NaN values) in the sea ice thickness or freeboard are masked 
	  and replaced with NaN in the output arrays.
	- The sea ice draft is calculated as: `sea_ice_draft = sea_ice_thickness - sea_ice_freeboard`.

	Examples
	--------
	>>> cryosat_df = get_cyro("/path/to/cryosat/data")
	>>> print(cryosat_df.head())  # Displays first few rows of the dataset
	"""
	cryosat_files = glob.glob(os.path.join(folder_path, "*.nc"))
	data_list = []

	for file in cryosat_files:
		with xr.open_dataset(file) as ds:
			year = ds.attrs.get("Year", None)
			month = ds.attrs.get("Month", None)

			if year is None or month is None:
				print(f"Skipping file {file} due to missing date info")
				continue

			date_key = f"{year}-{str(month).zfill(2)}"

			# Extract relevant data
			lat = ds["latitude"].values
			lon = ds["longitude"].values
			sit = ds["sea_ice_thickness"].values
			ifb = ds["sea_ice_freeboard"].values
			
			# Mask invalid data
			mask = ~np.isnan(sit)

			# Calculate sea ice draft
			sid = sit - ifb

			#lat = lat[mask]
			#lon = lon[mask]
			#sit = sit[mask]
			#ifb = ifb[mask]
			#sid = sid[mask]
   
   
			# Append data as a dictionary for DataFrame conversion
			data_list.append({
				"date": date_key,
				"latitude": lat,
				"longitude": lon,
				"sea_ice_thickness": sit,
				"sea_ice_freeboard": ifb,
				"sea_ice_draft": sid
			})

	# Convert to pandas DataFrame
	cryosat_df = pd.DataFrame(data_list)

	# Convert date column to datetime format for time-series operations
	cryosat_df["date"] = pd.to_datetime(cryosat_df["date"])

	return cryosat_df

def get_smos(folder_path):
	"""
	Reads SMOS (Soil Moisture and Ocean Salinity) data from NetCDF files in the specified folder 
	and extracts relevant information into a pandas DataFrame for easier time-series analysis.

	Parameters
	----------
	folder_path : str
		The path to the folder containing SMOS NetCDF files.

	Returns
	-------
	pd.DataFrame
		A DataFrame where each row corresponds to a unique timestamp and contains the following columns:
		- **date** (str): Date in "YYYY-MM-DD" format.
		- **latitude** (numpy.ndarray): 2D array of latitude values.
		- **longitude** (numpy.ndarray): 2D array of longitude values.
		- **sea_ice_thickness** (numpy.ndarray): 2D array of sea ice thickness values.
		- **sea_ice_draft** (numpy.ndarray): 2D array of sea ice draft values.

	Notes
	-----
	- The function expects NetCDF files to have a `date` attribute in the format `"YYYY-MM-DD"`.
	- If a file lacks the `date` attribute, it is skipped with a warning message.
	- The extracted data is assumed to be stored under the variable names `"mean_ice_thickness"` 
	  and `"sea_ice_draft"` within the NetCDF file.

	Examples
	--------
	>>> smos_df = get_smos("/path/to/smos/data")
	>>> print(smos_df.head())  # Displays first few rows of the dataset
	"""
	smos_files = glob.glob(os.path.join(folder_path, "*.nc"))
	data_list = []

	for file in smos_files:
		with xr.open_dataset(file) as ds:
			date = ds.attrs.get("date", None)

			if date is None:
				print(f"Skipping file {file} due to missing date info")
				continue

			lat = ds["latitude"].values
			lon = ds["longitude"].values
			sit = ds["mean_ice_thickness"].values
			sit_un = ds["uncertainty"].values
			sid = ds["sea_ice_draft"].values
			
			#mask = ~np.isnan(sit)
			#lat = lat[mask]
			#lon = lon[mask]
			#sit = sit[mask]
			#sit_un = sit_un[mask]
			#sid = sid[mask]
   
			# mask out values == 0.0
			#mask = sit == 0.0
			#lat = lat[~mask]
			#lon = lon[~mask]
			#sit = sit[~mask]
			#sit_un = sit_un[~mask]
			#sid = sid[~mask]

			# Append data as a dictionary (structured for pandas)
			data_list.append({
				"date": date,
				"latitude": lat,
				"longitude": lon,
				"sea_ice_thickness": sit,
				"uncertainty": sit_un,
				"sea_ice_draft": sid
			})

	# Convert list of dictionaries to a DataFrame
	smos_df = pd.DataFrame(data_list)

	# Convert date column to datetime format for time-series operations
	smos_df["date"] = pd.to_datetime(smos_df["date"])

	return smos_df
   
cryosat_df = get_cyro(cs_uit_folder)
smos_df = get_smos(smos_folder)

def reprojecting(lon, lat, proj=ccrs.NorthPolarStereo()):
	"""
	Reprojects given latitude and longitude coordinates to the specified projection.
	
	Parameters
	----------
	lon : float, list, or numpy.ndarray
		Longitude values (single value, list, or 2D NumPy array).
	lat : float, list, or numpy.ndarray
		Latitude values (single value, list, or 2D NumPy array).
	proj : cartopy.crs.Projection, optional
		The target projection. Default is North Polar Stereographic.
	
	Returns
	-------
	x, y : numpy.ndarray
		Transformed x and y coordinates in the new projection.
	"""
	# Convert inputs to numpy arrays (ensures compatibility for all cases)
	lon = np.array(lon)
	lat = np.array(lat)

	# Flatten any multi-dimensional input to avoid shape errors
	lon_flat = lon.flatten()
	lat_flat = lat.flatten()
	# Perform the coordinate transformation
	transformer = proj.transform_points(ccrs.PlateCarree(), lon_flat, lat_flat)
	
	# Extract transformed x, y coordinates
	x_flat = transformer[:, 0]
	y_flat = transformer[:, 1]
	x = x_flat.reshape(lon.shape)
	y = y_flat.reshape(lat.shape)

	return x, y


# project the mooring coordinates to polarstereo
mooring_a_x, mooring_a_y = reprojecting(MOORING_A_LON, MOORING_A_LAT)
mooring_b_x, mooring_b_y = reprojecting(MOORING_B_LON, MOORING_B_LAT)
mooring_d_x, mooring_d_y = reprojecting(MOORING_D_LON, MOORING_D_LAT)


# project the all satellite coordinates to polarstereo
lon_cs = np.array([np.array(l) for l in cryosat_df["longitude"].values])
lat_cs = np.array([np.array(l) for l in cryosat_df["latitude"].values])
lon_cs = np.vstack(lon_cs)
lat_cs = np.vstack(lat_cs)
cryosat_x, cryosat_y = reprojecting(lon_cs, lat_cs)

lon_smos = np.array([np.array(l) for l in smos_df["longitude"].values])
lat_smos = np.array([np.array(l) for l in smos_df["latitude"].values])
lon_smos = np.vstack(lon_smos)
lat_smos = np.vstack(lat_smos)
smos_x, smos_y = reprojecting(lon_smos, lat_smos)

print("cryosat_x shape:", cryosat_x.shape, "cryosat_y shape:", cryosat_y.shape)
print("cryo_sid shape:", cryosat_df["sea_ice_draft"].shape)
print("cryosat_df shape:", cryosat_df.shape)
print("smos_x shape:", smos_x.shape, "smos_y shape:", smos_y.shape)
print("smos_sid shape:", smos_df["sea_ice_draft"].shape)
print("smos_df shape:", smos_df.shape)


def indentify_cells(x_source, y_source, sid, x_ref, y_ref, search_radius = RADIUS_RANGE):
	"""
	Identify cells in the reference grid that are within a specified search radius of the source points.
	- calculate the mean sea ice draft within the search radius for the satellite products. 
	- median, and standard deviation of the sea ice draft within the search radius for the mooring data.
	"""
	pass

def rangesearch(coords, mooring_coords, radius):
	dist = np.linalg.norm(coords - mooring_coords, axis=1)
	return np.where(dist <= radius)[0]  # Return indices of points within the radius


mooring_A_coords = np.array([mooring_a_x, mooring_a_y])
mooring_B_coords = np.array([mooring_b_x, mooring_b_y])
mooring_D_coords = np.array([mooring_d_x, mooring_d_y])

grid_coords_cryo = np.vstack((cryosat_x.ravel(), cryosat_y.ravel())).T

IDX_A = rangesearch(grid_coords_cryo, mooring_A_coords, RADIUS_RANGE)
IDX_B = rangesearch(grid_coords_cryo, mooring_B_coords, RADIUS_RANGE)
IDX_D = rangesearch(grid_coords_cryo, mooring_D_coords, RADIUS_RANGE)

cryosat_sid = np.vstack(cryosat_df["sea_ice_draft"].values)
smos_sid = np.vstack(smos_df["sea_ice_draft"].values)
print("New cryosat_sid shape:", cryosat_sid.shape) 
print(f"Min index in IDX_A: {np.min(IDX_A)}")
print(f"Max index in IDX_A: {np.max(IDX_A)}")
print(f"Size of cryo_sid: {cryosat_sid.size}")

def calc_draft(draft_values, search_indices):
	valid_indices = np.array(search_indices)
	valid_indices = valid_indices[valid_indices < draft_values.size]  # Ensure valid indices
	
	if len(valid_indices) == 0:  # Handle case where no valid indices exist
		return np.nan
	
	return np.nanmean(np.take(draft_values, valid_indices))

def calc_monthly_draft(df, draft_values, search_indices):
    df = df.iloc[search_indices]  # Filter only relevant indices
    df = df.dropna(subset=["sea_ice_draft"])  # Remove NaN values
    monthly_draft = df.groupby(df["date"].dt.to_period("M"))["sea_ice_draft"].mean()
    return monthly_draft

monthly_draft_A_cryo = calc_monthly_draft(cryosat_df, cryosat_sid, IDX_A)
monthly_draft_B_cryo = calc_monthly_draft(cryosat_df, cryosat_sid, IDX_B)
monthly_draft_D_cryo = calc_monthly_draft(cryosat_df, cryosat_sid, IDX_D)

# Compute monthly mean sea ice draft for SMOS at each mooring
monthly_draft_A_smos = calc_monthly_draft(smos_df, smos_sid, IDX_A)
monthly_draft_B_smos = calc_monthly_draft(smos_df, smos_sid, IDX_B)
monthly_draft_D_smos = calc_monthly_draft(smos_df, smos_sid, IDX_D)

# Calculate the mean sea ice draft for CryoSat-2 for mooring A, B, and D
#mean_draft_A_cryo = calc_draft(cryosat_sid, IDX_A)
#mean_draft_B_cryo = calc_draft(cryosat_sid, IDX_B)
#mean_draft_D_cryo = calc_draft(cryosat_sid, IDX_D)
#
#mean_draft_A_smos = calc_draft(smos_sid, IDX_A)
#mean_draft_B_smos = calc_draft(smos_sid, IDX_B)
#mean_draft_D_smos = calc_draft(smos_sid, IDX_D)


def time_series_A():
    # getting draft at mooring A
	cryo_draft_A = mean_draft_A_cryo
	smos_draft_A = mean_draft_A_smos
	mooring_draft_A = monthly_stats_A["mean_draft"].values
	pass



#time_series_A()

""" def plot_sea_ice_draft(cryosat_df):
	# Extract median sea ice draft for each time step (assuming it's a 2D array)
	cryosat_df["sid_median"] = cryosat_df["sea_ice_draft"].apply(lambda x: np.nanmean(x))

	# Sort by date
	cryosat_df = cryosat_df.sort_values(by="date")

	# Plot
	plt.figure(figsize=(12, 6))
	plt.plot(cryosat_df["date"], cryosat_df["sid_median"], marker="o", linestyle="-", color="b", label="Median Sea Ice Draft")
	
	plt.xlabel("Date")
	plt.ylabel("Sea Ice Draft (m)")
	plt.title("Time Series of Sea Ice Draft from CryoSat-2")
	plt.xticks(rotation=45)
	plt.grid(True)
	plt.legend()
	plt.show() """
	
#plot_sea_ice_draft(cryosat_df)