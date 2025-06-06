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
import matplotlib.path as mpath
from matplotlib.ticker import PercentFormatter

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

 
def get_month_number(month, month_map):
	if month.isdigit():
		return int(month)
	return month_map.get(month, -1)  # Return -1 if the month is invalid

def filter_valid_dates(dates, ids):
	# Mapping from month abbreviations to numbers
	month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 
				 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
				 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
	
	valid_months = {1, 2, 3, 4, 10, 11, 12}
	
	# Convert month abbreviations or numbers to numeric months using the mapping
	months = np.array([get_month_number(date.split("-")[1], month_map) for date in dates])
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


def identify_cells_df(df, mooring_x, mooring_y, label, search_radius=RADIUS_RANGE):
	"""
	Identify satellite grid cells within a specified search radius of a mooring for each month,
	and compute statistics (mean, median, std, min, max) of sea ice draft for that month.
	
	This function assumes that each row in the DataFrame `df` contains:
	  - "longitude": a 2D array of longitude values,
	  - "latitude": a 2D array of latitude values,
	  - "sea_ice_draft": a 2D array of sea ice draft values.
	
	Parameters
	----------
	df : pd.DataFrame
		DataFrame where each row corresponds to a month (or time step) and contains
		the satellite grid data.
	mooring_x : float
		Mooring x coordinate (projected to the same coordinate system as the satellite grid).
	mooring_y : float
		Mooring y coordinate.
	search_radius : float, optional
		Search radius (in same units as x/y coordinates), default is 200e3.
		
	Returns
	-------
	pd.DataFrame
		A DataFrame with one row per time step containing:
		 - date
		 - mean_draft
		 - median_draft
		 - std_draft
		 - min_draft
		 - max_draft
		 - suspicious (1 if more than 25% of valid cells > 1m, else 0)
	"""
	results = []
	
	# Loop over each month (row) in the DataFrame
	for idx, row in df.iterrows():
		# Reproject the grid for this month
		# (if the row already contains reprojected coordinates, you can use them directly)
		x_grid, y_grid = reprojecting(row["longitude"], row["latitude"]) 
		
		# Flatten the grids to get a list of grid cell coordinates (n_points x 2)
		points = np.column_stack((x_grid.flatten(), y_grid.flatten()))
		tree = cKDTree(points)
		
		# Find indices of grid cells within the search radius of the mooring point
		indices = tree.query_ball_point([mooring_x, mooring_y], r=search_radius)
		
		# Extract the sea ice draft for this month and flatten
		sid_grid = row["sea_ice_draft"]
		sid_values = sid_grid.flatten()[indices]
		
		# Remove NaNs from the draft values
		sid_valid = sid_values[~np.isnan(sid_values)]
		#print(f"number of valid cells for {label}: {len(sid_valid)}")

		stats = {
			"date": row["date"],
			"mean_draft": np.nan,
			"median_draft": np.nan,
			"std_draft": np.nan,
			"min_draft": np.nan,
			"max_draft": np.nan,
			"suspicious": 0  # Default to not suspicious
		}
		
		if len(sid_valid) > 0:
			stats["mean_draft"] = np.mean(sid_valid)
			stats["median_draft"] = np.median(sid_valid)
			stats["std_draft"] = np.std(sid_valid)
			stats["min_draft"] = np.min(sid_valid)
			stats["max_draft"] = np.max(sid_valid)
			
   			# suspicious if more than 40% of valid cells are > 1m
			frac_above_1m = np.sum(sid_valid >= 1.0) / len(sid_valid)
			if frac_above_1m > 0.40: #and np.mean(sid_valid) > 0.8
				stats["suspicious"] = 1
		else:
			stats = {
				"mean_draft": np.nan,
				"median_draft": np.nan,
				"std_draft": np.nan,
				"min_draft": np.nan,
				"max_draft": np.nan,
				"suspicious": 0  # Default to not suspicious
			}
		
		stats["date"] = row["date"]  # assuming the date column exists in df
		results.append(stats)
		
	return pd.DataFrame(results)

cryosat_stats_A = identify_cells_df(cryosat_df, mooring_a_x, mooring_a_y, "")
cryosat_stats_B = identify_cells_df(cryosat_df, mooring_b_x, mooring_b_y, "")
cryosat_stats_D = identify_cells_df(cryosat_df, mooring_d_x, mooring_d_y, "")

smos_stats_A = identify_cells_df(smos_df, mooring_a_x, mooring_a_y, "SMOS A")
smos_stats_B = identify_cells_df(smos_df, mooring_b_x, mooring_b_y, "SMOS B")
smos_stats_D = identify_cells_df(smos_df, mooring_d_x, mooring_d_y, "SMOS D")

mooring_A_draft = monthly_stats_A["mean_draft"]
cryosat_A_draft = cryosat_stats_A["mean_draft"]
smos_A_draft = smos_stats_A["mean_draft"]
mooring_B_draft = monthly_stats_B["mean_draft"]
cryosat_B_draft = cryosat_stats_B["mean_draft"]
smos_B_draft = smos_stats_B["mean_draft"]
mooring_D_draft = monthly_stats_D["mean_draft"]
cryosat_D_draft = cryosat_stats_D["mean_draft"]
smos_D_draft = smos_stats_D["mean_draft"]

plt.rcParams.update({
		'font.family':      'serif',
		'font.size':         12,
		'axes.labelsize':    12,
		'xtick.labelsize':   11,
		'ytick.labelsize':   11,
		'legend.fontsize':   12,
		'figure.titlesize':  12,
})

 

def mooring_locations():
	fig = plt.figure(figsize=(6.733, 4.5))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], crs=ccrs.NorthPolarStereo())
 
	ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1, zorder=2)
	ax.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.5, zorder=1)
	ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=3)
	ax.add_feature(cfeature.RIVERS, edgecolor='lightgray', facecolor="white", linewidth=0.5, alpha=0.5, zorder=4)
	ax.add_feature(cfeature.COASTLINE, color = "black", linewidth=0.1, zorder=5)
	ax.gridlines(draw_labels=True, color="dimgray", zorder = 7)
 
	ax.scatter([MOORING_A_LON, MOORING_B_LON, MOORING_D_LON],
			   [MOORING_A_LAT, MOORING_B_LAT, MOORING_D_LAT],
			   s=100, marker='o', edgecolors='red', facecolors='none', linewidth=1,
			   transform=ccrs.PlateCarree())

	ax.scatter([MOORING_A_LON, MOORING_B_LON, MOORING_D_LON],
			   [MOORING_A_LAT, MOORING_B_LAT, MOORING_D_LAT],
			   s=30, marker='o', c='red', label='BGEP moorings',
			   transform=ccrs.PlateCarree())

	ax.text(-6.824e5, 1.511e6, 'A', color='black')
	ax.text(-5.208e5, 1.176e6, 'B', color='black')
	ax.text(-1.48e6, 1.424e6, 'D', color='black')

	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
 
	ax.set_boundary(circle, transform=ax.transAxes)
	plt.legend(loc="lower left", fontsize=11)
	#plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\BGEP validations - CryoSat and SMOS\mooring_locations.png", dpi=300, bbox_inches='tight')
	#plt.show()

def histogram_mooring():
	#print(monthly_stats_A.head())
	plt.figure(figsize=(6.733, 3.5))
	plt.hist(mooring_A_draft, bins=10, label='Mooring A SID', color='black', alpha=0.7, weights=np.ones_like(mooring_A_draft) / len(mooring_A_draft))
	plt.axvspan(0, 1, color='red', alpha=0.3, label='Area of interest (0-1 m)')
	plt.tick_params(axis='both', direction='in')
	plt.xlabel('Sea Ice Draft [m]')
	plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
	plt.ylabel('Observations (%)')
	plt.legend(loc='lower left', bbox_to_anchor=(0., 1.01, 1., .102), frameon=False, ncol=2, mode="expand", borderaxespad=0.0, handletextpad=0.3)
	plt.grid()
	#plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\BGEP validations - CryoSat and SMOS\Histogram_Mooring_A.png", dpi=300)
	#plt.show()

def mooring_draft_range(mooring_df, satellite_df):
	mask = (mooring_df["mean_draft"] >= 0) & (mooring_df["mean_draft"] <= 1)
	mooring_df_f = mooring_df[mask]
	valid_dates = mooring_df_f["date"].unique()
	satellite_df_f = satellite_df[satellite_df["date"].isin(valid_dates)]
	return mooring_df_f, satellite_df_f


def times_series_all():
	monthly_stats_A["date"] = pd.to_datetime(monthly_stats_A["date"])
	monthly_stats_B["date"] = pd.to_datetime(monthly_stats_B["date"])
	monthly_stats_D["date"] = pd.to_datetime(monthly_stats_D["date"])
	
	cryosat_stats_A["date"] = pd.to_datetime(cryosat_stats_A["date"])
	cryosat_stats_B["date"] = pd.to_datetime(cryosat_stats_B["date"])
	cryosat_stats_D["date"] = pd.to_datetime(cryosat_stats_D["date"])
 
	smos_stats_A["date"] = pd.to_datetime(smos_stats_A["date"])
	smos_stats_B["date"] = pd.to_datetime(smos_stats_B["date"])	
	smos_stats_D["date"] = pd.to_datetime(smos_stats_D["date"])
 
	msA_f, csA_f = mooring_draft_range(monthly_stats_A, cryosat_stats_A)
	msB_f, csB_f = mooring_draft_range(monthly_stats_B, cryosat_stats_B)
	msD_f, csD_f = mooring_draft_range(monthly_stats_D, cryosat_stats_D)
 
	_, smA_f = mooring_draft_range(monthly_stats_A, smos_stats_A)
	_, smB_f = mooring_draft_range(monthly_stats_B, smos_stats_B)
	_, smD_f = mooring_draft_range(monthly_stats_D, smos_stats_D)
 
	smA_f_suspicious = smA_f[smA_f["suspicious"] == 1]
	smB_f_suspicious = smB_f[smB_f["suspicious"] == 1]
	smD_f_suspicious = smD_f[smD_f["suspicious"] == 1]
 
	# 3 subplots for each mooring
	fig, ax = plt.subplots(3, 1, figsize=(6.733, 4.7), sharex=True)
	
	# Plot for mooring A
	ax[0].plot(msA_f["date"], msA_f["mean_draft"], marker="d",label="Mooring", color="black", zorder = 0)
	ax[0].scatter(csA_f["date"], csA_f["mean_draft"], label="UiT", color="#4ca64c", zorder = 1)
	ax[0].scatter(smA_f["date"], smA_f["mean_draft"], label="SMOS", color="#4c4cff", zorder = 2)
	ax[0].scatter(smA_f_suspicious["date"], smA_f_suspicious["mean_draft"], color="red", label="Saturated SMOS", zorder=3, marker="x")
	for date in msA_f["date"]:
		ax[0].axvline(date, linestyle='--', color='gray', alpha=0.5, linewidth=1)
	ax[0].set_ylabel("SID [m]")
	ax[0].set_title("Mooring A")
 
	handles, labels = ax[0].get_legend_handles_labels()
	left_handles = [handles[0], handles[1]]  # Mooring, UiT
	right_handles = [handles[2], handles[3]]  # SMOS, Saturated SMOS
	ax[0].legend(left_handles, ["Mooring", "UiT"],
				 loc='lower left', bbox_to_anchor=(0.0, 3), frameon=False,
				 ncol=2, borderaxespad=0.0, handletextpad=0.3)

	# Place right-aligned legend
	ax[0].legend(right_handles, ["SMOS", "Saturated SMOS"],
				 loc='lower right', bbox_to_anchor=(1.0, 3), frameon=False,
				 ncol=2, borderaxespad=0.0, handletextpad=0.3)

	# Prevent legend overlap with each other
	ax[0].add_artist(ax[0].legend(left_handles, ["Mooring", "UiT"], loc='lower left', bbox_to_anchor=(0.0, 1.2), frameon=False, ncol=2, borderaxespad=0.0, handletextpad=0.3))
	ax[0].add_artist(ax[0].legend(right_handles, ["SMOS", "Saturated SMOS"], loc='lower right', bbox_to_anchor=(1, 1.2), frameon=False, ncol=2, borderaxespad=0.0, handletextpad=0.3))
	ax[0].tick_params(axis='both', direction='in')
 
	ax[0].grid(True)

	# Plot for mooring B
	ax[1].plot(msB_f["date"], msB_f["mean_draft"], marker="d", color="black", zorder = 0)
	ax[1].scatter(csB_f["date"], csB_f["mean_draft"], color="#4ca64c", zorder = 1)
	ax[1].scatter(smB_f["date"], smB_f["mean_draft"], color="#4c4cff", zorder = 2)
	ax[1].scatter(smB_f_suspicious["date"], smB_f_suspicious["mean_draft"], color="red", label="Saturated SMOS", zorder=3, marker="x")
	for date in msB_f["date"]:
		ax[1].axvline(date, linestyle='--', color='gray', alpha=0.5, linewidth=1)
	ax[1].set_ylabel("SID [m]")
	ax[1].set_title("Mooring B")
	ax[1].tick_params(axis='both', direction='in')
	ax[1].grid(True)

	# Plot for mooring D
	ax[2].plot(msD_f["date"], msD_f["mean_draft"], marker="d", color="black", zorder = 0)
	ax[2].scatter(csD_f["date"], csD_f["mean_draft"], color="#4ca64c", zorder = 1)
	ax[2].scatter(smD_f["date"], smD_f["mean_draft"], color="#4c4cff", zorder = 2)
	ax[2].scatter(smD_f_suspicious["date"], smD_f_suspicious["mean_draft"], color="red", label="Suspicious SMOS", zorder=3, marker="x")
	for date in msD_f["date"]:
		ax[2].axvline(date, linestyle='--', color='gray', alpha=0.5, linewidth=1)
	ax[2].set_ylabel("SID [m]")
	ax[2].set_xlabel("Date")
	ax[2].set_title("Mooring D")
	ax[2].tick_params(axis='both', direction='in')
	ax[2].grid(True)

	# Improve layout
	plt.subplots_adjust(left=0.079, right=0.999, top=0.9, bottom=0.084 , hspace=0.24) #0.084 bottom
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\BGEP validations - CryoSat and SMOS\TimeSerier_BGEP_SMOS_Cryo.png", dpi=300)
	#plt.show()
 
 
def box_scatter():
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	# stack the mooring data, to an single 
	mooring_data = np.concatenate([mooring_A_draft, mooring_B_draft, mooring_D_draft])
	cryosat_data = np.concatenate([cryosat_A_draft, cryosat_B_draft, cryosat_D_draft])
	smos_data = np.concatenate([smos_A_draft, smos_B_draft, smos_D_draft])
 
	nan_mask = ~np.isnan(mooring_data) & ~np.isnan(cryosat_data) & ~np.isnan(smos_data)
	uls, cryo, smos = [arr[nan_mask] for arr in (mooring_data, cryosat_data, smos_data)]
 
	binned_uls_cryo, binned_uls_smos = [], []
	for i in range(len(bins) - 1):
		bin_mask = (uls >= bins[i]) & (uls < bins[i + 1])
		binned_uls_cryo.append(cryo[bin_mask])
		binned_uls_smos.append(smos[bin_mask])
  
	fig = plt.figure(figsize=(10, 5))  # 10x10 figure size
	box_width = 0.4
	gap = (1 - 2 * box_width) / 3

	ax1 = fig.add_axes([gap, 0.2, box_width, 0.6])
	ax2 = fig.add_axes([2 * gap + box_width, 0.2, box_width, 0.6])

	# Left box plot
	ax1.boxplot(binned_uls_cryo, labels = bin_labels, medianprops=dict(color='black'), meanprops=dict(color="#f7022a"), showmeans=True, meanline=True, patch_artist=True, showfliers=False, boxprops=dict(facecolor='lightgray', alpha=0.6))
	ax1.plot([], [], "--", linewidth=1, color = "#f7022a", label = "Mean")
	ax1.plot([], [], "-", linewidth=1, color = "black", label = "Median")
 
	# scatter 
	for i, data in enumerate(binned_uls_cryo):
		x = np.random.normal(i + 1, 0.05, size=len(data))
		ax1.scatter(x, data, color='green', alpha=0.7, edgecolor="none", linewidth=0.)
	ax1.set_ylabel('UiT SID [m]')
	ax1.set_xlabel('ULS SID bins [m]')
	ax1.tick_params(axis='both', direction='in')
	ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.08), ncol=2, borderaxespad=0.0, handletextpad=0.3)
	
	
	# Right box plot
	#ax2.boxplot(binned_oib_smos_data, labels = bin_labels, medianprops=dict(color='black'), patch_artist=True, showfliers=False, boxprops=dict(facecolor='lightgray', alpha=0.6))
	ax2.boxplot(binned_uls_smos, labels = bin_labels, medianprops=dict(color='black'), showmeans=True, meanline=True, meanprops=dict(color="#f7022a"), patch_artist=True, showfliers=False, boxprops=dict(facecolor='lightgray', alpha=0.6))
	ax2.plot([], [], "--", linewidth=1, color = "#f7022a", label = "Mean")
	ax2.plot([], [], "-", linewidth=1, color = "black", label = "Median")
	for i, data in enumerate(binned_uls_smos):
		x = np.random.normal(i + 1, 0.05, size=len(data))
		ax2.scatter(x, data, color='blue', alpha=0.7, edgecolor="none", linewidth=0.)
	ax2.set_ylabel('SMOS SID [m]')
	ax2.set_xlabel('ULS SID bins [m]')
	ax2.tick_params(axis='both', direction='in')
	#ax2.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.5, 1.08), ncol=2, borderaxespad=0.0, handletextpad=0.3)

	# Add grid lines
	ax1.grid(axis='y', linestyle='--', alpha=0.5)
	ax2.grid(axis='y', linestyle='--', alpha=0.5)

	plt.show()
 
 
def statistics(x, y):
	""" 
	Perform linear regression and calculate statistics.
	
	Parameters
	----------
 	x : numpy.ndarray
		Independent variable (e.g., mooring draft).
	y : numpy.ndarray
		Dependent variable (e.g., satellite draft).
  
	Returns
	-------
	x_range : numpy.ndarray
		Range of x values for plotting the regression line.
	y_pred : numpy.ndarray
		Predicted y values from the regression model.
	n : int
		Number of data points.
	bias : float
		Mean bias between x and y.
	rmse : float
		Root Mean Square Error between x and y.
	r : float
		Correlation coefficient between x and y.
	slope : float
		Slope of the regression line.
	intercept : float
		y-intercept of the regression line.
	"""
	model = LinearRegression()
	x_reshaped = np.array(x).reshape(-1, 1)
	y_reshaped = np.array(y).reshape(-1, 1)
	model.fit(x_reshaped, y_reshaped)
	
	x_range = np.linspace(np.min(x), np.max(x), 100)
	y_pred = model.predict(x_range.reshape(-1, 1))	
 
	n = len(x)
	bias = np.mean(y - x)
	rmse = np.sqrt(np.mean((y - x) ** 2))
	r = np.corrcoef(x, y)[0, 1]
	slope = model.coef_[0][0]
	intercept = model.intercept_[0]
	
	return x_range, y_pred, n, bias, rmse, r, slope, intercept

def single_anomaly():
	# Create a mask for values between 0 and 1 m for both datasets
	valid_mask = (mooring_A_draft >= 0) & (mooring_A_draft <= 1) & \
				 (cryosat_A_draft >= 0) & (cryosat_A_draft <= 1)
	
	# Filter the data using the mask
 
	mooring_A_draft_valid = mooring_A_draft[valid_mask]
	cryosat_A_draft_valid = cryosat_A_draft[valid_mask]

	# Calculate anomalies from the filtered data
	mooring_A_anomalies = mooring_A_draft_valid - np.mean(mooring_A_draft_valid)
	cryosat_A_anomalies = cryosat_A_draft_valid - np.mean(cryosat_A_draft_valid)

	# Compute statistics and regression
	cs_mooring_A_x, cs_mooring_A_y, cs_mo_A_n, cs_mo_A_bias, cs_mo_A_rmse, cs_mo_A_r, cs_mo_A_slope, cs_mo_A_intercept = statistics(mooring_A_anomalies, cryosat_A_anomalies)

	# Create figure
	fig, ax = plt.subplots()

	# Scatter plot
	ax.scatter(mooring_A_anomalies, cryosat_A_anomalies, label="CryoSat-2", color="teal", zorder=3)
	ax.plot(cs_mooring_A_x, cs_mooring_A_y, label="Regression", color="red", zorder=1)
	# -- Add boxplots in bins of Mooring anomalies --

	num_bins = 8
	bins = np.linspace(mooring_A_anomalies.min(), mooring_A_anomalies.max(), num_bins + 1)
	bin_indices = np.digitize(mooring_A_anomalies, bins)
	box_data = [cryosat_A_anomalies[bin_indices == b] for b in range(1, num_bins + 1)]
	box_positions = [0.5 * (bins[b - 1] + bins[b]) for b in range(1, num_bins + 1)]
	
	bp = ax.boxplot(box_data, positions=box_positions, widths=(bins[1] - bins[0]) * 0.7, patch_artist=True, showfliers=True, zorder=2, manage_ticks=False)
	
	# Customize appearance (optional)
	for box in bp['boxes']:
		box.set(facecolor='lightgray', alpha=0.5)
	for whisker in bp['whiskers']:
		whisker.set(color='black', linewidth=1)
	for cap in bp['caps']:
		cap.set(color='black', linewidth=1)
	for median in bp['medians']:
		median.set(color='darkred', linewidth=2)


	y_limits = ax.get_ylim()
	x_limits = ax.get_xlim()
	lower = max(x_limits[0], y_limits[0])
	upper = min(x_limits[1], y_limits[1])
	ax.plot([lower, upper], [lower, upper], '--', color='gray', zorder=0)

	# Labels and Title
	ax.set_title("Mooring A vs CryoSat-2 (Sea Ice Draft: 0-1 m)")
	ax.set_xlabel("Mooring Anomalies (m)")
	ax.set_ylabel("CryoSat-2 Anomalies (m)")
	ax.legend()
	ax.grid(True)

	# Add statistics text
	stats_text = (f"n: {cs_mo_A_n}\n"
				  f"Bias: {cs_mo_A_bias:.2f}\n"
				  f"RMSE: {cs_mo_A_rmse:.2f}\n"
				  f"r: {cs_mo_A_r:.2f}\n"
				  f"Slope: {cs_mo_A_slope:.2f}\n"
				  f"Intercept: {cs_mo_A_intercept:.2f}")
	ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10, ha='right', va='bottom')

	# Show plot
	plt.show()
 
def valid_mask(mooring_anomalies, satellite_anomalies):
	valid_mask = (mooring_anomalies >= 0) & (mooring_anomalies <= 1) & (satellite_anomalies >= 0) & (satellite_anomalies <= 1)
	#valid_mask_mooring = (mooring_anomalies >= 0) & (mooring_anomalies <= 1)
	#valid_mask_satellite = (satellite_anomalies >= 0) & (satellite_anomalies <= 1)
	mooring_anomalies = mooring_anomalies[valid_mask]
	satellite_anomalies = satellite_anomalies[valid_mask]	
	return mooring_anomalies, satellite_anomalies	
 
def clean_and_stats(mooring_anomalies, satellite_anomalies):
	valid_mask = (~np.isnan(mooring_anomalies)) & (~np.isnan(satellite_anomalies))
	mooring_valid = mooring_anomalies[valid_mask]
	satellite_valid = satellite_anomalies[valid_mask]
	stats = statistics(mooring_valid, satellite_valid)
	return mooring_valid, satellite_valid, stats

def add_boxplot(ax, x_data, y_data, num_bins=8):
	bins = np.linspace(x_data.min(), x_data.max(), num_bins + 1)
	bin_indices = np.digitize(x_data, bins)
	box_data = [y_data[bin_indices == b] for b in range(1, num_bins + 1)]
	box_positions = [0.5 * (bins[b - 1] + bins[b]) for b in range(1, num_bins + 1)]

	bp = ax.boxplot(
		box_data,
		positions=box_positions,
		widths=(bins[1] - bins[0]) * 0.7,
		patch_artist=True,
		showfliers=True,
		zorder=2,
		manage_ticks=False
	)
	for box in bp['boxes']:
		box.set(facecolor='lightgray', alpha=0.5)
	for whisker in bp['whiskers']:
		whisker.set(color='black', linewidth=1)
	for cap in bp['caps']:
		cap.set(color='black', linewidth=1)
	for median in bp['medians']:
		median.set(color='black', linewidth=1)


def plot_subplot(ax, x, y, reg_x, reg_y, stats, label, color, xlabel):
	ax.scatter(x, y, label=label, zorder=3, color=color)
	ax.plot(reg_x, reg_y, label="Regression", color="red", zorder=1)
	add_boxplot(ax, x, y)

	lower = max(ax.get_xlim()[0], ax.get_ylim()[0])
	upper = min(ax.get_xlim()[1], ax.get_ylim()[1])
	ax.plot([lower, upper], [lower, upper], '--', color='gray', zorder=0)

	ax.set_xlabel(xlabel)
	#ax.set_ylabel(f"{label} Anomalies (m)")
	#ax.legend()
	ax.grid(True)
	ax.tick_params(axis='both', direction='in')

	# Add statistics text
	n, bias, rmse, r, slope, intercept = stats[2:8]
	stats_text = (f"n: {n}\n"
				  f"Bias: {bias:.2f} m\n"
				  f"RMSE: {rmse:.2f} m\n"
				  f"r: {r:.2f}\n"
				  f"Slope: {slope:.2f}\n"
				  f"Intercept: {intercept:.2f}")
	ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
			ha='right', va='bottom', bbox=dict(facecolor='gray', alpha=0.5))



def compute_monthly_anomalies(draft_data, dates):
	"""
	Computes anomalies by subtracting the monthly mean (climatology) from the draft data.
	
	Parameters
	----------
	draft_data : array-like
		The draft data (e.g. mooring or satellite values).
	dates : array-like
		Dates corresponding to the draft values. They must be convertible to datetime.
	
	Returns
	-------
	anomalies : numpy.ndarray
		An array of anomalies computed as: data - monthly_mean.
	"""
	# Create a DataFrame from your data
	df = pd.DataFrame({
		"date": pd.to_datetime(dates),
		"draft": draft_data
	})
	
	# Extract the month (as an integer 1-12)
	df["month"] = df["date"].dt.month
	
	# Compute the monthly mean (climatology) across all years.
	# The transform('mean') call replicates the monthly mean for each row.
	df["monthly_mean"] = df.groupby("month")["draft"].transform("mean")
	#print(df["monthly_mean"])
	# Calculate anomalies: difference from the monthly mean.
	df["anomaly"] = df["draft"] - df["monthly_mean"]
	
	return df["anomaly"].values

def bar_hist_plot():
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	mooring_A_draft_C, cryosat_A_draft_c = valid_mask(mooring_A_draft, cryosat_A_draft)
	mooring_B_draft_C, cryosat_B_draft_c = valid_mask(mooring_B_draft, cryosat_B_draft)
	mooring_D_draft_C, cryosat_D_draft_c = valid_mask(mooring_D_draft, cryosat_D_draft)
 
	mooring_A_draft_S, smos_A_draft_s = valid_mask(mooring_A_draft, smos_A_draft)
	mooring_B_draft_S, smos_B_draft_s = valid_mask(mooring_B_draft, smos_B_draft)
	mooring_D_draft_S, smos_D_draft_s = valid_mask(mooring_D_draft, smos_D_draft)

	#mooring_A_draft_C, mooring_B_draft_C, mooring_D_draft_C = mooring_A_draft, mooring_B_draft, mooring_D_draft
	#cryosat_A_draft_c, cryosat_B_draft_c, cryosat_D_draft_c = cryosat_A_draft, cryosat_B_draft, cryosat_D_draft
	#mooring_A_draft_S, mooring_B_draft_S, mooring_D_draft_S = mooring_A_draft, mooring_B_draft, mooring_D_draft
	#smos_A_draft_s, smos_B_draft_s, smos_D_draft_s = smos_A_draft, smos_B_draft, smos_D_draft
 
	mooring_c = np.concatenate([mooring_A_draft_C, mooring_B_draft_C, mooring_D_draft_C])
	cryosat_c = np.concatenate([cryosat_A_draft_c, cryosat_B_draft_c, cryosat_D_draft_c])
 
	mooring_s = np.concatenate([mooring_A_draft_S, mooring_B_draft_S, mooring_D_draft_S])
	smos_s = np.concatenate([smos_A_draft_s, smos_B_draft_s, smos_D_draft_s])

	nan_mask_c = ~np.isnan(mooring_c) & ~np.isnan(cryosat_c)
	nan_mask_s = ~np.isnan(mooring_s) & ~np.isnan(smos_s)
	mooring_c, cryosat_c = [arr[nan_mask_c] for arr in (mooring_c, cryosat_c)]
	mooring_s, smos_s = [arr[nan_mask_s] for arr in (mooring_s, smos_s)]
 
	smos_means = []
	cryo_means = []
	mooring_means = []
	for i in range(len(bins)-1): #
		bin_mask_c = (mooring_c >= bins[i]) & (mooring_c < bins[i + 1])
		bin_mask_s = (mooring_s >= bins[i]) & (mooring_s < bins[i + 1])
		binned_cryo = cryosat_c[bin_mask_c]
		binned_smos = smos_s[bin_mask_s]
		cryo_means.append(np.mean(binned_cryo))
		smos_means.append(np.mean(binned_smos))
		bin_mask_moor = (mooring_c >= bins[i]) & (mooring_c < bins[i + 1])
		mooring_means.append(np.mean(mooring_c[bin_mask_moor]) if bin_mask_moor.any() else np.nan)
  
		# Set up the figure with one main plot and two smaller plots
	fig = plt.figure(figsize=(6.733, 5.5))

	# Layout parameters
	box_size = 0.35
	main_height = 0.5
	gap = (1 - 2 * box_size) / 3
	gap_main = (1 - main_height - box_size) / 3

	# Create axes
	ax_main = fig.add_axes([gap, 2 * gap_main + box_size, 1 - 2 * gap, main_height])
	ax_left = fig.add_axes([gap, gap_main-0.06, box_size, box_size])
	ax_right = fig.add_axes([2 * gap + box_size, gap_main-0.06, box_size, box_size])

	# --- Main plot: Bar plot ---
	x = np.arange(len(bin_labels))
	width = 0.35  # width of the bars
 
	ax_main.bar(x + width/2, cryo_means, width, label='UiT', color='green', alpha=0.7)
	ax_main.bar(x - width/2, smos_means, width, label='SMOS', color='blue', alpha=0.7)
	for i, mean_val in enumerate(mooring_means):
		if not np.isnan(mean_val):
			ax_main.hlines(mean_val, x[i] - width, x[i] + width, colors='black', linestyles='--', linewidth=1, label='Mooring' if i == 0 else None)
	ax_main.set_ylabel('Mean SID [m]')
	ax_main.set_xlabel('Mooring SID bins [m]')
	ax_main.tick_params(axis='both', direction='in')
	ax_main.set_xticks(x)
	ax_main.set_xticklabels(bin_labels)
	ax_main.legend(bbox_to_anchor=(0.65, 1.1), loc='upper right', ncol=3, borderaxespad=0.0, handletextpad=0.3, frameon=False)
	ax_main.grid(axis='y', linestyle='--', alpha=0.5)
 
	range_mask_c = (mooring_c >= 0) & (mooring_c <= 1)
	range_mask_s = (mooring_s >= 0) & (mooring_s <= 1)
	mooring_cm, cryosat_cm = mooring_c[range_mask_c], cryosat_c[range_mask_c]
	mooring_sm, smos_sm = mooring_s[range_mask_s], smos_s[range_mask_s]
	bin_edges=np.linspace(0,1,11)
 
	# --- Bottom right plot: Histogram OIB vs Cryo ---
	ax_right.hist(mooring_cm, bins=bin_edges, alpha=0.7, label='Mooring', color='black', density=True)
	ax_right.hist(cryosat_cm, bins=bin_edges, edgecolor='green', color="green", fill=True, linewidth=1, hatch='xx', alpha=0.7, label='UiT', density=True)
	ax_right.tick_params(axis='both', direction='in')
	ax_right.set_xlabel('SID [m]')
	ax_right.yaxis.set_major_formatter(PercentFormatter(1))
	ax_right.grid(alpha=0.5, linestyle='--')
	ax_right.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2, mode="expand", borderaxespad=0.0, handletextpad=0.3, frameon=False) 
	
	# --- Bottom left plot: Histogram OIB vs SMOS ---
	ax_left.hist(mooring_sm, bins=bin_edges, alpha=0.7, label='Mooring', color='black', density=True)
	ax_left.hist(smos_sm, bins=bin_edges, edgecolor='blue', color="blue", fill=True, linewidth=1, hatch='xx', alpha=0.7, label='SMOS', density=True)
	ax_left.set_xlabel('SID [m]')
	ax_left.set_ylabel('Observations (%)')
	ax_left.tick_params(axis='both', direction='in')
	ax_left.yaxis.set_major_formatter(PercentFormatter(1))
	ax_left.grid(alpha=0.5, linestyle='--')
	ax_left.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2, mode="expand", borderaxespad=0.0, handletextpad=0.3, frameon=False)


	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\BGEP validations - CryoSat and SMOS\BarHist_all_filtered.png", dpi=300, bbox_inches='tight')
	#plt.show()

	

def scatter_plot():
	# scatter plot with out calculating the anomalies
	mooring_A_draft_C, cryosat_A_draft_c = valid_mask(mooring_A_draft, cryosat_A_draft)
	mooring_B_draft_C, cryosat_B_draft_c = valid_mask(mooring_B_draft, cryosat_B_draft)
	mooring_D_draft_C, cryosat_D_draft_c = valid_mask(mooring_D_draft, cryosat_D_draft)
 
	mooring_A_draft_S, smos_A_draft_s = valid_mask(mooring_A_draft, smos_A_draft)
	mooring_B_draft_S, smos_B_draft_s = valid_mask(mooring_B_draft, smos_B_draft)
	mooring_D_draft_S, smos_D_draft_s = valid_mask(mooring_D_draft, smos_D_draft)
 
	fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
 
	# CryoSat-2
	xA, yA, statsA = clean_and_stats(mooring_A_draft_C, cryosat_A_draft_c)
	plot_subplot(ax[0, 0], xA, yA, statsA[0], statsA[1], statsA, "UiT", "#4ca64c", "")
	ax[0, 0].set_ylabel("UiT SID [m]")
	ax[0, 0].set_title("Mooring A")
 
	xB, yB, statsB = clean_and_stats(mooring_B_draft_C, cryosat_B_draft_c)
	plot_subplot(ax[0, 1], xB, yB, statsB[0], statsB[1], statsB, "UiT", "#4ca64c", "")
	ax[0, 1].set_title("Mooring B")
	
	xD, yD, statsD = clean_and_stats(mooring_D_draft_C, cryosat_D_draft_c)
	plot_subplot(ax[0, 2], xD, yD, statsD[0], statsD[1], statsD, "UIT", "#4ca64c", "")
	ax[0, 2].set_title("Mooring D")
 
	# SMOS
	xA_smos, yA_smos, statsA_smos = clean_and_stats(mooring_A_draft_S, smos_A_draft_s)
	plot_subplot(ax[1, 0], xA_smos, yA_smos, statsA_smos[0], statsA_smos[1], statsA_smos, "SMOS", "#4c4cff", "Mooring SID [m]")
	ax[1, 0].set_ylabel("SMOS SID [m]")
	#ax[1, 0].set_title("Mooring A vs SMOS")
 
	xB_smos, yB_smos, statsB_smos = clean_and_stats(mooring_B_draft_S, smos_B_draft_s)
	plot_subplot(ax[1, 1], xB_smos, yB_smos, statsB_smos[0], statsB_smos[1], statsB_smos, "SMOS", "#4c4cff", "Mooring SID [m]")
	#ax[1, 1].set_title("Mooring B vs SMOS")
 
	xD_smos, yD_smos, statsD_smos = clean_and_stats(mooring_D_draft_S, smos_D_draft_s)
	plot_subplot(ax[1, 2], xD_smos, yD_smos, statsD_smos[0], statsD_smos[1], statsD_smos, "SMOS", "#4c4cff", "Mooring SID [m]")
	#ax[1, 2].set_title("Mooring D vs SMOS")
 
	plt.tight_layout()
	plt.show()

def total_scatter_plot():
	mooring_A_draft_C, cryosat_A_draft_c = valid_mask(mooring_A_draft, cryosat_A_draft)
	mooring_B_draft_C, cryosat_B_draft_c = valid_mask(mooring_B_draft, cryosat_B_draft)
	mooring_D_draft_C, cryosat_D_draft_c = valid_mask(mooring_D_draft, cryosat_D_draft)

	mooring_A_draft_S, smos_A_draft_s = valid_mask(mooring_A_draft, smos_A_draft)
	mooring_B_draft_S, smos_B_draft_s = valid_mask(mooring_B_draft, smos_B_draft)
	mooring_D_draft_S, smos_D_draft_s = valid_mask(mooring_D_draft, smos_D_draft)

	mooring_draft_c = np.concatenate([mooring_A_draft_C, mooring_B_draft_C, mooring_D_draft_C])
	cryosat_draft_c = np.concatenate([cryosat_A_draft_c, cryosat_B_draft_c, cryosat_D_draft_c])
	mooring_draft_s = np.concatenate([mooring_A_draft_S, mooring_B_draft_S, mooring_D_draft_S])
	smos_draft_s = np.concatenate([smos_A_draft_s, smos_B_draft_s, smos_D_draft_s])

	# Use subplots instead of add_axes
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.733, 3.5))

	# Left subplot: UiT
	x, y, stats = clean_and_stats(mooring_draft_c, cryosat_draft_c)
	plot_subplot(ax1, x, y, stats[0], stats[1], stats, "UiT", "#4ca64c", "Mooring SID [m]")
	ax1.set_ylabel("UiT SID [m]")
	ax1.set_xlabel("Mooring SID [m]")

	# Right subplot: SMOS
	x_smos, y_smos, stats_smos = clean_and_stats(mooring_draft_s, smos_draft_s)
	plot_subplot(ax2, x_smos, y_smos, stats_smos[0], stats_smos[1], stats_smos, "SMOS", "#4c4cff", "Mooring SID [m]")
	ax2.set_ylabel("SMOS SID [m]")
	ax2.set_xlabel("Mooring SID [m]")

	plt.subplots_adjust(left=0.08, right=0.996, top=0.842, bottom=0.11 , wspace=0.2) 
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\BGEP validations - CryoSat and SMOS\PairScatter_allULS_raw.png", dpi=300, bbox_inches='tight')
	#plt.show()
 
 
	

def draft_anomalies():
	# Calculate anomalies
	mooring_A_draft_C, cryosat_A_draft_c = valid_mask(mooring_A_draft, cryosat_A_draft)
	mooring_B_draft_C, cryosat_B_draft_c = valid_mask(mooring_B_draft, cryosat_B_draft)
	mooring_D_draft_C, cryosat_D_draft_c = valid_mask(mooring_D_draft, cryosat_D_draft)
 
	mooring_A_draft_S, smos_A_draft_s = valid_mask(mooring_A_draft, smos_A_draft)
	mooring_B_draft_S, smos_B_draft_s = valid_mask(mooring_B_draft, smos_B_draft)
	mooring_D_draft_S, smos_D_draft_s = valid_mask(mooring_D_draft, smos_D_draft)
 
	# Calculate anomalies
	# We do that by taking the mean of the months (oct, nov, dec, etc) for all the years, then replicate that for the number of years
	# and subtract from the original data (anomalies = data - mean_months)
	# This is done for both mooring and satellite data
	
	mooring_A_anom_c = compute_monthly_anomalies(mooring_A_draft_C, monthly_stats_A["date"])
	cryosat_A_anom = compute_monthly_anomalies(cryosat_A_draft_c, cryosat_stats_A["date"])
	mooring_B_anom_c = compute_monthly_anomalies(mooring_B_draft_C, monthly_stats_B["date"])
	cryosat_B_anom = compute_monthly_anomalies(cryosat_B_draft_c, cryosat_stats_B["date"])
	mooring_D_anom_c = compute_monthly_anomalies(mooring_D_draft_C, monthly_stats_D["date"])
	cryosat_D_anom = compute_monthly_anomalies(cryosat_D_draft_c, cryosat_stats_D["date"])
 
	mooring_A_anom_s = compute_monthly_anomalies(mooring_A_draft_S, monthly_stats_A["date"])
	smos_A_anom = compute_monthly_anomalies(smos_A_draft_s, smos_stats_A["date"])
	mooring_B_anom_s = compute_monthly_anomalies(mooring_B_draft_S, monthly_stats_B["date"])
	smos_B_anom = compute_monthly_anomalies(smos_B_draft_s, smos_stats_B["date"])
	mooring_D_anom_s = compute_monthly_anomalies(mooring_D_draft_S, monthly_stats_D["date"])
	smos_D_anom = compute_monthly_anomalies(smos_D_draft_s, smos_stats_D["date"])

	fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharex=True)

	# CryoSat-2
	xA, yA, statsA = clean_and_stats(mooring_A_anom_c, cryosat_A_anom)
	plot_subplot(ax[0, 0], xA, yA, statsA[0], statsA[1], statsA, "CryoSat-2", "#4ca64c", "")
	ax[0, 0].set_ylabel("UiT SID Anomalies [m]")
	ax[0, 0].set_title("Mooring A")
 
	xB, yB, statsB = clean_and_stats(mooring_B_anom_c, cryosat_B_anom)
	plot_subplot(ax[0, 1], xB, yB, statsB[0], statsB[1], statsB, "CryoSat-2", "#4ca64c", "")
	ax[0, 1].set_title("Mooring B")

	xD, yD, statsD = clean_and_stats(mooring_D_anom_c, cryosat_D_anom)
	plot_subplot(ax[0, 2], xD, yD, statsD[0], statsD[1], statsD, "CryoSat-2", "#4ca64c", "")
	ax[0, 2].set_title("Mooring D")

	# SMOS
	xA_smos, yA_smos, statsA_smos = clean_and_stats(mooring_A_anom_s, smos_A_anom)
	plot_subplot(ax[1, 0], xA_smos, yA_smos, statsA_smos[0], statsA_smos[1], statsA_smos, "SMOS", "#4c4cff", "Mooring SID Anomalies [m]")
	ax[1, 0].set_ylabel("SMOS SID Anomalies [m]")
	#ax[1, 0].set_title("Mooring A vs SMOS")

	xB_smos, yB_smos, statsB_smos = clean_and_stats(mooring_B_anom_s, smos_B_anom)
	plot_subplot(ax[1, 1], xB_smos, yB_smos, statsB_smos[0], statsB_smos[1], statsB_smos, "SMOS", "#4c4cff", "Mooring SID Anomalies [m]")
	#ax[1, 1].set_title("Mooring B vs SMOS")

	xD_smos, yD_smos, statsD_smos = clean_and_stats(mooring_D_anom_s, smos_D_anom)
	plot_subplot(ax[1, 2], xD_smos, yD_smos, statsD_smos[0], statsD_smos[1], statsD_smos, "SMOS", "#4c4cff", "Mooring SID Anomalies [m]")
	#ax[1, 2].set_title("Mooring D vs SMOS")

	plt.tight_layout()
	plt.show()
	
def total_draft_anomalies():
	mooring_A_draft_C, cryosat_A_draft_c = valid_mask(mooring_A_draft, cryosat_A_draft)
	mooring_B_draft_C, cryosat_B_draft_c = valid_mask(mooring_B_draft, cryosat_B_draft)
	mooring_D_draft_C, cryosat_D_draft_c = valid_mask(mooring_D_draft, cryosat_D_draft)

	mooring_A_draft_S, smos_A_draft_s = valid_mask(mooring_A_draft, smos_A_draft)
	mooring_B_draft_S, smos_B_draft_s = valid_mask(mooring_B_draft, smos_B_draft)
	mooring_D_draft_S, smos_D_draft_s = valid_mask(mooring_D_draft, smos_D_draft)

	mooring_A_anom_c = compute_monthly_anomalies(mooring_A_draft_C, monthly_stats_A["date"])
	cryosat_A_anom = compute_monthly_anomalies(cryosat_A_draft_c, cryosat_stats_A["date"])
	mooring_B_anom_c = compute_monthly_anomalies(mooring_B_draft_C, monthly_stats_B["date"])
	cryosat_B_anom = compute_monthly_anomalies(cryosat_B_draft_c, cryosat_stats_B["date"])
	mooring_D_anom_c = compute_monthly_anomalies(mooring_D_draft_C, monthly_stats_D["date"])
	cryosat_D_anom = compute_monthly_anomalies(cryosat_D_draft_c, cryosat_stats_D["date"])

	mooring_A_anom_s = compute_monthly_anomalies(mooring_A_draft_S, monthly_stats_A["date"])
	smos_A_anom = compute_monthly_anomalies(smos_A_draft_s, smos_stats_A["date"])
	mooring_B_anom_s = compute_monthly_anomalies(mooring_B_draft_S, monthly_stats_B["date"])
	smos_B_anom = compute_monthly_anomalies(smos_B_draft_s, smos_stats_B["date"])
	mooring_D_anom_s = compute_monthly_anomalies(mooring_D_draft_S, monthly_stats_D["date"])
	smos_D_anom = compute_monthly_anomalies(smos_D_draft_s, smos_stats_D["date"])

	mooring_anom_c = np.concatenate([mooring_A_anom_c, mooring_B_anom_c, mooring_D_anom_c])
	cryosat_anom = np.concatenate([cryosat_A_anom, cryosat_B_anom, cryosat_D_anom])

	mooring_anom_s = np.concatenate([mooring_A_anom_s, mooring_B_anom_s, mooring_D_anom_s])
	smos_anom = np.concatenate([smos_A_anom, smos_B_anom, smos_D_anom])

	# Create 1x2 subplots
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.733, 3.5))

	# Plot CryoSat anomalies
	x, y, stats = clean_and_stats(mooring_anom_c, cryosat_anom)
	plot_subplot(ax1, x, y, stats[0], stats[1], stats, "UiT", "#4ca64c", "Mooring SID Anomalies [m]")
	ax1.set_ylabel("UiT SID Anomalies [m]")
	ax1.set_xlabel("Mooring SID Anomalies [m]")

	# Plot SMOS anomalies
	x_smos, y_smos, stats_smos = clean_and_stats(mooring_anom_s, smos_anom)
	plot_subplot(ax2, x_smos, y_smos, stats_smos[0], stats_smos[1], stats_smos, "SMOS", "#4c4cff", "Mooring SID Anomalies [m]")
	ax2.set_ylabel("SMOS SID Anomalies [m]")
	ax2.set_xlabel("Mooring SID Anomalies [m]")

	plt.subplots_adjust(left=0.095, right=0.996, top=0.842, bottom=0.11 , wspace=0.262) 
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\BGEP validations - CryoSat and SMOS\PairScatter_allULS_seasonal_anomalies.png", dpi=300, bbox_inches='tight')
	#plt.show()
 
def histogram():
	""" 
	Plot histograms of the mooring sea ice draft, compared with the satellite sea ice draft ranging from 0 to 1 m.
	"""
	mooring_A_draft_C, cryosat_A_draft_c = valid_mask(mooring_A_draft, cryosat_A_draft)
	mooring_B_draft_C, cryosat_B_draft_c = valid_mask(mooring_B_draft, cryosat_B_draft)
	mooring_D_draft_C, cryosat_D_draft_c = valid_mask(mooring_D_draft, cryosat_D_draft)
 
	mooring_A_draft_S, smos_A_draft_s = valid_mask(mooring_A_draft, smos_A_draft)
	mooring_B_draft_S, smos_B_draft_s = valid_mask(mooring_B_draft, smos_B_draft)
	mooring_D_draft_S, smos_D_draft_s = valid_mask(mooring_D_draft, smos_D_draft)
	
	fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

	# Mooring A
	ax[0, 0].hist(mooring_A_draft_C, bins=10, color="#4D4D4D", label="Mooring", zorder = 1)
	ax[0, 0].hist(cryosat_A_draft_c, bins=10, edgecolor="#3D7E9A", fill=False, hatch = "xx",label="CryoSat-2", zorder = 2)
	ax[0, 0].set_title("Mooring A")
	ax[0, 0].set_ylabel("Count")
	ax[0, 0].grid(True, zorder = 0)
	ax[0, 0].legend()

	# Mooring B
	ax[0, 1].hist(mooring_B_draft_C, bins=10, color="#4D4D4D", zorder = 1)
	ax[0, 1].hist(cryosat_B_draft_c, bins=10, edgecolor="#3D7E9A", fill=False, hatch = "xx", zorder = 2)
	ax[0, 1].grid(True, zorder = 0)
	ax[0, 1].set_title("Mooring B")

	# Mooring D
	ax[0, 2].hist(mooring_D_draft_C, bins=10, color="#4D4D4D", zorder = 1)
	ax[0, 2].hist(cryosat_D_draft_c, bins=10, edgecolor="#3D7E9A", fill=False, hatch = "xx", zorder = 2)
	ax[0, 2].grid(True, zorder = 0)
	ax[0, 2].set_title("Mooring D")

	# SMOS Mooring A
	ax[1, 0].hist(mooring_A_draft_S, bins=10, color="#4D4D4D", label="Mooring", zorder = 1)
	ax[1, 0].hist(smos_A_draft_s, bins=10, edgecolor="#E07A5F", fill=False, hatch = "xx", label="SMOS", zorder = 2)
	ax[1, 0].grid(True, zorder = 0)
	ax[1, 0].set_xlabel("Ice draft [m]")
	ax[1, 0].set_ylabel("Count")
	ax[1, 0].legend()
 
 
	# SMOS Mooring B
	ax[1, 1].hist(mooring_B_draft_S, bins=10, color="#4D4D4D", zorder = 1)
	ax[1, 1].hist(smos_B_draft_s, bins=10, edgecolor="#E07A5F", fill=False, hatch = "xx", zorder = 2)
	ax[1, 1].grid(True, zorder = 0)
	ax[1, 1].set_xlabel("Ice draft [m]")	
 
	# SMOS Mooring D
	ax[1, 2].hist(mooring_D_draft_S, bins=10, color="#4D4D4D", zorder = 1)
	ax[1, 2].hist(smos_D_draft_s, bins=10, edgecolor="#E07A5F", fill=False, hatch = "xx", zorder = 2)
	ax[1, 2].grid(True, zorder = 0)
	ax[1, 2].set_xlabel("Ice draft [m]")
 
	plt.tight_layout()
	plt.show()
 
def full_stat_metric():
	mooring_A_draft_C, cryosat_A_draft_c = valid_mask(mooring_A_draft, cryosat_A_draft)
	mooring_B_draft_C, cryosat_B_draft_c = valid_mask(mooring_B_draft, cryosat_B_draft)
	mooring_D_draft_C, cryosat_D_draft_c = valid_mask(mooring_D_draft, cryosat_D_draft)
 
	mooring_A_draft_S, smos_A_draft_s = valid_mask(mooring_A_draft, smos_A_draft)
	mooring_B_draft_S, smos_B_draft_s = valid_mask(mooring_B_draft, smos_B_draft)
	mooring_D_draft_S, smos_D_draft_s = valid_mask(mooring_D_draft, smos_D_draft)
 
	mooring_c = np.concatenate([mooring_A_draft_C, mooring_B_draft_C, mooring_D_draft_C])
	cryosat_c = np.concatenate([cryosat_A_draft_c, cryosat_B_draft_c, cryosat_D_draft_c])
 
	mooring_s = np.concatenate([mooring_A_draft_S, mooring_B_draft_S, mooring_D_draft_S])
	smos_s = np.concatenate([smos_A_draft_s, smos_B_draft_s, smos_D_draft_s])

	nan_mask_c = ~np.isnan(mooring_c) & ~np.isnan(cryosat_c)  
	nan_mask_s = ~np.isnan(mooring_s) & ~np.isnan(smos_s)
	mooring_c, cryosat_c = [arr[nan_mask_c] for arr in (mooring_c, cryosat_c)]
	mooring_s, smos_s = [arr[nan_mask_s] for arr in (mooring_s, smos_s)]
 
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	products = {'UiT': cryosat_c, "SMOS": smos_s}
	results = {p: {} for p in products}
 
	for i in range(len(bins)-1):
		lo, hi = bins[i], bins[i+1]
		label = labels[i]
		bin_mask_c = (mooring_c >= lo) & (mooring_c < hi)
		bin_mask_s = (mooring_s >= lo) & (mooring_s < hi)
    
		cryo_bin = cryosat_c[bin_mask_c]
		smos_bin = smos_s[bin_mask_s]
		mooring_c_bin = mooring_c[bin_mask_c]
		mooring_s_bin = mooring_s[bin_mask_s]
  
		arrs = {
			'UiT': cryo_bin,
			'SMOS': smos_bin
		}
  
		mooring_for_product = {
			'UiT': mooring_c_bin,
			'SMOS': mooring_s_bin
		}
  
		raw = {}
		for p, arr in products:
			mooring_ref = mooring_for_product[p]
			b = np.mean(arr - mooring_ref)
			r = np.sqrt(np.mean((arr - mooring_ref)**2))
			c = np.corrcoef(arr, mooring_ref)[0,1]
			raw[p] = dict(bias=b, rmse=r, cc=c)

		# normalize each metric across products in this bin
		for metric in ('bias','rmse','cc'):
			vals = np.array([ raw[p][metric] for p in products ])
			mn, mx = vals.min(), vals.max()
			span = (mx - mn) if mx!=mn else 1.0
			for p in products:
				raw[p]['N'+metric] = (raw[p][metric] - mn) / span

		# compute DISO for each product
		for p in products:
			NB = raw[p]['Nbias']
			NR = raw[p]['Nrmse']
			NC = raw[p]['Ncc']
			raw[p]['DISO'] = np.sqrt(NB**2 + NR**2 + (NC - 1.0)**2)
			# store
			results[p][label] = raw[p]
   
	for prod, bins in results.items():
		print(f"\n{prod}:")
		for bin_label, m in bins.items():
			print(f"  {bin_label}: Bias={m['bias']:.3f}, RMSE={m['rmse']:.3f}, "
				  f"CC={m['cc']:.3f}, DISO={m['DISO']:.3f}")
  
def full_stat_metric():
	mooring_A_draft_C, cryosat_A_draft_c = valid_mask(mooring_A_draft, cryosat_A_draft)
	mooring_B_draft_C, cryosat_B_draft_c = valid_mask(mooring_B_draft, cryosat_B_draft)
	mooring_D_draft_C, cryosat_D_draft_c = valid_mask(mooring_D_draft, cryosat_D_draft)
 
	mooring_A_draft_S, smos_A_draft_s = valid_mask(mooring_A_draft, smos_A_draft)
	mooring_B_draft_S, smos_B_draft_s = valid_mask(mooring_B_draft, smos_B_draft)
	mooring_D_draft_S, smos_D_draft_s = valid_mask(mooring_D_draft, smos_D_draft)
 
	mooring_c = np.concatenate([mooring_A_draft_C, mooring_B_draft_C, mooring_D_draft_C])
	cryosat_c = np.concatenate([cryosat_A_draft_c, cryosat_B_draft_c, cryosat_D_draft_c])
 
	mooring_s = np.concatenate([mooring_A_draft_S, mooring_B_draft_S, mooring_D_draft_S])
	smos_s = np.concatenate([smos_A_draft_s, smos_B_draft_s, smos_D_draft_s])

	nan_mask_c = ~np.isnan(mooring_c) & ~np.isnan(cryosat_c)  
	nan_mask_s = ~np.isnan(mooring_s) & ~np.isnan(smos_s)
	mooring_c, cryosat_c = [arr[nan_mask_c] for arr in (mooring_c, cryosat_c)]
	mooring_s, smos_s = [arr[nan_mask_s] for arr in (mooring_s, smos_s)]
 
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	products = {'UiT': cryosat_c, "SMOS": smos_s}
	results = {p: {} for p in products}
 
	for i in range(len(bins)-1):
		lo, hi = bins[i], bins[i+1]
		label = labels[i]
		bin_mask_c = (mooring_c >= lo) & (mooring_c < hi)
		bin_mask_s = (mooring_s >= lo) & (mooring_s < hi)
    
		cryo_bin = cryosat_c[bin_mask_c]
		smos_bin = smos_s[bin_mask_s]
		mooring_c_bin = mooring_c[bin_mask_c]
		mooring_s_bin = mooring_s[bin_mask_s]
  
		arrs = {
			'UiT': cryo_bin,
			'SMOS': smos_bin
		}
  
		mooring_for_product = {
			'UiT': mooring_c_bin,
			'SMOS': mooring_s_bin
		}
  
		raw = {}
		for p, arr in arrs.items():
			mooring_ref = mooring_for_product[p]
			b = np.mean(arr - mooring_ref)
			r = np.sqrt(np.mean((arr - mooring_ref)**2))
			c = np.corrcoef(arr, mooring_ref)[0,1]
			raw[p] = dict(bias=b, rmse=r, cc=c)

		# normalize each metric across products in this bin
		for metric in ('bias','rmse'):
			vals = np.array([ raw[p][metric] for p in products ])
			mn, mx = vals.min(), vals.max()
			span = (mx - mn) if mx!=mn else 1.0
			for p in products:
				raw[p]['N'+metric] = (raw[p][metric] - mn) / span

		# compute DISO for each product
		for p in products:
			NB = raw[p]['Nbias']
			NR = raw[p]['Nrmse']
			NC = raw[p]['cc']
			raw[p]['DISO'] = np.sqrt(NB**2 + NR**2 + (NC - 1.0)**2)
			# store
			results[p][label] = raw[p]
   
	for prod, bins in results.items():
		print(f"\n{prod}:")
		for bin_label, m in bins.items():
			print(f"  {bin_label}: Bias={m['bias']:.3f}, RMSE={m['rmse']:.3f}, "
				  f"CC={m['cc']:.3f}, DISO={m['DISO']:.3f}")
   
def comp_stat_metric():
	mooring_A_draft_C, cryosat_A_draft_c = valid_mask(mooring_A_draft, cryosat_A_draft)
	mooring_B_draft_C, cryosat_B_draft_c = valid_mask(mooring_B_draft, cryosat_B_draft)
	mooring_D_draft_C, cryosat_D_draft_c = valid_mask(mooring_D_draft, cryosat_D_draft)
 
	mooring_A_draft_S, smos_A_draft_s = valid_mask(mooring_A_draft, smos_A_draft)
	mooring_B_draft_S, smos_B_draft_s = valid_mask(mooring_B_draft, smos_B_draft)
	mooring_D_draft_S, smos_D_draft_s = valid_mask(mooring_D_draft, smos_D_draft)
 
	mooring_c = np.concatenate([mooring_A_draft_C, mooring_B_draft_C, mooring_D_draft_C])
	cryosat_c = np.concatenate([cryosat_A_draft_c, cryosat_B_draft_c, cryosat_D_draft_c])
 
	mooring_s = np.concatenate([mooring_A_draft_S, mooring_B_draft_S, mooring_D_draft_S])
	smos_s = np.concatenate([smos_A_draft_s, smos_B_draft_s, smos_D_draft_s])

	nan_mask_c = ~np.isnan(mooring_c) & ~np.isnan(cryosat_c)  
	nan_mask_s = ~np.isnan(mooring_s) & ~np.isnan(smos_s)
	mooring_c, cryosat_c = [arr[nan_mask_c] for arr in (mooring_c, cryosat_c)]
	mooring_s, smos_s = [arr[nan_mask_s] for arr in (mooring_s, smos_s)]
 
	bins = [0, 0.4, 1]
	labels = ['0-0.4', '0.4-1']
 
	products = {'UiT': cryosat_c, "SMOS": smos_s}
	results = {p: {} for p in products}
 
	for i in range(len(bins)-1):
		lo, hi = bins[i], bins[i+1]
		label = labels[i]
		bin_mask_c = (mooring_c >= lo) & (mooring_c < hi)
		bin_mask_s = (mooring_s >= lo) & (mooring_s < hi)
    
		cryo_bin = cryosat_c[bin_mask_c]
		smos_bin = smos_s[bin_mask_s]
		mooring_c_bin = mooring_c[bin_mask_c]
		mooring_s_bin = mooring_s[bin_mask_s]
  
		arrs = {
			'UiT': cryo_bin,
			'SMOS': smos_bin
		}
  
		mooring_for_product = {
			'UiT': mooring_c_bin,
			'SMOS': mooring_s_bin
		}
  
		raw = {}
		for p, arr in arrs.items():
			mooring_ref = mooring_for_product[p]
			b = np.mean(arr - mooring_ref)
			r = np.sqrt(np.mean((arr - mooring_ref)**2))
			c = np.corrcoef(arr, mooring_ref)[0,1]
			raw[p] = dict(bias=b, rmse=r, cc=c)

		# normalize each metric across products in this bin
		for metric in ('bias','rmse'):
			vals = np.array([ raw[p][metric] for p in products ])
			mn, mx = vals.min(), vals.max()
			span = (mx - mn) if mx!=mn else 1.0
			for p in products:
				raw[p]['N'+metric] = (raw[p][metric] - mn) / span

		# compute DISO for each product
		for p in products:
			NB = raw[p]['Nbias']
			NR = raw[p]['Nrmse']
			NC = raw[p]['cc']
			raw[p]['DISO'] = np.sqrt(NB**2 + NR**2 + (NC - 1.0)**2)
			# store
			results[p][label] = raw[p]
   
	for prod, bins in results.items():
		print(f"\n{prod}:")
		for bin_label, m in bins.items():
			print(f"  {bin_label}: Bias={m['bias']:.3f}, RMSE={m['rmse']:.3f}, "
				  f"CC={m['cc']:.3f}, DISO={m['DISO']:.3f}")

 
 
if __name__ == "__main__":
	#mooring_locations()
	#histogram_mooring()
 
	#times_series_all()
	bar_hist_plot()
	#box_scatter()
	#scatter_plot()
	#total_scatter_plot()
 
	#single_anomaly()
	#draft_anomalies()
	#total_draft_anomalies()
	#full_stat_metric()
	#comp_stat_metric()