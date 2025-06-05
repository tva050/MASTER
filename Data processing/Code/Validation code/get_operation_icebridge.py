import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.path as mpath
from matplotlib.ticker import PercentFormatter
import seaborn as sns


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
	si_thickness = data.variables['mean_ice_thickness'][:]
	si_thickness_un = data.variables['uncertainty'][:]
 
	#mask = ~np.isnan(si_thickness_un)
	#si_thickness_un = np.where(mask, si_thickness_un, np.nan)
	
	#print(si_thickness.shape)
	return lat, lon, si_thickness, si_thickness_un


def get_UiT(path):
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


# Need to all data from all years from both OIB, SMOS and UiT, maybe save each year in a dict
# for the corresponding product, for making it easier to handle 

def get_all_data():
	oib_2011 = extract_all_oib(oib_paths_2011)
	oib_2012 = extract_all_oib(oib_paths_2012)
	oib_2013 = extract_all_oib(oib_paths_2013)
	oib_2014 = extract_all_oib(oib_paths_2014)
	oib_2015 = extract_all_oib(oib_paths_2015)
	oib_2017 = extract_all_oib(oib_paths_2017)

	smos_2011 = get_smos(smos_2011_path)
	smos_2012 = get_smos(smos_2012_path)
	smos_2013 = get_smos(smos_2013_path)
	smos_2014 = get_smos(smos_2014_path)
	smos_2015 = get_smos(smos_2015_path)
	smos_2017 = get_smos(smos_2017_path)

	uit_2011 = get_UiT(uit_2011_path)
	uit_2012 = get_UiT(uit_2012_path)
	uit_2013 = get_UiT(uit_2013_path)
	uit_2014 = get_UiT(uit_2014_path)
	uit_2015 = get_UiT(uit_2015_path)
	uit_2017 = get_UiT(uit_2017_path)

	return {
		'oib': {
			'lat': [oib[0] for oib in [oib_2011, oib_2012, oib_2013, oib_2014, oib_2015, oib_2017]],
			'lon': [oib[1] for oib in [oib_2011, oib_2012, oib_2013, oib_2014, oib_2015, oib_2017]],
			'sit': [oib[2] for oib in [oib_2011, oib_2012, oib_2013, oib_2014, oib_2015, oib_2017]],
			'sit_un': [oib[3] for oib in [oib_2011, oib_2012, oib_2013, oib_2014, oib_2015, oib_2017]],	
		},
		'smos': {
			'lat': [smos[0] for smos in [smos_2011, smos_2012, smos_2013, smos_2014, smos_2015, smos_2017]],
			'lon': [smos[1] for smos in [smos_2011, smos_2012, smos_2013, smos_2014, smos_2015, smos_2017]],
			'sit': [smos[2] for smos in [smos_2011, smos_2012, smos_2013, smos_2014, smos_2015, smos_2017]],
			'sit_un': [smos[3] for smos in [smos_2011, smos_2012, smos_2013, smos_2014, smos_2015, smos_2017]],	
		},
		'uit': {
			'lat': [uit[0] for uit in [uit_2011, uit_2012, uit_2013, uit_2014, uit_2015, uit_2017]],
			'lon': [uit[1] for uit in [uit_2011, uit_2012, uit_2013, uit_2014, uit_2015, uit_2017]],
			'sit': [uit[2] for uit in [uit_2011, uit_2012, uit_2013, uit_2014, uit_2015, uit_2017]],
			'sit_un': [uit[3] for uit in [uit_2011, uit_2012, uit_2013, uit_2014, uit_2015, uit_2017]],	
		}
	}
 
# ------------------------------ Data processing ------------------------------

def reprojecting(lon, lat, proj=ccrs.NorthPolarStereo()):
	transformer = proj.transform_points(ccrs.PlateCarree(), lon, lat)
	x = transformer[..., 0]
	y = transformer[..., 1]
	return x, y

# reprojecting all data to the same projection

def reproject_all_data(data):
	out = {}
	for key in ('oib','smos','uit'):
		# each of these is currently a list of arrays [yr1, yr2, …]
		lats_list   = data[key]['lat']
		lons_list   = data[key]['lon']
		sit_list    = data[key]['sit']
		sit_un_list = data[key]['sit_un']
		
		# concatenate into one big 1-D array per variable
		all_lats   = np.concatenate(lats_list)
		all_lons   = np.concatenate(lons_list)
		all_sit    = np.concatenate(sit_list)
		all_sit_un = np.concatenate(sit_un_list)
		
		# now these are true numpy arrays → OK to reproject
		x, y = reprojecting(all_lons, all_lats)
		
		out[key] = (x, y, all_sit, all_sit_un)
	return out

def resample_to_cryo_grid(x_source, y_source, source_sit, source_sit_un, x_target, y_target, radius=12500):
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

# Resample all data to the cryosat (UiT) grid

def resample_all_data(data, radius=12500):
	oib_x, oib_y, oib_sit, oib_sit_un = data['oib']
	smos_x, smos_y, smos_sit, smos_sit_un = data['smos']
	uit_x, uit_y, uit_sit, uit_sit_un = data['uit']

	# Resample to UiT grid
	resampled_oib = resample_to_cryo_grid(oib_x, oib_y, oib_sit, oib_sit_un, uit_x, uit_y, radius)
	resampled_smos = resample_to_cryo_grid(smos_x, smos_y, smos_sit, smos_sit_un, uit_x, uit_y, radius)

	return {
		'oib': resampled_oib,
		'smos': resampled_smos,
		'uit': (uit_x, uit_y, uit_sit),
	}
 
plt.rcParams.update({
		'font.family':      'serif',
		'font.size':         12,
		'axes.labelsize':    12,
		'xtick.labelsize':   11,
		'ytick.labelsize':   11,
		'legend.fontsize':   12,
		'figure.titlesize':  12,
})
 
def plot_flight_paths_all(data):
	"""
	Plot all OIB flight paths on a NorthPolarStereo map,
	first as a combined background, then year-by-year.
	"""
	
	# Unpack
	oib_lats = data['oib']['lat']
	oib_lons = data['oib']['lon']
	# Make sure your years list matches the order in get_all_data()
	years = ['2011','2012','2013','2014','2015','2017']
	assert len(years) == len(oib_lats), "Years list must match data length"

	# Combine everything for a grey “backdrop”
	all_lats = np.concatenate(oib_lats)
	all_lons = np.concatenate(oib_lons)

	# Create figure and axis with North Polar Stereographic projection
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
	
	#ax.add_feature(cfeature.LAND, color='lightgray', zorder=2)
	#ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor='white', linewidth=0.5, zorder=3)
	#ax.add_feature(cfeature.COASTLINE, color='gray', linewidth=0.5, zorder=4)

	# 1) All flight paths combined, light grey
	#ax.scatter(all_lons, all_lats, s=0.5, color='lightgray', label='All OIB', transform=ccrs.PlateCarree(), zorder=6)

	# 2) Year-by-year on top
	# choose a colormap with enough distinct entries
	cmap = plt.get_cmap('plasma', len(years))
	for idx, (yr, lats, lons) in enumerate(zip(years, oib_lats, oib_lons)):
		ax.scatter(lons, lats, s=1, linewidth=0, color=cmap(idx), label=f'OIB {yr}', transform=ccrs.PlateCarree(), zorder=7)


	#meridians = np.arange(10, 360, 20)
	#for lon in meridians:
	#	latitudes = np.linspace(60, 90, 100)  # or 0 to 90 if you want full lines
	#	longitudes = np.full_like(latitudes, lon)
	#	ax.plot(longitudes, latitudes,
	#			transform=ccrs.PlateCarree(),
	#			color='gray', linewidth=0.5, zorder=0)

	#plt.legend(markerscale=5, loc='lower left', ncols = 2, fontsize=10, handletextpad=0.3, bbox_to_anchor=(0., 1.0), borderaxespad=0.0)
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Thesis\OIB_all_flight_paths.png", dpi=300, bbox_inches='tight', transparent=True)
	plt.show()


def histogram_oib(data):
	""" 
	Plot an histogram of the OIB sit data,
	showing the distribution of the data.
	"""
	# Unpack
	oib_sit = data['oib']['sit']

	# Flatten the list of arrays into a single array
	all_sit = np.concatenate(oib_sit)
 
	# make an trancparent red box, to display the area of interes between 0 and 1m
	

	# Create histogram
	plt.figure(figsize=(6.733, 3.5))
	plt.hist(all_sit, bins=50, label='OIB SIT', color='black', alpha=0.7, weights=np.ones_like(all_sit) / len(all_sit))
	plt.axvspan(0, 1, color='red', alpha=0.3, label='Area of interest (0-1 m)')
	#plt.axvline(x=1, color='red', linestyle='--', alpha=0.5)
	# add thicks inside the histogram
	plt.tick_params(axis='both', direction='in')
	plt.xlim(0, 15)
	plt.xlabel('Sea Ice Thickness (m)')
	plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
	plt.ylabel('Observations (%)')
	plt.legend(loc='lower left', bbox_to_anchor=(0., 1.01, 1., .102), frameon=False, ncol=2, mode="expand", borderaxespad=0.0, handletextpad=0.3)
	plt.grid()
	plt.savefig(r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\OIB validations - CryoSat and SMOS\OIB_SIT_histogram.png", dpi=300)
	#plt.show()
 

 
 
def plot_sit(data, cryo_lat, cryo_lon):
	oib_sit = data['oib']
	
	all_sit = oib_sit.flatten()
	mask = (all_sit >= 0) & (all_sit <= 1)
	all_sit = all_sit[mask]
	plt.figure(figsize=(10, 6))
	plt.hist(all_sit, bins=50, label='OIB SIT', color='black', alpha=0.7)
	#plt.axvspan(0, 1, color='red', alpha=0.3, label='Area of interest (0-1 m)')
	plt.tick_params(axis='both', direction='in')
	plt.xlim(0, 15)
	plt.xlabel('Sea Ice Thickness (m)')
	plt.ylabel('Observations (counts)')
	plt.title('filtered OIB + Quicklook SIT (25km grid)')
	plt.legend()
	plt.grid()
	plt.show()
 
 
	fig = plt.figure(figsize=(10,6))
	ax  = fig.add_subplot(1,1,1, projection=ccrs.NorthPolarStereo())
	ax.set_extent([-3e6, 3e6, -3e6, 3e6], ccrs.NorthPolarStereo())

	# map features
	ax.coastlines()
	ax.add_feature(cfeature.LAND, facecolor="gray", alpha=1,   zorder=2)
	ax.add_feature(cfeature.OCEAN,facecolor="lightgray",alpha=0.5,zorder=1)
	ax.add_feature(cfeature.LAKES,edgecolor='gray',facecolor="white",
				   linewidth=0.5,alpha=0.5,zorder=3)
	ax.add_feature(cfeature.RIVERS,edgecolor='lightgray',facecolor="white",
				   linewidth=0.5,alpha=0.5,zorder=4)
	ax.add_feature(cfeature.COASTLINE,color="black",linewidth=0.1,zorder=5)
	ax.gridlines(draw_labels=True, color="dimgray", zorder=7)
	
	cryo_lon = np.array(cryo_lon)
	cryo_lat = np.array(cryo_lat)
	#mesh=ax.pcolormesh(cryo_lon, cryo_lat, oib_sit, transform=ccrs.PlateCarree(), cmap='viridis', zorder=6)
	#plt.colorbar(mesh, ax=ax, label='Sea Ice Thickness (m)')
	scatter = ax.scatter(cryo_lon, cryo_lat, c=oib_sit, cmap='viridis', s=1, transform=ccrs.PlateCarree(), zorder=6)
	plt.colorbar(scatter, ax=ax, label='Sea Ice Thickness (m)')
	plt.title('OIB Sea Ice Thickness (25km grid)')
	plt.show()
	

#-- Plotting functions -- 
 
def bar_hist_plot(data):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
	
	# unpacking the resampled data
	oib_sit = data['oib']
	smos_sit = data['smos']
	cryo_sit = data['uit'][2]  # Assuming the third element is the SIT data
	
	# Flatten the list of arrays into a single array
	oib = np.concatenate(oib_sit)
	smos = np.concatenate(smos_sit)
	cryo = np.concatenate(cryo_sit)
	
	nan_mask = ~np.isnan(oib) & ~np.isnan(smos) & ~np.isnan(cryo)
	oib, smos, cryo = [arr[nan_mask] for arr in (oib, smos, cryo)]

	pr_range_mask = (cryo >= 0) & (cryo <= 1) & (smos >= 0) & (smos <= 1)
	cryo = cryo[pr_range_mask]
	smos = smos[pr_range_mask]
	oib = oib[pr_range_mask]
	print(f"Number of valid data points: {len(oib)}")
	smos_means = []
	cryo_means = []
	for i in range(len(bins)-1): #
		bin_mask = (oib >= bins[i]) & (oib < bins[i+1])
		#bin_mask_cryo = (oib_c >= bins[i]) & (oib_c < bins[i + 1])
		smos_means.append(np.mean(smos[bin_mask])) 
		cryo_means.append(np.mean(cryo[bin_mask]))

	# Set up the figure with one main plot and two smaller plots
	fig = plt.figure(figsize=(6.733, 6))

	# Layout parameters
	box_size = 0.35
	main_height = 0.5
	gap = (1 - 2 * box_size) / 3
	gap_main = (1 - main_height - box_size) / 3

	# Create axes
	ax_main = fig.add_axes([gap, 2 * gap_main + box_size, 1 - 2 * gap, main_height])
	ax_left = fig.add_axes([gap, gap_main-0.06, box_size, box_size-0.001])
	ax_right = fig.add_axes([2 * gap + box_size, gap_main-0.06, box_size, box_size-0.001])

	# --- Main plot: Bar plot ---
	x = np.arange(len(bin_labels))
	width = 0.35  # width of the bars

	ax_main.bar(x - width/2, smos_means, width, label='SMOS', color='blue', alpha=0.7)
	ax_main.bar(x + width/2, cryo_means, width, label='UiT', color='green', alpha=0.7)
	ax_main.set_ylabel('Mean SIT [m]')
	ax_main.set_xlabel('OIB SIT bins [m]')
	ax_main.tick_params(axis='both', direction='in')
	ax_main.set_xticks(x)
	ax_main.set_xticklabels(bin_labels)
	ax_main.legend(bbox_to_anchor=(0.4, 1.1), loc='upper right', ncol=2, borderaxespad=0.0, handletextpad=0.3, frameon=False)
	ax_main.grid(axis='y', linestyle='--', alpha=0.5)
 
	# histogram mask
	range_mask = (oib >= 0) & (oib <= 1) 
	#range_mask_c = (oib_c >= 0) & (oib_c <= 1)
	oib_m, smos_m, cryo_m = oib[range_mask], smos[range_mask], cryo[range_mask]
	#oib_m_c = oib_c[range_mask_c]
	bin_edges=np.linspace(0,1,11)
 
	# --- Bottom right plot: Histogram OIB vs Cryo ---
	ax_right.hist(oib_m, bins=bin_edges, alpha=0.7, label='OIB', color='black', density=True)
	ax_right.hist(cryo_m, bins=bin_edges, edgecolor='green', color="green", fill=True, linewidth=1, hatch='xx', alpha=0.7, label='UiT', density=True)
	ax_right.tick_params(axis='both', direction='in')
	ax_right.set_xlabel('SIT [m]')
	ax_right.yaxis.set_major_formatter(PercentFormatter(1))
	ax_right.grid(alpha=0.5, linestyle='--')
	ax_right.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2, mode="expand", borderaxespad=0.0, handletextpad=0.3, frameon=False) 
	
	# --- Bottom left plot: Histogram OIB vs SMOS ---
	ax_left.hist(oib_m, bins=bin_edges, alpha=0.7, label='OIB', color='black', density=True)
	ax_left.hist(smos_m, bins=bin_edges, edgecolor='blue', color="blue", fill=True, linewidth=1, hatch='xx', alpha=0.7, label='SMOS', density=True)
	ax_left.set_xlabel('SIT [m]')
	ax_left.set_ylabel('Counts')
	ax_left.tick_params(axis='both', direction='in')
	ax_left.yaxis.set_major_formatter(PercentFormatter(1))
	ax_left.grid(alpha=0.5, linestyle='--')
	ax_left.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2, mode="expand", borderaxespad=0.0, handletextpad=0.3, frameon=False)

	plt.savefig(r'C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\OIB validations - CryoSat and SMOS\OIB_bar_histo_plot_with_cryo_mask.png', dpi=300, bbox_inches='tight')
	#plt.show()
	
	
def box_plot(data):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
	
	# unpacking the resampled data
	oib_sit = data['oib']
	smos_sit = data['smos']
	cryo_sit = data['uit'][2]  # Third element is the SIT data

	# Flatten and clean
	oib, smos, cryo = oib_sit.flatten(), smos_sit.flatten(), cryo_sit.flatten()
	nan_mask = ~np.isnan(oib) & ~np.isnan(smos) & ~np.isnan(cryo)
	oib, smos, cryo = [arr[nan_mask] for arr in (oib, smos, cryo)]

	# Restrict to physical range
	pr_range_mask = (cryo >= 0) & (cryo <= 1) & (smos >= 0) & (smos <= 1)
	cryo = cryo[pr_range_mask]
	smos = smos[pr_range_mask]
	oib = oib[pr_range_mask]
 
	print(f"Number of valid data points: {len(oib)}")

	# Bin the data
	binned_oib_cryo_data, binned_oib_smos_data = [], []
	for i in range(len(bins) - 1):
		bin_mask = (oib >= bins[i]) & (oib < bins[i + 1])
		binned_oib_cryo_data.append(cryo[bin_mask])
		binned_oib_smos_data.append(smos[bin_mask])

	# --- Plot ---
	fig, axs = plt.subplots(1, 2, figsize=(6.733, 3.6))
	ax1, ax2 = axs

	# Box 1: UiT vs OIB
	ax1.boxplot(binned_oib_cryo_data, labels=bin_labels,
				medianprops=dict(color='black'),
				meanprops=dict(color="#f7022a"), showmeans=True, meanline=True,
				patch_artist=True, showfliers=False,
				boxprops=dict(facecolor='lightgray', alpha=0.6))
	ax1.plot([], [], "--", linewidth=1, color="#f7022a", label="Mean")
	ax1.plot([], [], "-", linewidth=1, color="black", label="Median")

	for i, data in enumerate(binned_oib_cryo_data):
		x = np.random.normal(i + 1, 0.05, size=len(data))
		ax1.scatter(x, data, color='green', alpha=0.7, edgecolor="none", linewidth=0.)

	ax1.set_ylabel('UiT SIT [m]')
	ax1.set_xlabel('OIB SIT bins [m]')
	ax1.tick_params(axis='both', direction='in')
	ax1.grid(axis='y', linestyle='--', alpha=0.5)
	ax1.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.08),
			   ncol=2, borderaxespad=0.0, handletextpad=0.3)

	# Box 2: SMOS vs OIB
	ax2.boxplot(binned_oib_smos_data, labels=bin_labels,
				medianprops=dict(color='black'),
				meanprops=dict(color="#f7022a"), showmeans=True, meanline=True,
				patch_artist=True, showfliers=False,
				boxprops=dict(facecolor='lightgray', alpha=0.6))
	ax2.plot([], [], "--", linewidth=1, color="#f7022a", label="Mean")
	ax2.plot([], [], "-", linewidth=1, color="black", label="Median")

	for i, data in enumerate(binned_oib_smos_data):
		x = np.random.normal(i + 1, 0.05, size=len(data))
		ax2.scatter(x, data, color='blue', alpha=0.7, edgecolor="none", linewidth=0.)

	ax2.set_ylabel('SMOS SIT [m]')
	ax2.set_xlabel('OIB SIT bins [m]')
	ax2.tick_params(axis='both', direction='in')
	ax2.grid(axis='y', linestyle='--', alpha=0.5)
	
	plt.subplots_adjust(left=0.076, right=1, top=0.955, bottom=0.112, wspace=0.2)
	plt.savefig(r'C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Figures\OIB validations - CryoSat and SMOS\OIB_box_scatter_plot_WithCryoRangeFiltering.png', dpi=300, bbox_inches='tight')
	#plt.show()

def heat_map(data):
	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
	satellite_products = ["CryoSat-2", "SMOS"]
 
	# Unpack
	oib_sit = data['oib']
	smos_sit = data['smos']
	cryo_sit = data['uit'][2]  # Assuming the third element is the SIT data

	# Flatten the list of arrays into a single array
	oib = np.concatenate(oib_sit)
	smos = np.concatenate(smos_sit)
	cryo = np.concatenate(cryo_sit)

	nan_mask = ~np.isnan(oib) & ~np.isnan(smos) & ~np.isnan(cryo)
	oib, smos, cryo = [arr[nan_mask] for arr in (oib, smos, cryo)]
 
	pr_range_mask = (cryo >= 0) & (cryo <= 1) & (smos >= 0) & (smos <= 1)
	cryo = cryo[pr_range_mask]
	smos = smos[pr_range_mask]
	oib = oib[pr_range_mask]
 
	mean_difference = np.zeros((len(satellite_products), len(bins) - 1))
	for i in range(len(bins) - 1):
		bin_mask = (oib >= bins[i]) & (oib < bins[i + 1])
		mean_difference[0, i] = np.mean(cryo[bin_mask]) - np.mean(oib[bin_mask])
		mean_difference[1, i] = np.mean(smos[bin_mask]) - np.mean(oib[bin_mask])
  
	plt.figure(figsize=(10,5))
	ax = sns.heatmap(mean_difference, annot=True, cmap="plasma", fmt=".2f", xticklabels=bin_labels, yticklabels=satellite_products, center=0, cbar_kws={"label": "Mean Difference [m]"})
	plt.xlabel('OIB SIT bins [m]')
	plt.show()
 
def stat_matrics_all(data):
	# Unpack
	oib_sit = data['oib']
	smos_sit = data['smos']
	cryo_sit = data['uit'][2]  # Assuming the third element is the SIT data

	# Flatten the list of arrays into a single array
	oib = np.concatenate(oib_sit)
	smos = np.concatenate(smos_sit)
	cryo = np.concatenate(cryo_sit)

	nan_mask = ~np.isnan(oib) & ~np.isnan(smos) & ~np.isnan(cryo)
	oib, smos, cryo = [arr[nan_mask] for arr in (oib, smos, cryo)]
 
	pr_range_mask = (cryo >= 0) & (cryo <= 1) & (smos >= 0) & (smos <= 1)
	cryo = cryo[pr_range_mask]
	smos = smos[pr_range_mask]
	oib = oib[pr_range_mask]

	bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
	labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
 
	products = {'UiT': cryo, "SMOS": smos}
	results = {p: {} for p in products}
 
		# 4) Loop over bins
	for i in range(len(bins)-1):
		lo, hi = bins[i], bins[i+1]
		label = labels[i]

		# mask for this bin
		bin_mask = (oib >= lo) & (oib < hi)
		#bin_mask_cryo = (oib_c >= lo) & (oib_c < hi)

		if not np.any(bin_mask):
			# fill NaNs if empty
			for p in products:
				results[p][label] = dict(
					bias=np.nan, rmse=np.nan, cc=np.nan,
					Nbias=np.nan, Nrmse=np.nan, Ncc=np.nan,
					DISO=np.nan
				)
			continue

		oib_bin = oib[bin_mask]
		smos_bin = smos[bin_mask]
		cryo_bin = cryo[bin_mask]

		arrs = {
			'UiT': cryo_bin,
			'SMOS': smos_bin
		}
		oib_for_product = {
			'UiT': oib_bin,
			'SMOS': oib_bin
		}
  
		raw = {}
		for p, arr in arrs.items():
			oib_ref = oib_for_product[p]
			b = np.mean(arr - oib_ref)
			r = np.sqrt(np.mean((arr - oib_ref)**2))
			c = np.corrcoef(arr, oib_ref)[0,1]
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
	
def stat_matrics_comp(data):
	# Unpack
	oib_sit = data['oib']
	smos_sit = data['smos']
	cryo_sit = data['uit'][2]  # Assuming the third element is the SIT data

	# Flatten the list of arrays into a single array
	oib = np.concatenate(oib_sit)
	smos = np.concatenate(smos_sit)
	cryo = np.concatenate(cryo_sit)

	nan_mask = ~np.isnan(oib) & ~np.isnan(smos) & ~np.isnan(cryo)
	oib, smos, cryo = [arr[nan_mask] for arr in (oib, smos, cryo)]
 
	pr_range_mask = (cryo >= 0) & (cryo <= 1) & (smos >= 0) & (smos <= 1)
	cryo = cryo[pr_range_mask]
	smos = smos[pr_range_mask]
	oib = oib[pr_range_mask]

	bins = [0, 0.4, 1]
	labels = ['0.0-0.4', '0.4-1']
 
	products = {'UiT': cryo, "SMOS": smos}
	results = {p: {} for p in products}
 
		# 4) Loop over bins
	for i in range(len(bins)-1):
		lo, hi = bins[i], bins[i+1]
		label = labels[i]

		# mask for this bin
		bin_mask = (oib >= lo) & (oib < hi)
		#bin_mask_cryo = (oib_c >= lo) & (oib_c < hi)

		if not np.any(bin_mask):
			# fill NaNs if empty
			for p in products:
				results[p][label] = dict(
					bias=np.nan, rmse=np.nan, cc=np.nan,
					Nbias=np.nan, Nrmse=np.nan, Ncc=np.nan,
					DISO=np.nan
				)
			continue

		oib_bin = oib[bin_mask]
		smos_bin = smos[bin_mask]
		cryo_bin = cryo[bin_mask]

		arrs = {
			'UiT': cryo_bin,
			'SMOS': smos_bin
		}
		oib_for_product = {
			'UiT': oib_bin,
			'SMOS': oib_bin
		}
  
		raw = {}
		for p, arr in arrs.items():
			oib_ref = oib_for_product[p]
			b = np.mean(arr - oib_ref)
			r = np.sqrt(np.mean((arr - oib_ref)**2))
			c = np.corrcoef(arr, oib_ref)[0,1]
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
	data = get_all_data()
	reprojected_data = reproject_all_data(data)
	resampled_data = resample_all_data(reprojected_data, radius=12500)
	
	plot_flight_paths_all(data)
	#histogram_oib(data)
	#plot_sit(resampled_data, data['uit']['lat'], data['uit']['lon'])
 
	#bar_hist_plot(resampled_data)
	#box_plot(resampled_data)
	#heat_map(resampled_data)
	#stat_matrics_all(resampled_data)
	#stat_matrics_comp(resampled_data)
