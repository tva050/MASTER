import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from cartopy import crs as ccrs, feature as cfeature
import netCDF4 as nc
from  scipy.interpolate import griddata
import pandas as pd
from pyproj import Proj, transform

paths = [
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130321.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130322.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130323.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130324.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130326.txt",
	r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\Operation IceBridge\IDCSI4_20130425.txt"
]


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
