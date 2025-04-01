import os
import glob
import numpy as np
import pandas as pd
import h5py
import scipy.io
import datetime

""" 
Plan 
 - Extract the files for corresponding mooring, ✔️
	- these are represtented in the file name as "**a_**.mat", "**b_**.mat" and "**c_**.mat"
 - create an single data set for each mooring (a, b, d), which includes IDS and dates ✔️
	- Filter out data which not covers the month 01, 02, 03, 04, 10, 11, and 12
	- These are the only valid months for Cryosat and SMOS
 - IDS: daily ice draft statistics: number, mean, std, minimum, maximum, median  ✔️
 - Extract, satellite data SMOS and Cryo  
 - Grid mooring and smos satellite to the same grid (Cryo grid, 25km)
 - Calculate referance monthly mean ice draft thickness from the moorings
 - Identify the satellite grid cells within the search radius of moorings
 - Satellite grid cells within the search radius is averaged 
 - Convert the derived sea ice freeboard to draft (possible earlier)
	- sid = sit - ifb
 - Plot time series
	- Ranging from 0.9 - 1m draft 
	
Note:
Variables in the .mat file: ['BETA', 'ID', 'IDS', 'WLS', 'TILTS', 'OWBETA', 'BTBETA', 'WL', 'T', 'yday', 'dates', 'name']
"""

mooring_folder = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data converted"
cs_uit_folder = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\CryoSat-2\UiT product"

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

def get_cyro(folder_path):
    # Get the CryoSat-2 nc data, making them into a single data set for an time series
    cryo_files = glob.glob(os.path.join(folder_path, "*.nc"))
    cryo_files.sort()  # Sort files by date