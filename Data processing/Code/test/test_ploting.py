import os
import glob
import numpy as np
import pandas as pd
import h5py
import scipy.io
import datetime

mooring_folder = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data converted"




def get_mooring_data(path):
	mooring_files_A = glob.glob(os.path.join(path, "*a_*.mat"))
	mooring_files_B = glob.glob(os.path.join(path, "*b_*.mat"))
	mooring_files_D = glob.glob(os.path.join(path, "*d_*.mat"))
	
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

