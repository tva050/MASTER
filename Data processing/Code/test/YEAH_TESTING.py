import os
import glob
import numpy as np
from scipy.io import loadmat
from datetime import datetime
from datetime import timedelta, datetime
import h5py
import pandas as pd


path = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data mat"

#mooring_12a = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\extracted_data.mat"
#mooring_10a = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\uls10a_dailyn.mat"
mooring_10a = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data converted\Vuls10a_dailyn.mat"
mooring_12a = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data converted\Vuls12a_dailyn.mat"
mooring_16a = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data converted\Vuls16a_dailyn.mat"
mooring_21a = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data converted\Vuls21a_dailyn.mat"
def mooring_data(mooring_path):
	""" 
	- Extract matlab files from path 
	- The files are stored in a single data set for each mooring (a, b, d)
	- and with a single data set for each mooring time
	- and with a single data set for each mooring ice draft statistics
		- ice draft statistics: number, mean, std, minimum, maximum, median
	- Filter out data which does not cover the months 01, 02, 03, 04, 10, 11, and 12
	"""
	mooring_files_A = glob.glob(os.path.join(mooring_path, "*a_*.mat"))
	mooring_files_B = glob.glob(os.path.join(mooring_path, "*b_*.mat"))
	mooring_files_D = glob.glob(os.path.join(mooring_path, "*d_*.mat"))
	
	ids_A, ids_B, ids_D = [], [], []
	dates_A, dates_B, dates_D = [], [], []
	stats_A, stats_B, stats_D = [], [], []
	def process_files(files, ids, dates, stats):
		for file in files:
			data = loadmat(file)
			# Assuming 'ice_draft' and 'time' are keys in the MATLAB file
			ice_draft = data.get('ice_draft', [])
			time = data.get('time', [])
			
			# Convert MATLAB time to Python datetime
			time = [datetime.fromordinal(int(t)) + timedelta(days=t%1) - timedelta(days=366) for t in time]
			
			# Filter by months
			filtered_indices = [i for i, t in enumerate(time) if t.month in [1, 2, 3, 4, 10, 11, 12]]
			filtered_draft = [ice_draft[i] for i in filtered_indices]
			filtered_time = [time[i] for i in filtered_indices]
			
			# Calculate statistics
			if len(filtered_draft) > 0:
				filtered_draft_array = np.array(filtered_draft)
				stats.append({
					'number': len(filtered_draft),
					'mean': np.mean(filtered_draft_array),
					'std': np.std(filtered_draft_array),
					'minimum': np.min(filtered_draft_array),
					'maximum': np.max(filtered_draft_array),
					'median': np.median(filtered_draft_array)
				})
				ids.append(file)
				dates.append(filtered_time)
	
	process_files(mooring_files_A, ids_A, dates_A, stats_A)
	process_files(mooring_files_B, ids_B, dates_B, stats_B)
	process_files(mooring_files_D, ids_D, dates_D, stats_D)
	return {
		'A': {'ids': ids_A, 'dates': dates_A, 'stats': stats_A},
		'B': {'ids': ids_B, 'dates': dates_B, 'stats': stats_B},
		'D': {'ids': ids_D, 'dates': dates_D, 'stats': stats_D}
	}
	

def easy_extract(path):
	mooring_files_A = glob.glob(os.path.join(path, "*a_*.mat"))
	mooring_files_B = glob.glob(os.path.join(path, "*b_*.mat"))
	mooring_files_D = glob.glob(os.path.join(path, "*d_*.mat"))
	print(f"Found {len(mooring_files_A)} A files, {len(mooring_files_B)} B files, and {len(mooring_files_D)} D files.")
	ids_A, ids_B, ids_D = [], [], []
	dates_A, dates_B, dates_D = [], [], []

	for file in mooring_files_A:
		print(f"Processing A file: {file}")
		try:
			data = loadmat(file)
			ids_A.append(data["IDS"])
			dates_A.append(data["dates"])
		except Exception as e:
			print(f"Error processing A file {file}: {e}")

def process_single_file_scipy(path):
	file = loadmat(path)
	print(f"whole: {file}")
	print(f"keys: {file.keys()}")
	# Extract data from the file
	try:
		ids = file["IDS"]
		dates = file["dates"]
		# Convert MATLAB serial date to pandas datetime
		dates = pd.to_datetime(dates - 719529, unit="D", origin="unix")
	except KeyError as e:
		print(f"Key error: {e} in file {path}")
	except Exception as e:
		print(f"Error processing file {path}: {e}")
	print(ids.shape, dates.shape)
 
def single_file(path):
	file = loadmat(path)
	print(file.keys())
	ids = file["IDS"]
	dates = file["dates"]
	ids_a = ids[:, :6]

 
def extraction(path):
	mooring_files_A = glob.glob(os.path.join(path, "*a_*.mat"))
	mooring_files_B = glob.glob(os.path.join(path, "*b_*.mat"))
	mooring_files_D = glob.glob(os.path.join(path, "*d_*.mat"))
	
	ids_A, ids_B, ids_D = [], [], []
	dates_A, dates_B, dates_D = [], [], []
	stats_A, stats_B, stats_D = [], [], []
	
	for files in mooring_files_A:
		data = loadmat(files)



#easy_extract(path)
single_file(mooring_21a)
#process_single_file_scipy(mooring_12a)
#process_single_file_h5(mooring_12a)
#extraction(path)