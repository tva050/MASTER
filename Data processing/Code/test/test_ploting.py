import os
import glob
import numpy as np
import pandas as pd
import h5py
import scipy.io

ref_folder = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data mat"
bgep_files_A = glob.glob(os.path.join(ref_folder, "*a_*.mat"))

IDS_full_A = []
t_full_A = []

for file in bgep_files_A:
    try:
        # Try opening as v7.3 (HDF5 format)
        with h5py.File(file, "r") as f:
            print(f"Processing HDF5 file: {file}")
            print(list(f.keys()))  # Inspect available keys
            if "dates" in f:
                dates = np.array(f["dates"]).ravel()  # Use ravel instead of flatten
                date_unit = f["dates"].attrs.get('UNIT', None)
                if date_unit == 'D':  # MATLAB serial date (days)
                    t_full_A.extend(pd.to_datetime(dates - 719529, unit="D", origin="unix"))
                else:
                    print(f"Unexpected date unit '{date_unit}' in {file}. Skipping conversion.")
                    t_full_A.extend(dates)
            if "IDS" in f:
                IDS_sub = np.array(f["IDS"]).T  # Transpose for correct format
                IDS_full_A.append(IDS_sub[:, :6])
    except OSError:
        # If HDF5 fails, try loading as an older MATLAB format
        try:
            print(f"Trying scipy.io for non-HDF5 file: {file}")
            mat_data = scipy.io.loadmat(file)
            print(f"Loaded using scipy.io: {list(mat_data.keys())}")
            if "dates" in mat_data:
                dates = mat_data["dates"].flatten()
                # Check if dates are numerical, if not convert them
                if dates.dtype.kind in 'iuf':  # Check if dates are integer, unsigned, or float
                    t_full_A.extend(pd.to_datetime(dates - 719529, unit="D", origin="unix"))
                else:
                    print(f"Dates in {file} are not numerical. Skipping conversion.")
            if "IDS" in mat_data:
                IDS_sub = mat_data["IDS"]
                IDS_full_A.append(IDS_sub[:, :6])  # Extract first 6 columns
        except Exception as e:
            print(f"Error processing file {file} with scipy.io: {e}")
# Convert lists to NumPy arrays
IDS_full_A = np.vstack(IDS_full_A) if IDS_full_A else np.array([])
t_full_A = np.array(t_full_A)
# Print first few entries
print("First 5 IDS rows:", IDS_full_A[:5])
print("First 5 Dates:", t_full_A[:5])



