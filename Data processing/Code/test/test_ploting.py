import os
import glob
import numpy as np
import pandas as pd
import h5py

ref_folder = r"C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\Data"

# Find all matching .mat files
bgep_files_A = glob.glob(os.path.join(ref_folder, "*a_*.mat"))

# Initialize empty lists
IDS_full_A = []
t_full_A = []

# Loop through each .mat file
for file in bgep_files_A:
    try:
        # Attempt to open the file using h5py (for version 7.3 .mat files)
        with h5py.File(file, "r") as f:
            print(f"Processing file: {file}")
            print(f.keys())  # Print the keys in the file to inspect

            # Check if the file contains 'dates' key
            if "dates" in f:
                dates = np.array(f["dates"]).flatten()  # Convert to numpy array
                date_unit = f["dates"].attrs.get('UNIT', None)
                
                # Handle different date formats
                if date_unit == 'D':  # MATLAB serial date (days)
                    print(f"Converting MATLAB serial date to datetime for file: {file}")
                    t_full_A.extend(pd.to_datetime(dates - 719529, unit="D", origin="unix"))
                elif date_unit is None:  # No unit found
                    print(f"Assuming dates are already in a compatible format for file: {file}")
                    t_full_A.extend(dates)  # Assuming they are already datetime or timestamps
                else:
                    print(f"Unexpected date unit '{date_unit}' in {file}. Skipping conversion.")

            # Load IDS (Ice Draft Statistics)
            if "IDS" in f:
                IDS_sub = np.array(f["IDS"]).T  # Transpose since MATLAB stores as column-major
                IDS_full_A.append(IDS_sub[:, :6])  # Extract first 6 columns

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Convert to NumPy arrays
IDS_full_A = np.vstack(IDS_full_A) if IDS_full_A else np.array([])  # Stack into one array
t_full_A = np.array(t_full_A)

# Print first few entries
print("First 5 IDS rows:", IDS_full_A[:5])
print("First 5 Dates:", t_full_A)




