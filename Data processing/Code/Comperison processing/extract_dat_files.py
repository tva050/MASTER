import zipfile
import os
import glob

# Directory containing zip files
zip_dir = r'C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data txt'
extract_dir = r'C:\Users\trym7\OneDrive - UiT Office 365\skole\MASTER\Data processing\Data\BGEP moorings\data txt'

# Ensure the extraction directory exists
os.makedirs(extract_dir, exist_ok=True)

# Find all .zip files in the directory
zip_files = glob.glob(os.path.join(zip_dir, '*.zip'))

# Loop through each zip file
for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # List all files in the zip
        for file_name in zip_ref.namelist():
            # If the file is a .dat file
            if file_name.endswith('.dat'):
                # Extract the .dat file to the extraction directory
                zip_ref.extract(file_name, extract_dir)
                print(f"Extracted: {file_name} from {zip_file}")