import os
from appdirs import user_data_dir
import urllib.request
import tarfile


url = "https://henke.lbl.gov/optical_constants/sf.tar.gz"
data_dir = user_data_dir('diffractionimaging', 'KuschelUlmer')
extract_dir = os.path.join(data_dir, 'henke')
print("Destination Folder:")
print(extract_dir)
os.makedirs(extract_dir, exist_ok=True)

print("Starting Download ...")
tar_path, _ = urllib.request.urlretrieve(url)
print("Extracting Files ...")
file = tarfile.open(tar_path)
file.extractall(extract_dir)
file.close()
print("Done!")

