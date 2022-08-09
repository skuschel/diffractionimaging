# This file is NOT part of the diffractionimaging package but serves 
# exclusively the purpose of downloading the atomic scattering factor
# database from "https://henke.lbl.gov/optical_constants/sf.tar.gz".

# Requires internet connection! 

# Usage: Run this file using 
#     'python ./download_henke.py'
# from within it's directory

# Copyright (C) 2022 Anatoli Ulmer

from atomicformfactors import atomic_form_factor_henke

# Try to load data for last element 'Zr'. If the data file does not
# exist, it will download the data automatically from the internet.
atomic_form_factor_henke('Zr')

