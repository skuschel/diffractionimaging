# This file is part of diffractionimaging.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Copyright (C) 2022 Stephan Kuschel, Anatoli Ulmer

import functools
import requests
from io import StringIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
from scipy.constants import physical_constants as const

import os
from appdirs import user_data_dir
import urllib
import zipfile


__all__ = [
    "atomic_form_factor_henke",
    "atomic_form_factor_nist",
    "ev2wavelength",
    "scattering_cross_section",
    "absorption_cross_section",
    "atomic_number2element",
    "element2atomic_number",
    "absorption_length",
    "atomic_mass_amu",
    "atomic_mass_kg",
    "atomic_mass",
    "atomic_form_factor",
    "refractive_index",
    "delta",
    "beta",
]


symbols = ["h", "he",
           "li", "be", "b", "c", "n", "o", "f", "ne",
           "na", "mg", "al", "si", "p", "s", "cl", "ar",
           "k", "ca",
           "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn",
           "ga", "ge", "as", "se", "br", "kr",
           "rb", "sr",
           "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd",
           "in", "sn", "sb", "te", "i", "xe",
           "cs", "ba",
           "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd",
           "tb", "dy", "ho", "er", "tm", "yb", "lu",
           "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg",
           "tl", "pb", "bi", "po", "at", "rn",
           "fr", "ra",
           "ac", "th", "pa", "u", "np", "pu", "am", "cm",
           "bk", "cf", "es", "fm", "md", "no", "lr",
           "rf", "db", "sg", "bh", "hs", "mt", "ds", "rg", "cn",
           "nh", "fl", "mc", "lv", "ts", "og"]


atomic_mass_dict = {'h': 1.008, 'he': 4.003, 'li': 6.941, 'be': 9.012,
                    'b': 10.811, 'c': 12.011, 'n': 14.007, 'o': 15.999,
                    'f': 18.998, 'ne': 20.180, 'na': 22.990, 'mg': 24.305,
                    'al': 26.982, 'si': 28.086, 'p': 30.974, 's': 32.066,
                    'cl': 35.453, 'ar': 39.948, 'k': 39.098, 'ca': 40.078,
                    'sc': 44.956, 'ti': 47.867, 'v': 50.942, 'cr': 51.996,
                    'mn': 54.938, 'fe': 55.845, 'co': 58.933, 'ni': 58.693,
                    'cu': 63.546, 'zn': 65.38, 'ga': 69.723, 'ge': 72.631,
                    'as': 74.922, 'se': 78.971, 'br': 79.904, 'kr': 84.798,
                    'rb': 84.468, 'sr': 87.62, 'y': 88.906, 'zr': 91.224,
                    'nb': 92.906, 'mo': 95.95, 'tc': 98.907, 'ru': 101.07,
                    'rh': 102.906, 'pd': 106.42, 'ag': 107.868, 'cd': 112.414,
                    'in': 114.818, 'sn': 118.711, 'sb': 121.760, 'te': 126.7,
                    'i': 126.904, 'xe': 131.294, 'cs': 132.905, 'ba': 137.328,
                    'la': 138.905, 'ce': 140.116, 'pr': 140.908, 'nd': 144.243,
                    'pm': 144.913, 'sm': 150.36, 'eu': 151.964, 'gd': 157.25,
                    'tb': 158.925, 'dy': 162.500, 'ho': 164.930, 'er': 167.259,
                    'tm': 168.934, 'yb': 173.055, 'lu': 174.967, 'hf': 178.49,
                    'ta': 180.948, 'w': 183.84, 're': 186.207, 'os': 190.23,
                    'ir': 192.217, 'pt': 195.085, 'au': 196.967, 'hg': 200.592,
                    'tl': 204.383, 'pb': 207.2, 'bi': 208.980, 'po': 208.982,
                    'at': 209.987, 'rn': 222.081, 'fr': 223.020, 'ra': 226.025,
                    'ac': 227.028, 'th': 232.038, 'pa': 231.036, 'u': 238.029,
                    'np': 237, 'pu': 244, 'am': 243, 'cm': 247, 'bk': 247,
                    'ct': 251, 'es': 252, 'fm': 257, 'md': 258, 'no': 259,
                    'lr': 262, 'rf': 261, 'db': 262, 'sg': 266, 'bh': 264,
                    'hs': 269, 'mt': 268, 'ds': 271, 'rg': 272, 'cn': 285,
                    'nh': 284, 'fl': 289, 'mc': 288, 'lv': 292, 'ts': 294,
                    'og': 294}


triple_point_density_dict = {'ne': 1444, 'ar': 1636, 'kr': 2900, 'xe': 3500}


def _download_and_unzip(url, extract_dir):
    '''
    'https://stackoverflow.com/questions/6861323/download-and-unzip-file-with-python'
    '''
    print("Starting Download ...")
    zip_path, _ = urllib.request.urlretrieve(url)
    print("Extracting Files ...")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(extract_dir)
    print("Done!")


def _data_dir():
    return user_data_dir(__name__, __author__)


def _download_henke_db():
    url = "https://henke.lbl.gov/optical_constants/sf.tar.gz"
    extract_dir = os.path.join(_data_dir(), 'henke')
    os.makedirs(extract_dir, exist_ok=True)
    download_and_unzip(url, extract_dir)


def atomic_form_factor_henke(element):
    """
    energy array and atomic form factor array from henke in complex form.

    returns:
      eV, f0
    """
    # urltmplate = "https://henke.lbl.gov/optical_constants/sf/{element}.nff"
    # url = urltmplate.format(element=element)
    filename = os.path.join(_data_dir(), "henke", "{element}.nff".format(element=element))
    if not os.path.exists(filename):
        _download_henke_db()

    energy_ev, f1, f2 = np.loadtxt(filename, skiprows=1, unpack=True)
    f1[f1 == -9.99900e+03] = np.nan
    f2[f2 == -9.99900e+03] = np.nan
    f0 = f1 + 1j * f2
    return energy_ev, f0


def _load_local_henke(element):
    '''
    loads atomic form factor for a given element symbol from locally stored data

    returns:
      eV, f0
    '''
    import os
    energy_ev, f1, f2 = np.loadtxt(os.path.dirname(__file__) + '/henke/' + element.lower() +
                                   '.nff', skiprows=1, unpack=True)
    f1[f1 == -9.99900e+03] = np.nan
    f2[f2 == -9.99900e+03] = np.nan
    f0 = f1 + 1j * f2
    return energy_ev, f0


@functools.lru_cache(maxsize=None)
def _download_henke(element):
    urltmplate = "https://henke.lbl.gov/optical_constants/sf/{element}.nff"
    url = urltmplate.format(element=element)
    r = requests.get(url)
    data = np.genfromtxt(StringIO(r.text))[1:]
    f = data[:, 1] + 1j * data[:, 2]
    eV = data[:, 0]
    return eV, f


@functools.lru_cache(maxsize=None)
def atomic_form_factor_nist(element):
    """
    atomic form factor from nist in complex form.

    returns:
      eV, f0
    """
    Z = element2atomic_number(element)
    # baseurl = 'https://physics.nist.gov/cgi-bin'
    urltmplate = "https://physics.nist.gov/cgi-bin/ffast/ffast.pl?gtype=4&Z={Z}"
    url = urltmplate.format(Z=Z)
    session = requests.Session()
    r = session.get(url).content
    # parse page
    soup = BeautifulSoup(r)
    tabledata = soup.select("body > pre")[0].text.splitlines()[3:]
    tabledata = "\n".join(tabledata)
    data = np.genfromtxt(StringIO(tabledata))
    eV = data[:, 0] * 1e3
    f = data[:, 1] + 1j * data[:, 2]
    return eV, f


def ev2wavelength(energy_ev):
    """
    returns wavelength in meter.
    """
    energy = energy_ev * const['electron volt-joule relationship'][0]
    wavelength = const['Planck constant'][0] * const['speed of light in vacuum'][0] / energy  # m
    return wavelength


def scattering_cross_section(f0):
    r"""
    converts the complex atomic form factor f0 to the real valued
    scattering cross section in m**2.

    $ \sigma_\mathrm{scatt} = \frac{8 \pi}{3} r_e^2 \abs{f^0}^2 $
    with the classical electron radius
    $ r_e = 2.81794 \times 10^{-15} m $
    """
    return 8 / 3 * np.pi * const["classical electron radius"][0] ** 2 * np.abs(f0) ** 2


def absorption_cross_section(f0, wavelength=1):
    r"""
    converts the complex atomic form factor f0 and the wavelength in
    meter to the real valued atomic photoabsorption cross section in m**2.

    \( \sigma_\mathrm{scatt} = 2 r_e f^0_2 \)
    with the classical electron radius
    \( r_e = 2.81794 \times 10^{-15} m\)
    """
    return -2 * const["classical electron radius"][0] * wavelength * np.imag(f0)


def atomic_number2element(atomic_number):
    """
    returns element symbol for a given atomic number
    """
    return symbols[atomic_number - 1]


def element2atomic_number(element):
    """
    returns atomic number for a given element symbol
    """
    return symbols.index(element.lower()) + 1


def absorption_length(n, wavelength):
    r"""
    returns absorption length for a given complex-valued refractive index n
    and a given wavelength in m

    \( \ell_\textrm{abs} = \frac{\lambda}{4\pi\beta} = \frac{1}{2n_a r_e \lambda f_2^0(\omega)}. \)
    """
    return wavelength / 4 / np.pi / np.imag(n)


def atomic_mass_amu(element):
    '''
    returns the atomic mass in atomic mass units for a given element symbol
    '''
    return atomic_mass_dict[element.lower()]


def atomic_mass_kg(element):
    '''
    returns the atomic mass in kg for a given element symbol
    '''
    return atomic_mass_amu(element.lower()) * const['atomic mass constant']


def atomic_mass(element):
    '''
    returns the atomic mass in kg for a given element symbol
    '''
    return atomic_mass_kg(elemen.lower()t)


def atomic_form_factor(element, energy_ev):
    '''
    returns the atomic form factor f0 for a queried element at a queried energy in eV.
    f0 is interpolated from tabulated databases

    input:
        element - element symbol consisting of one or two letters, e.g., 'H', 'Xe', 'Mg'
        energy_ev - energy in electronvolts in the range 10eV < energyInElectronVolts < 30000eV

    returns:
        f0
    '''
    henke_energy_ev, f1, f2 = _load_local_henke(element)
    f1_interp = np.interp(energy_ev, henke_energy_ev, f1)
    f2_interp = np.interp(energy_ev, henke_energy_ev, f2)
    f0_interp = f1_interp + 1j * f2_interp  # atomic scattering factor
    return f0_interp


def refractive_index(element, atom_density, energy_ev):
    '''
    returns the refractive index for a given element, atom density and photon energy in eV

    input:
        element - element symbol consisting of one or two letters, e.g., 'H', 'Xe', 'Mg'
        atom_density - atom number density in 1/m^3
        energy_ev - energy in electronvolts in the range 10eV < energyInElectronVolts < 30000eV

    returns:
        n - refractive index
    '''
    f0 = atomic_form_factor(element, energy_ev)
    wavelength = ev2wavelength(energy_ev)
    n = 1 - atom_density * const['classical electron radius'][0] * wavelength**2/2/np.pi * f0
    return n


def delta(n):
    '''
    n = 1 - delta + 1j*beta
        n - refractive index
        delta - phase coefficient
        beta - absorption coefficient
    '''
    return (1.0-np.real(n))


def beta(n):
    '''
    n = 1 - delta + 1j*beta
        n - refractive index
        delta - phase coefficient
        beta - absorption coefficient
    '''
    return -np.imag(n)

