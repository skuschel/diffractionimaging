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
]


symbols = ["H", "He",\
           "Li", "Be", "B", "C", "N", "O", "F", "Ne",\
           "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",\
           "K", "Ca",\
           "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",\
           "Ga", "Ge", "As", "Se", "Br", "Kr",\
           "Rb", "Sr",\
           "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",\
           "In", "Sn", "Sb", "Te", "I", "Xe",\
           "Cs", "Ba",\
           "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",\
           "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",\
           "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",\
           "Tl", "Pb", "Bi", "Po", "At", "Rn",\
           "Fr", "Ra",\
           "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",\
           "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",\
           "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",\
           "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]


atomic_mass_dict = {'H' : 1.008, 'HE' : 4.003, 'LI' : 6.941, 'BE' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'NE' : 20.180, 'NA' : 22.990, 'MG' : 24.305,\
                 'AL' : 26.982, 'SI' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'CL' : 35.453, 'AR' : 39.948, 'K' : 39.098, 'CA' : 40.078,\
                 'SC' : 44.956, 'TI' : 47.867, 'V' : 50.942, 'CR' : 51.996,\
                 'MN' : 54.938, 'FE' : 55.845, 'CO' : 58.933, 'NI' : 58.693,\
                 'CU' : 63.546, 'ZN' : 65.38, 'GA' : 69.723, 'GE' : 72.631,\
                 'AS' : 74.922, 'SE' : 78.971, 'BR' : 79.904, 'KR' : 84.798,\
                 'RB' : 84.468, 'SR' : 87.62, 'Y' : 88.906, 'ZR' : 91.224,\
                 'NB' : 92.906, 'MO' : 95.95, 'TC' : 98.907, 'RU' : 101.07,\
                 'RH' : 102.906, 'PD' : 106.42, 'AG' : 107.868, 'CD' : 112.414,\
                 'IN' : 114.818, 'SN' : 118.711, 'SB' : 121.760, 'TE' : 126.7,\
                 'I' : 126.904, 'XE' : 131.294, 'CS' : 132.905, 'BA' : 137.328,\
                 'LA' : 138.905, 'CE' : 140.116, 'PR' : 140.908, 'ND' : 144.243,\
                 'PM' : 144.913, 'SM' : 150.36, 'EU' : 151.964, 'GD' : 157.25,\
                 'TB' : 158.925, 'DY': 162.500, 'HO' : 164.930, 'ER' : 167.259,\
                 'TM' : 168.934, 'YB' : 173.055, 'LU' : 174.967, 'HF' : 178.49,\
                 'TA' : 180.948, 'W' : 183.84, 'RE' : 186.207, 'OS' : 190.23,\
                 'IR' : 192.217, 'PT' : 195.085, 'AU' : 196.967, 'HG' : 200.592,\
                 'TL' : 204.383, 'PB' : 207.2, 'BI' : 208.980, 'PO' : 208.982,\
                 'AT' : 209.987, 'RN' : 222.081, 'FR' : 223.020, 'RA' : 226.025,\
                 'AC' : 227.028, 'TH' : 232.038, 'PA' : 231.036, 'U' : 238.029,\
                 'NP' : 237, 'PU' : 244, 'AM' : 243, 'CM' : 247, 'BK' : 247,\
                 'CT' : 251, 'ES' : 252, 'FM' : 257, 'MD' : 258, 'NO' : 259,\
                 'LR' : 262, 'RF' : 261, 'DB' : 262, 'SG' : 266, 'BH' : 264,\
                 'HS' : 269, 'MT' : 268, 'DS' : 271, 'RG' : 272, 'CN' : 285,\
                 'NH' : 284, 'FL' : 289, 'MC' : 288, 'LV' : 292, 'TS' : 294,\
                 'OG' : 294}


triple_point_density_dict = {'ne' : 1444, 'ar' : 1636, 'kr' : 2900, 'xe' : 3500}


def atomic_form_factor_henke(element):
    """
    atomic form factor from henke in complex form.

    returns:
      eV, f
    """
    element = element.lower()
    return _download_henke(element)


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
      eV, f
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
    return symbols.index(element) + 1


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
    return atomic_mass_dict[element.upper()]


def atomic_mass_kg(element):
    '''
    returns the atomic mass in kg for a given element symbol
    '''
    return atomic_mass_amu(element) * const['atomic mass constant']


def atomic_mass(element):
    '''
    returns the atomic mass in kg for a given element symbol
    '''
    return atomic_mass_kg(element)
    
    