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
    "atomicformfactor_henke",
    "atomicformfactor_nist",
    "ev2lambda",
    "sigmascatt",
    "sigmaabsorb",
    "atomic_number2element",
    "element2atomic_number",
]


symbols = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
           "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
           "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
           "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
           "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
           "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
           "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
           "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
           "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
           "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
           "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts",
           "Og"]


def atomic_form_factor_henke(element):
    """
    atomic form factor from henke in complex form.

    returns:
      eV, f
    """
    element = element.lower()
    return _download_henke(element)


@functools.cache
def _download_henke(element):
    urltmplate = "https://henke.lbl.gov/optical_constants/sf/{element}.nff"
    url = urltmplate.format(element=element)
    r = requests.get(url)
    data = np.genfromtxt(StringIO(r.text))[1:]
    f = data[:, 1] + 1j * data[:, 2]
    eV = data[:, 0]
    return eV, f


@functools.cache
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


def ev2wavelength(ev):
    """
    returns wavelength in meter.
    """
    return 1.2398 / ev / 1e6


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
