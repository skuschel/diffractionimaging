import functools
import requests
from io import StringIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np


__all__ = ['atomicformfactor_henke', 'atomicformfactor_nist', 'ev2lambda',
           'sigmascatt', 'sigmaabsorb']


def atomicformfactor_henke(element):
    '''
    atomic form factor from henke in complex form.

    returns:
      eV, f
    '''
    element = element.lower()
    return _download_henke(element)


@functools.cache
def _download_henke(element):
    urltmplate = 'https://henke.lbl.gov/optical_constants/sf/{element}.nff'
    url = urltmplate.format(element=element)
    r = requests.get(url)
    data = np.genfromtxt(StringIO(r.text))[1:]
    f = data[:, 1] + 1j*data[:, 2]
    eV = data[:, 0]
    return eV, f


@functools.cache
def atomicformfactor_nist(Z):
    '''
    atomic form factor from nist in complex form.

    returns:
      eV, f
    '''
    # baseurl = 'https://physics.nist.gov/cgi-bin'
    urltmplate = 'https://physics.nist.gov/cgi-bin/ffast/ffast.pl?gtype=4&Z={Z}'
    url = urltmplate.format(Z=Z)
    session = requests.Session()
    r = session.get(url).content
    # parse page
    soup = BeautifulSoup(r)
    tabledata = soup.select('body > pre')[0].text.splitlines()[3:]
    tabledata = '\n'.join(tabledata)
    data = np.genfromtxt(StringIO(tabledata))
    eV = data[:, 0] * 1e3
    f = data[:, 1] + 1j*data[:, 2]
    return eV, f


def ev2lambda(ev):
    '''
    returns lamdba in meter.
    '''
    return 1.2398/ev/1e6


def sigmascatt(f):
    '''
    converts the complext atomic form factor f to the real valued
    scattering cross section in m**2.
    '''
    return 6.65e-29 * np.abs(f)**2


def sigmaabsorb(f, lamb=1):
    '''
    lamb is the wavelength lambda in meter.
    '''
    return -2*2.8e-15 * lamb * np.imag(f)
