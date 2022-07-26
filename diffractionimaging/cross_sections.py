import numpy as np
from scipy.constants import physical_constants as const


def _cluster_data(element, mass_density=None):
    # element = 'Ne' | 'Ar' | 'Kr' | 'Xe'
    # mass_density in kg/m^3, if mass_density is None then solid densities at triple point are used

    if element.lower() == 'ne':
        data = {'element': 'ne', 'atomic_number': 10, 'atomic_mass_amu': 20.189,
                'atomic_mass': 20.189*1.6605e-27, 'triple_point_density': 1444}
    elif element.lower() == 'ar':
        data = {'element': 'ar', 'atomic_number': 18, 'atomic_mass_amu': 39.95,
                'atomic_mass': 39.95*1.6605e-27, 'triple_point_density': 1636}
    elif element.lower() == 'kr':
        data = {'element': 'kr', 'atomic_number': 36, 'atomic_mass_amu': 83.80,
                'atomic_mass': 83.80*1.6605e-27, 'triple_point_density': 2900}
    elif element.lower() == 'xe':
        data = {'element': 'xe', 'atomic_number': 54, 'atomic_mass_amu': 131.29,
                'atomic_mass': 131.29*1.6605e-27, 'triple_point_density': 3500}
    else:
        print('Error: element not in database!')

    if mass_density is None:
        data['mass_density'] = data['triple_point_density']  # kg/m^3
    else:
        data['mass_density'] = mass_density

    data['atom_density'] = data['mass_density']/data['atomic_mass']  # 1/m^3
    data['electron_density'] = data['atom_density']*data['atomic_number']  # 1/m^3
    return data


def create_cluster(element, radius, mass_density=None):
    cluster = _cluster_data(element, mass_density)
    cluster['radius'] = radius
    cluster['volume'] = 4/3*np.pi*radius**3
    cluster['n_atoms'] = cluster['atom_density'] * cluster['volume']
    cluster['geometric_cross_section'] = np.pi * cluster['radius']**2
    return cluster


def get_refraction_data(element, atom_density, energy_ev):
    wavelength = _wavelength(energy_ev)
    f0 = _f0(element, energy_ev)
    n = _n(wavelength, atom_density, f0)
    atom_cross_section = _atom_cross_section(f0)
    return wavelength, f0, n, atom_cross_section


def _wavelength(energy_ev):
    energy = energy_ev * const['electron volt-joule relationship'][0]
    wavelength = const['Planck constant'][0] * const['speed of light in vacuum'][0] / energy  # m
    return wavelength


def load_henke_f0(element):
    import os
    energy_ev, f1, f2 = np.loadtxt(os.path.dirname(__file__) + '/henke/' + element.lower() +
                                   '.nff', skiprows=1, unpack=True)
    f1[f1 == -9.99900e+03] = np.nan
    f2[f2 == -9.99900e+03] = np.nan
    return energy_ev, f1, f2


def _f0(element, energy_ev):
    # element = 'Ne' | 'Ar' | 'Kr' | 'Xe'
    # 10eV < energyInElectronVolts < 30000eV

    henke_energy_ev, f1, f2 = load_henke_f0(element)
    f1_interp = np.interp(energy_ev, henke_energy_ev, f1)
    f2_interp = np.interp(energy_ev, henke_energy_ev, f2)
    f0_interp = f1_interp + 1j * f2_interp  # atomic scattering factor
    return f0_interp


def _n(wavelength, atom_density, f0):
    return 1 - atom_density * const['classical electron radius'][0] * wavelength**2/2/np.pi * f0


def _atom_cross_section(f0):
    return 8/3 * np.pi * const['classical electron radius'][0]**2 * np.abs(f0)**2


def delta(n):
    return (1.0-np.real(n))


def beta(n):
    return -np.imag(n)


def guinier_cross_section(cluster, wavelength, f0):
    element = cluster['element']
    atom_density = cluster['atom_density']
    cross_sections = np.zeros((np.size(cluster['n_atoms']), np.size(wavelength)))
    for idx_r, radius in enumerate(cluster['radius']):
        atom_cross_section = _atom_cross_section(f0)
        n_atoms = cluster['n_atoms'][idx_r]
        cross_sections[idx_r, :] = 9/16/np.pi*wavelength**2 * radius * atom_density * n_atoms \
            * atom_cross_section
    return cross_sections

