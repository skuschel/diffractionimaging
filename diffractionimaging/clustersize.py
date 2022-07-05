
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

# Copyright (C) 2022 Stephan Kuschel
'''
Module containing routines to determine the ball size from its scattering pattern.
We approximate this by a hard sphere.

Includes routines for finding the center or radial profiles.
'''


import numpy as np
import numpy.ma as ma
import scipy.optimize as so


def radial_profile(data, center=None, anglerange=None, returnmask=False):
    '''
    Nans are ignored. data can be a masked array.

    Center is assumed to be center if not given.
    If given, is allowed to lay outside.

    returns the radial profile rprof.
    r values are always np.arange(len(rprof))

    Stephan Kuschel, 2022
    '''
    if center is None:
        center = data.shape[0] // 2, data.shape[1] // 2
    y, x = np.indices(data.shape)
    r = np.rint((np.sqrt((x - center[0])**2 + (y - center[1])**2))).astype(int)
    nanmask = ~np.isnan(data)
    if anglerange is not None:
        phi = np.arctan2(y - center[1], x - center[0])
        anglerange = sorted(anglerange)
        nanmask *= phi > anglerange[0]
        nanmask *= phi < anglerange[1]
    rr = np.ravel(r[nanmask])
    nr = np.bincount(rr)
    tbin = np.bincount(rr, data[nanmask].ravel())
    nr = np.ma.array(nr, dtype=int)
    nr[nr == 0] = np.ma.masked
    radialprofile = tbin / nr
    if returnmask:
        return radialprofile, nanmask
    else:
        return radialprofile


def _find_center_iter(bild, center=None, kernelsize=None):
    '''
    does a single iteration such that the we are getting a better estimate for the center.
    "center" is actually the center of symmetry.

    Stephan Kuschel, 2018-2019
    '''
    from scipy import signal
    if center is None:
        center = [s / 2 for s in bild.shape]
    if kernelsize is None:
        kernelrad = [s // 4 for s in bild.shape]  # more than enough
    slices = tuple(slice(int(c - r), int(c + r))
                   for c, r in zip(center, kernelrad))
    kernel = bild[slices]
    conv = signal.fftconvolve(bild, kernel.T, mode='same')
    # print(conv.shape)
    maxidx = np.unravel_index(np.argmax(conv, axis=None), conv.shape)  # :)
    newcenter = [(n + o) / 2 for n, o in zip(maxidx, center)]
    return conv, newcenter


def find_center_auto_data(bild, center=None, n=4):
    '''
    Automatically return the center.

    Stephan Kuschel, 2018
    '''
    bild = np.array(bild)
    bild[bild < 0] = 0
    data = np.log(bild + 1)  # better metrik
    _, idx = _find_center_iter(data, center=center)
    for i in range(n):
        _, idx = _find_center_iter(data, center=idx)
    return idx


def diffraction_sphere(pxl, I0, minpos, c=0):
    '''
    minpos is position of first minimum
    '''
    x = pxl / minpos * 4.493
    return I0 * (9 * (np.sin(x) / x**2 - np.cos(x) / x)**2) / x**2 + c


def diffraction_pattern_extrema_linfit(rprof, c=0.1):
    '''
    takes the radial profile, returns m, b and position of first minimum.
    m and b is the straight line connecting all min and max values.
    '''
    rprof = ma.masked_invalid(rprof)
    rprof = np.log(rprof + c)  # better metric
    rprof = np.convolve(rprof, np.ones(5) / 5, 'same')
    from scipy.signal import argrelmin, argrelmax
    relmin = argrelmin(rprof, order=15)[0]
    relmax = argrelmax(rprof, order=15)[0]

    relmm = np.array(sorted(np.concatenate([relmin, relmax])))
    minidx = np.searchsorted(relmm, relmin)
    maxidx = np.searchsorted(relmm, relmax)
    if len(minidx) < 1 or len(maxidx) < 1:
        return 0, 0
    # nur so lange daten verwenden, wie minima und maxima abwechselnd aufeinander folgen
    last = None
    for i in range(len(relmm)):
        last = i
        if i % 2 == 0 and i not in maxidx:  # erster ist immer ein max
            break
        if i % 2 == 1 and i not in minidx:  # zweiter immer ein min
            break
    if last < 2:
        raise ValueError('too few points to fit.')
    relmm = relmm[:last]  # last is not included
    minidx = minidx[minidx < last]
    maxidx = maxidx[maxidx < last]
    assert len(minidx) + len(maxidx) == len(relmm)
    # sonst erfahren alle minima verschwindendes Gewicht
    w = np.convolve(rprof[relmm], [0.5, 0.5], 'same')
    m, b = np.polyfit(np.array(range(len(relmm))), relmm, 1, w=w)
    return m, b, relmin[0]


def linfit2guess(guess):
    minposs = guess[0] + guess[1], guess[2]
    return np.mean(minposs)


def fit_diffraction_sphere(radprof, I0, minpos):
    '''
    minpos and I0 are the starting values.
    nans will be ignored.
    '''
    rprof = ma.masked_invalid(radprof)
    I0 = 2 * np.max(rprof) if I0 is None else I0
    r = np.arange(len(radprof))
    popt = so.curve_fit(diffraction_sphere,
                        r[~rprof.mask], radprof[~rprof.mask], p0=(I0, minpos))
    return popt


def fit2size(popt, lamb, pixelsize=50e-6, d=425e-3, npixel=100):
    '''
    returns the fitted sphere radius +- fit error in nm.

    popt: output of fit_diffraction_sphere
    npixel: number of pixels used for the fit. used to estimate the error
    lamb: photon wavelength im m
    pixelsize: pixelsize of the detector in m
    d: detector distance in m
    '''
    fit = popt
    size = np.array([1 / np.abs(fit[0][1]), 0.5 *
                    np.sqrt(npixel * fit[1][1, 1]) * fit[0][1]**-2])
    return size * d / pixelsize * 0.5 * lamb * 1.43
