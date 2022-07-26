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
This module containes correlations routines for incohrent diffraction imaging.
'''

import numpy as np
import scipy.signal as ss
import numpy.ma as ma

from collections import OrderedDict
import functools


class _LRUCache:

    def __init__(self, maxlen=1):
        '''
        simple LRUCache.
        Use this, so the key can be given explicitly.
        '''
        self.cache = OrderedDict()
        self.maxlen = maxlen

    def __getitem__(self, key):
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, val):
        self.cache[key] = val
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxlen:
            self.cache.popitem(last=False)

    def __contains__(self, key):
        return key in self.cache


def lru_cache_bool(maxlen=1):
    '''
    decorator to add an lru_cache for numpy arrays with boolean elements only.
    '''
    def decorator(f):
        cache = _LRUCache(maxlen=maxlen)

        @functools.wraps(f)
        def ret(img):
            if not img.dtype == np.bool:
                # print('no cache')
                # kein Caching
                return f(img)
            h = hash(img.tobytes())
            if h not in cache:
                cache[h] = f(img)
            return cache[h]
        return ret
    return decorator


def correct_mask(f):
    '''
    decorator to apply a mask correction to a given function.

    The decorator will return `f(img)/f(~mask)` of the decorated function.
    '''
    @functools.wraps(f)
    def ret(img):
        # preseves existing mask and np.nan and np.inf also become invalid.
        img = ma.masked_invalid(img)
        res = f(img.filled(0))  # replace invalid values with 0
        norm = f(~img.mask)
        return res / norm
    predoc = '''
    returns the following function with an applied mask correction (i.e. `f(img)/f(~mask)`):

    '''
    ret.__doc__ = ''.join([predoc, ret.__doc__])
    return ret


def second_order_correlation_direct(img):
    r'''
    calculates the second order correlation function $g^{(2)}$ of the image img.
    This is effectively the autocorrelation with the correct normalization.

    $g^{(2)}(x', y') = \frac{\left< I(x, y) \cdot I(x+x', y+y')
    \right>_{x, y}}{\left< \left| I(x, y) \right| \right>_{x, y}^2}$

    with `I = img`.

    Note:
      * The output is biased towards 0 towards the edged of the image,
        but correct in the central part of the image
      * The integral is evaluated as written here.
        No mask are taken into account and np.nan or np.inf are propagated
        according to any other numpy function.
        You are probably looking for the function `second_order_correlation`.
    '''
    img = np.asarray(img, dtype=float)
    s = np.sum(np.abs(img))
    # np.product(img.shape) is an apporximation to normalize by
    # the number of pixels within this correlation. However, this overestimates
    # the number of pixels towards the edge of the image.
    ret = ss.fftconvolve(img, img[::-1, ::-1], mode='same') / s**2 * np.product(img.shape)
    return ret


second_order_correlation = correct_mask(lru_cache_bool(maxlen=1)(second_order_correlation_direct))
