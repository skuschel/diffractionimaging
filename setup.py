#!/usr/bin/env python

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


from setuptools import setup, find_packages
import numpy
import os

import versioneer

setup(name='diffractionimaging',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      include_package_data = True,
      author='Stephan Kuschel',
      author_email='stephan.kuschel@gmail.de',
      description='Diffraction Imaging Tools for Python',
      url='https://github.com/skuschel/diffractionimaging',
      include_dirs = [numpy.get_include()],
      license='GPLv3+',
      python_requires='>=3.6',
      setup_requires=['numpy>=1.8'],
      install_requires=['numpy>=1.8',
                        'scipy',
                        'urllib3',
                        'requests',
                        'beautifulsoup4'],
      extras_require = {},
      keywords = ['diffraction imaging', 'cluster size', 'cdi', 'physics', 'optics',
                  'xray'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Operating System :: Unix',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)']
      )
