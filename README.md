# diffractionimaging
Coherent Diffraction Imaging Tools for Python

## Installation

Download the repo
```
git clone https://github.com/skuschel/diffractionimaging
```
and install using
```
cd diffractionimaging
pip install --user -e .
```

Download the recent atomic scattering factor data from
```
https://henke.lbl.gov/optical_constants/sf.tar.gz
```
and unzip the the content into subdirectory
```
./diffractionimaging/henke/
```

Test installation by running a python console somewhere else and execute
```python
import diffractionimaging
```

## Development

Run the tests before committing:
```
./run-tests.sh
```
