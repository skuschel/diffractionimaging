#!/bin/bash

set -eux

python setup.py install --user

python -m pytest test

# test codestyle last
python -m pycodestyle diffractionimaging --statistics --count --show-source --ignore=W391,E123,E226,E24,W504 --max-line-length=99
