#!/usr/bin/env bash

PYBIN=/opt/python/cp38-cp38/bin
sudo ${PYBIN}/pip install --upgrade -r ./dev-requirements.txt
${PYBIN}/python setup.py sdist
