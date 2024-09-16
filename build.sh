#!/bin/bash

rm -rf build/ dist/ *.egg-info
rm -rf extension_cpp/_C.*.so
python setup.py build_ext --inplace