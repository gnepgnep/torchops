#!/bin/bash

rm -rf build/ dist/ *.egg-info
rm -rf cuda_ext/_C.*.so
python setup.py build_ext --inplace