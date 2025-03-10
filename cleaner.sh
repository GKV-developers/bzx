#/bin/sh

find . -name .ipynb_checkpoints | xargs rm -rf
find . -name __pycache__ | xargs rm -rf
rm -rf build/
rm -rf src/bzx.egg-info/
rm -f examples/*.nc
rm -f examples/*.dat
rm -f tests/log_BZX.dat tests/metric_boozer.bin.dat
rm -rf .pytest_cache/

jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace examples/*.ipynb
jupyter nbconvert --to python examples/*.ipynb
