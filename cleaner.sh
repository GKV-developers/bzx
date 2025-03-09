#/bin/sh

find . -name .ipynb_checkpoints | xargs rm -rf
find . -name __pycache__ | xargs rm -rf
rm -rf build/
rm -rf src/bzx.egg-info/
rm -f examples/*.nc
rm -f examples/*.dat

jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace examples/*.ipynb
jupyter nbconvert --to python examples/*.ipynb
