# BZX - Boozer to GKV Coordinate Transformations
---

**BZX** is a Python package that transforms **BOOZ_XFORM** Boozer coordinate metrics into **GKV** field-aligned coordinates, enabling gyrokinetic analysis of **VMEC** equilibria.


## Installation

To install from PyPI:
```bash
pip install bzx
```
Or install the latest development version from GitHub:
```bash
pip install git+https://github.com/GKV-developers/bzx.git
```

## Usage

#### **(i) Basic usage: Convert to GKV input file using BZX**
Prepare **VMEC** equilibrium output (e.g. `wout.nc`) and its **BOOZ_XFORM** output (e.g. `boozmn.nc`).

```python
from bzx import bzx

# Define transformation parameters
Ntheta_gkv = 1   # N_theta value in GKV
nrho = 11        # Radial grid points
ntht = 64        # Poloidal grid points
nzeta = 0        # Toroidal grid points (nzeta=0 corresponds to output GKV field-aligned coordinates)
alpha_fix = 0.0  # Field-line label: alpha = zeta - q*theta (not used for nzeta > 0)

# Run BZX transformation
bzx(Ntheta_gkv, nrho, ntht, nzeta, alpha_fix, 
    fname_boozmn="boozmn.nc", fname_wout="wout.nc", 
    output_file="./metric_boozer.bin.dat")
```
- Reads **VMEC output file (`wout.nc`)** and **BOOZ_XFORM output file (`boozmn.nc`)**. NetCDF format is recommended, but Binary format is also acceptable.
- Converts them into field-aligned data and saves the result as **GKV input binary file (`metric_boozer.bin.dat`)**.


#### **(ii) Example of workflow: From VMEC to GKV via BOOZ_XFORM and BZX**
The script [`examples/run_vmecpp_boozxform_bzx.py`](examples/run_vmecpp_boozxform_bzx.py) demonstrats how to use **BZX**, starting from generating an equilibrium.

- Step 1: Compute a MHD equilibrium using **VMEC++**
    - About **VMEC++** : 
        - [Proxima Fusion - VMEC++](https://www.proximafusion.com/press-news/introducing-vmecpp-open-source-software-for-fusion-research)
        - [GitHub - vmecpp](https://github.com/proximafusion/vmecpp)

- Step 2: Convert the **VMEC** output to Boozer coordinates using **BOOZ_XFORM**
    - About **BOOZ_XFORM** :
        - [BOOZ_XFORM Documentation](https://hiddensymmetries.github.io/booz_xform/)
        - [GitHub - BOOZ_XFORM](https://github.com/hiddenSymmetries/booz_xform)
- Step 3: Convert the **BOOZ_XFORM** output to GKV coordinates using **BZX**
    - This project.


#### **(iii) As a command line tool**
After installation, you can use **BZX** directly as a CLI tool in a terminal, e.g.,
```sh
python -m bzx --Ntheta_gkv 1 --nrho 11 --ntht 64 \
              --nzeta 0 --alpha_fix 0.0 \
              --fname_boozmn "boozmn.nc" --fname_wout "wout.nc" \
              --output_file "./metric_boozer.bin.dat"
```
All arguments of `bzx` function is specified by keyword arguments. See also help.
```sh
python -m bzx --help
```

## Dependencies

BZX requires the following Python packages:
- `numpy`, `scipy`, `xarray`
- (Optional) `matplotlib` for visualization in `examples/`
- (Optional) `booz_xform` for `examples/run_vmecpp_boozxform_bzx.py`
- (Optional) `vmecpp` for `examples/run_vmecpp_boozxform_bzx.py`

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## Author

Developed by Shinya Maeyama (maeyama.shinya@nifs.ac.jp)
