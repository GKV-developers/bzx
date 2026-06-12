#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pytest
import os
import io
from pathlib import Path
from bzx import bzx, read_GKV_metric_file
from bzx.bzx import (Metric, Spline, fill_missing_phi_b_from_wout,
                     input_from_boozmn, read_text)

TEST_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = TEST_DIR / "reference_data"
BOOZMN_FILE = REFERENCE_DIR / "boozmn_tests.nc"
WOUT_FILE = REFERENCE_DIR / "wout_tests.nc"
REFERENCE_METRIC_FILE = REFERENCE_DIR / "metric_boozer.bin.dat"

NTHETA_GKV = 1
NRHO = 5
NTHT = 12
NZETA = 4
ALPHA_FIX = 0.1


def _spline_all(xp, yp, x, *, warn_out_of_bounds=True):
    spline = Spline(len(xp), warn_out_of_bounds=warn_out_of_bounds)
    y = np.zeros_like(x, dtype=float)
    dydx = np.zeros_like(x, dtype=float)
    spline.cubic_spline_pre(xp, yp, len(xp))
    spline.cubic_spline_all(len(x), x, y, dydx)
    return y, dydx


def _rho_hf_from_jlist(boozmn):
    """jlist-based half-mesh rho grid with both edges (nsb1 points)."""
    rho_hf = np.zeros(boozmn.ns_b + 1)
    rho_hf[1:boozmn.ns_b] = np.sqrt(
        (boozmn.jlist.astype(float) - 1.5) / (boozmn.ns_b - 1.0))
    rho_hf[boozmn.ns_b] = 1.0
    return rho_hf


def _extrapolated_boozmn_and_metric():
    boozmn = input_from_boozmn(str(BOOZMN_FILE))
    B0_p, Aminor_p, Rmajor_p, volume_p = read_text(str(WOUT_FILE), "")
    metric = Metric(io.StringIO())
    metric.extrapolation_to_magnetic(boozmn)
    metric.normalization(boozmn, B0_p, Rmajor_p)
    return boozmn, metric


@pytest.fixture(scope="module")
def generated_metric_file(tmp_path_factory):
    """Runs BZX to generate a test output file"""
    work_dir = tmp_path_factory.mktemp("bzx")
    output_file = work_dir / "metric_boozer.bin.dat"

    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        bzx(NTHETA_GKV, NRHO, NTHT, NZETA, ALPHA_FIX,
            str(BOOZMN_FILE), str(WOUT_FILE), str(output_file))
    finally:
        os.chdir(cwd)

    # Ensure the output file exists
    assert os.path.exists(output_file), f"Test output file {output_file} was not created!"

    return output_file


def test_gkv_metric_comparison(generated_metric_file):
    """Compares reference data and generated test data using tuples"""
    # Read reference and test data as tuples
    reference_data = read_GKV_metric_file(str(REFERENCE_METRIC_FILE))
    test_data = read_GKV_metric_file(str(generated_metric_file))

    # Compare each element in the tuples
    keys = ("nfp_b", "nss", "ntht", "nzeta", "mnboz_b", "mboz_b", "nboz_b", "Rax", "Bax", "aa", "volume_p", "asym_flg", "alpha_fix",
            "rho", "theta", "zeta", "qq", "shat", "epst", "bb", "rootg_boz", "rootg_boz0", "ggup_boz",
            "dbb_drho", "dbb_dtht", "dbb_dzeta",
            "rr", "zz", "ph", "bbozc", "ixn_b", "ixm_b",
            "bbozs")
    for i, (ref_val, test_val) in enumerate(zip(reference_data, test_data)):
        if isinstance(ref_val, np.ndarray):  # Compare array values
            if np.isnan(ref_val).any() or np.isinf(ref_val).any():
                print(f"Warning: Reference data at index {i} ({keys[i]}) contains NaN or Inf")
            if np.isnan(test_val).any() or np.isinf(test_val).any():
                print(f"Warning: Test data at index {i} ({keys[i]}) contains NaN or Inf")

            assert np.allclose(ref_val, test_val, atol=1e-8, equal_nan=True), f"Mismatch at index {i} ({keys[i]}): Arrays differ!"
        else:  # Compare scalar values
            assert ref_val == test_val, f"Mismatch at index {i} ({keys[i]}): {ref_val} != {test_val}"


def test_q_profile_uses_half_mesh(generated_metric_file):
    data = read_GKV_metric_file(str(generated_metric_file))
    rho = data[13]
    qq = data[16]

    boozmn, metric = _extrapolated_boozmn_and_metric()
    rho_hf = _rho_hf_from_jlist(boozmn)
    expected_qq, _ = _spline_all(rho_hf, 1.0 / boozmn.iota_b_nu, rho)

    assert rho_hf[0] == 0.0
    assert rho_hf[-1] == 1.0
    assert len(rho_hf) == boozmn.ns_b + 1
    assert np.allclose(qq, expected_qq, atol=1e-10, rtol=1e-10)


def test_fourier_coefficients_use_half_mesh(generated_metric_file):
    data = read_GKV_metric_file(str(generated_metric_file))
    rho = data[13]
    bbozc = data[29]

    boozmn, metric = _extrapolated_boozmn_and_metric()
    rho_hf = _rho_hf_from_jlist(boozmn)
    mode_indices = [0, int(np.flatnonzero(boozmn.ixm_b != 0)[0])]

    for imn in mode_indices:
        expected_bbozc, _ = _spline_all(
            rho_hf, metric.bbozc_nu[imn, :], rho)
        assert np.allclose(bbozc[imn, :], expected_bbozc, atol=1e-10, rtol=1e-10)


def test_fourier_coefficients_extrapolated_to_both_edges():
    boozmn, metric = _extrapolated_boozmn_and_metric()

    assert metric.bbozc_nu.shape == (boozmn.mnboz_b, boozmn.ns_b + 1)
    # axis: m=0 modes are linearly extrapolated, m!=0 modes vanish
    m0 = (boozmn.ixm_b == 0)
    assert np.allclose(metric.bbozc_nu[m0, 0],
                       1.5 * metric.bbozc_nu[m0, 1] -
                       0.5 * metric.bbozc_nu[m0, 2])
    assert np.all(metric.bbozc_nu[~m0, 0] == 0.0)
    # LCFS: all modes are linearly extrapolated
    assert np.allclose(metric.bbozc_nu[:, -1],
                       1.5 * metric.bbozc_nu[:, -2] -
                       0.5 * metric.bbozc_nu[:, -3])


def test_axis_values_of_current_profiles():
    boozmn, metric = _extrapolated_boozmn_and_metric()

    # on axis, bvco (=cug) is finite while buco (=cui) vanishes
    assert boozmn.buco_b_nu[0] == 0.0
    assert np.isclose(boozmn.bvco_b_nu[0],
                      1.5 * boozmn.bvco_b_nu[1] - 0.5 * boozmn.bvco_b_nu[2])
    assert boozmn.bvco_b_nu[0] != 0.0
    # 1D profiles are extended to the LCFS (nsb1 points)
    assert boozmn.iota_b_nu.size == boozmn.ns_b + 1
    assert boozmn.bvco_b_nu.size == boozmn.ns_b + 1
    assert boozmn.buco_b_nu.size == boozmn.ns_b + 1


def test_negative_iota_flips_poloidal_angle(tmp_path):
    """An iota<0 equilibrium is converted to iota>0 by flipping the
    poloidal angle direction (iota, buco, ixm sign flip), so that the
    direction of increasing poloidal angle aligns with the field."""
    import xarray as xr

    flipped_file = tmp_path / "boozmn_negative_iota.nc"
    with xr.load_dataset(str(BOOZMN_FILE)) as ds:
        ds["iota_b"] = -ds["iota_b"]
        ds["buco_b"] = -ds["buco_b"]
        ds.to_netcdf(str(flipped_file))

    boozmn_ref = input_from_boozmn(str(BOOZMN_FILE))
    boozmn = input_from_boozmn(str(flipped_file))

    assert np.all(boozmn.iota_b_nu[1:] > 0.0)
    assert np.allclose(boozmn.iota_b_nu, boozmn_ref.iota_b_nu)
    assert np.allclose(boozmn.buco_b_nu, boozmn_ref.buco_b_nu)
    assert np.array_equal(boozmn.ixm_b, -boozmn_ref.ixm_b)
    assert np.array_equal(boozmn.ixn_b, boozmn_ref.ixn_b)


def test_mixed_sign_iota_is_rejected(tmp_path):
    import xarray as xr

    mixed_file = tmp_path / "boozmn_mixed_sign_iota.nc"
    with xr.load_dataset(str(BOOZMN_FILE)) as ds:
        iota = ds["iota_b"].copy()
        iota_values = iota.to_numpy()
        iota_values[1] = -abs(iota_values[1])
        ds["iota_b"] = (iota.dims, iota_values)
        ds.to_netcdf(str(mixed_file))

    with pytest.raises(ValueError, match="changes sign"):
        input_from_boozmn(str(mixed_file))


def test_phi_interpolation_uses_half_mesh_grid():
    boozmn, metric = _extrapolated_boozmn_and_metric()
    metric.q_profile(NTHETA_GKV, NRHO, NTHT, NZETA, boozmn)
    metric.interpolation_to_uniform(NRHO, boozmn)

    rho_hf = _rho_hf_from_jlist(boozmn)
    expected_phi, expected_dphidrho = _spline_all(
        rho_hf, boozmn.phi_hf_nu, metric.rho)

    assert boozmn.phi_hf_nu[0] == 0.0
    assert np.isclose(boozmn.phi_hf_nu[-1],
                      boozmn.phi_b_nu[boozmn.ns_b - 1])
    assert np.allclose(metric.rho_nu, rho_hf, atol=1e-15, rtol=1e-15)
    assert np.allclose(metric.phi_b, expected_phi, atol=1e-12, rtol=1e-12)
    assert np.allclose(metric.dphidrho, expected_dphidrho, atol=1e-12, rtol=1e-12)


def test_phi_b_fallback_noop_for_valid_boozmn():
    boozmn = input_from_boozmn(str(BOOZMN_FILE))
    phi_b_before = boozmn.phi_b_nu.copy()

    result = fill_missing_phi_b_from_wout(boozmn, str(WOUT_FILE))

    assert result is None
    assert np.array_equal(boozmn.phi_b_nu, phi_b_before)


def test_phi_b_fallback_fills_linspace(tmp_path):
    import xarray as xr

    zero_phi_file = tmp_path / "boozmn_zero_phi.nc"
    with xr.load_dataset(str(BOOZMN_FILE)) as ds:
        ds["phi_b"] = (ds["phi_b"].dims, np.zeros_like(ds["phi_b"].to_numpy()))
        ds.to_netcdf(str(zero_phi_file))

    boozmn = input_from_boozmn(str(zero_phi_file))
    result = fill_missing_phi_b_from_wout(boozmn, str(WOUT_FILE))

    with xr.load_dataset(str(WOUT_FILE)) as wds:
        phi_edge = float(wds["phi"].to_numpy()[-1])

    assert result == phi_edge
    assert np.allclose(boozmn.phi_b_nu,
                       np.linspace(0.0, phi_edge, boozmn.ns_b))


def test_phi_b_fallback_rejects_ascii_wout(tmp_path):
    import xarray as xr

    zero_phi_file = tmp_path / "boozmn_zero_phi.nc"
    with xr.load_dataset(str(BOOZMN_FILE)) as ds:
        ds["phi_b"] = (ds["phi_b"].dims, np.zeros_like(ds["phi_b"].to_numpy()))
        ds.to_netcdf(str(zero_phi_file))

    boozmn = input_from_boozmn(str(zero_phi_file))
    with pytest.raises(ValueError, match="NetCDF wout"):
        fill_missing_phi_b_from_wout(boozmn, "wout.txt")


def test_phi_b_fallback_metric_matches_reference(tmp_path):
    """A zero-phi_b boozmn must yield the same metric (incl. the
    analysis-facing aa) as the healthy boozmn, since wout phi is
    uniform in s and only Phi_edge enters the metric."""
    import xarray as xr

    zero_phi_file = tmp_path / "boozmn_zero_phi.nc"
    with xr.load_dataset(str(BOOZMN_FILE)) as ds:
        ds["phi_b"] = (ds["phi_b"].dims, np.zeros_like(ds["phi_b"].to_numpy()))
        ds.to_netcdf(str(zero_phi_file))

    output_file = tmp_path / "metric_boozer.bin.dat"
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        bzx(NTHETA_GKV, NRHO, NTHT, NZETA, ALPHA_FIX,
            str(zero_phi_file), str(WOUT_FILE), str(output_file))
    finally:
        os.chdir(cwd)

    reference_data = read_GKV_metric_file(str(REFERENCE_METRIC_FILE))
    test_data = read_GKV_metric_file(str(output_file))
    for ref_val, test_val in zip(reference_data, test_data):
        if isinstance(ref_val, np.ndarray):
            assert np.allclose(ref_val, test_val, atol=1e-8)
        else:
            assert ref_val == test_val
