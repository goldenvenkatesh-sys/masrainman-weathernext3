# Checks of the EFI / SOT maths against known cases.  Run: python -m pytest tests
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import efi_core as E  # noqa: E402

PROBS = [0.10, 0.25, 0.50, 0.75, 0.90]
P = np.clip(E.CLIM_PROBS, 1e-4, 1 - 1e-4)
CLIM = norm.ppf(P)[:, None]                        # standard normal climate, one point


def fc(mu, sd=1.0):
    return (mu + sd * norm.ppf(PROBS))[:, None]


def test_same_as_climate_is_zero():
    assert abs(E.efi_from_percentiles(CLIM, fc(0), PROBS)[0]) < 0.02


def test_sign_and_order():
    e = [E.efi_from_percentiles(CLIM, fc(m), PROBS)[0] for m in (-2, -1, 0, 1, 2)]
    assert e == sorted(e) and e[0] < -0.5 and e[-1] > 0.5
    assert abs(e[0] + e[-1]) < 1e-3                 # symmetric


def test_beyond_climate_is_one():
    assert E.efi_from_percentiles(CLIM, fc(10, 0.5), PROBS)[0] > 0.999
    assert E.efi_from_percentiles(CLIM, fc(-10, 0.5), PROBS)[0] < -0.999


def test_members_agree_with_percentiles():
    members = np.random.default_rng(1).normal(1.5, 1.0, (2000, 1))
    a = E.efi_from_members(CLIM, members)[0]
    b = E.efi_from_percentiles(CLIM, fc(1.5), PROBS)[0]
    assert abs(a - b) < 0.06


def test_dry_forecast_in_dry_climate_is_zero():
    zero = np.zeros((101, 1))
    assert abs(E.efi_from_percentiles(zero, np.zeros((5, 1)), PROBS, 0.0)[0]) < 1e-6


def test_sot():
    # forecast 90th exactly at climate 99th -> 0; one (Qc99 - Qc90) beyond -> 1
    c90, c99 = CLIM[90, 0], CLIM[99, 0]
    assert abs(E.sot_upper(CLIM, np.array([c99]))[0]) < 1e-6
    assert abs(E.sot_upper(CLIM, np.array([2 * c99 - c90]))[0] - 1) < 1e-5
    assert abs(E.sot_lower(CLIM, np.array([CLIM[1, 0]]))[0]) < 1e-6
    assert np.isnan(E.sot_upper(np.zeros((101, 1)), np.array([5.0]))[0])


def test_grid_shapes():
    clim = np.repeat(CLIM[:, :, None], 4, axis=2).repeat(3, axis=1)   # (101, 3, 4)
    f = np.repeat(fc(1)[:, :, None], 4, axis=2).repeat(3, axis=1)     # (5, 3, 4)
    assert E.efi_from_percentiles(clim, f, PROBS).shape == (3, 4)
