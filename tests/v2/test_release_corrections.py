"""CPU regression tests for scientifically consistent partial-regression plots."""
import numpy as np
import pandas as pd

from ncclenergy.v2.release_tables import temperature_partial_coordinates
from ncclenergy.v2.measval_posthoc import fit, load
from ncclenergy.v2.common import REPO


def test_temperature_axes_use_the_same_cell_residualization():
    d = pd.DataFrame({"arm": ["A", "A", "B", "B"], "length_s": [10] * 4,
                      "temp_c": [30., 32., 40., 42.], "y": [1., 1.02, 5., 5.02]})
    p = temperature_partial_coordinates(d)
    assert np.allclose(p.temp_resid_c, [-1., 1., -1., 1.])
    assert np.allclose(p.resid, [-.01, .01, -.01, .01])
    slope = np.dot(p.temp_resid_c, p.resid) / np.dot(p.temp_resid_c, p.temp_resid_c)
    assert np.isclose(slope, .01)
    assert np.isclose(slope, fit(d, True)[0])


def test_real_gate3b_plot_slope_matches_the_unchanged_full_model():
    d = load(REPO / "data/v2.2/raw/v2.2-measval-01")
    p = temperature_partial_coordinates(d)
    slope = np.dot(p.temp_resid_c, p.resid) / np.dot(p.temp_resid_c, p.temp_resid_c)
    assert np.isclose(slope, fit(d, True)[0], rtol=1e-10, atol=1e-12)
    assert np.isclose(slope, .004304742933811, rtol=1e-7)


def test_csv_embedded_numerical_arrays_use_tolerance_not_string_identity():
    from ncclenergy.v2.release_repro import _num_equal
    assert _num_equal("[0.1, 1.0000000000000002]", "[0.1, 1.0]")[0]
    assert not _num_equal("[0.1, 1.01]", "[0.1, 1.0]")[0]
    assert not _num_equal({"byte_identical": False}, {"byte_identical": True})[0]
