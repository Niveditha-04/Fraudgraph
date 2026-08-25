"""Unit tests for models/benford.py's Benford's Law deviation score."""
import numpy as np

from models.benford import BENFORD_EXPECTED, benford_mad, first_significant_digit


def test_benford_expected_distribution_sums_to_one():
    assert abs(BENFORD_EXPECTED.sum() - 1.0) < 1e-9
    assert BENFORD_EXPECTED[0] > BENFORD_EXPECTED[-1]  # digit 1 more common than digit 9


def test_first_significant_digit_basic_cases():
    values = np.array([0.0034, 152.7, 9.99, 1000.0])
    digits = first_significant_digit(values)
    assert list(digits) == [3, 1, 9, 1]


def test_first_significant_digit_drops_non_positive():
    values = np.array([-5.0, 0.0, 3.0])
    digits = first_significant_digit(values)
    assert list(digits) == [3]


def test_benford_mad_perfect_match_is_zero():
    # construct values whose first digits exactly follow Benford's expected
    # proportions for a moderately large synthetic sample
    rng = np.random.default_rng(0)
    log_uniform = 10 ** rng.uniform(0, 4, size=100_000)
    mad = benford_mad(log_uniform)
    assert mad is not None
    assert mad < 0.01  # should be very close to Benford's Law by construction


def test_benford_mad_none_below_minimum_sample_size():
    assert benford_mad(np.array([1.0, 2.0])) is None  # fewer than 3 positive values


def test_benford_mad_uniform_first_digits_deviates_from_benford():
    # all first digits forced to 5 -- maximally non-Benford
    values = np.array([5.0] * 20)
    mad = benford_mad(values)
    assert mad is not None
    assert mad > 0.1  # should show a clear deviation
