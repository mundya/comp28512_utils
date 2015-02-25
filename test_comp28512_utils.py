"""Tests for COMP28512 utils.
"""
import comp28512_utils
import numpy as np
import pytest


@pytest.mark.parametrize(
    "string, bools",
    [(b"BBC", np.array([False, True, False, False, False, False, True, False,
                        False, True, False, False, False, False, True, False,
                        False, True, False, False, False, False, True, True])),
     (b"0", np.array([False, False, True, True, False, False, False, False])),
     ])
def test_bytes_to_bit_array(string, bools):
    assert np.all(comp28512_utils.bytes_to_bit_array(string) == bools)


@pytest.mark.parametrize(
    "string, bits",
    [(b"BBC", np.array([False, True, False, False, False, False, True, False,
                        False, True, False, False, False, False, True, False,
                        False, True, False, False, False, False, True, True])),
     (b"0", np.array([False, False, True, True, False, False, False, False])),
     ])
def test_bit_array_to_bytes(bits, string):
    assert string == comp28512_utils.bit_array_to_bytes(np.array(bits))


@pytest.mark.parametrize(
    "string",
    ["Hello, Manchester!",
     "123@~",
     ])
def test_bytes_bit_bytes(string):
    """Feed the snake its tail."""
    assert string == comp28512_utils.bit_array_to_bytes(
        comp28512_utils.bytes_to_bit_array(string))


def test_insert_bit_errors():
    """Test inserting bit errors with probability 0 or 1."""
    data = np.array([True, False])

    assert np.all(comp28512_utils.insert_bit_errors(data, 0) == data)
    assert np.all(comp28512_utils.insert_bit_errors(data, 1) == ~data)
