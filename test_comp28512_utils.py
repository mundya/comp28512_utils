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
    data = np.random.choice([True, False], size=1000)

    assert np.all(comp28512_utils.insert_bit_errors(data, 0) == data)
    assert np.all(comp28512_utils.insert_bit_errors(data, 1) == ~data)


@pytest.mark.parametrize(
    "string, bits",
    [(np.array([66, 66, 67], dtype=np.uint8),
      np.array([False, True, False, False, False, False, True, False,
                False, True, False, False, False, False, True, False,
                False, True, False, False, False, False, True, True])),
     (np.array([48], dtype=np.uint8),
      np.array([False, False, True, True, False, False, False, False])),
     (np.array([255], dtype=np.uint16),
      np.array([True]*8 + [False]*8)),  # Little-endian
     (np.array([31], dtype=np.uint16),
      np.array([False]*3 + [True]*5 + [False]*8)),  # Little-endian
     ])
def test_numpy_array_to_bit_array(string, bits):
    """Test creation of a bit array from a NumPy array."""
    assert np.all(comp28512_utils.numpy_array_to_bit_array(string) == bits)


@pytest.mark.parametrize(
    "string, bits",
    [(np.array([66, 66, 67], dtype=np.uint8),
      np.array([False, True, False, False, False, False, True, False,
                False, True, False, False, False, False, True, False,
                False, True, False, False, False, False, True, True])),
     (np.array([48], dtype=np.uint8),
      np.array([False, False, True, True, False, False, False, False])),
     (np.array([255], dtype=np.uint16),
      np.array([True]*8 + [False]*8)),  # Little-endian
     (np.array([31], dtype=np.uint16),
      np.array([False]*3 + [True]*5 + [False]*8)),  # Little-endian
     ])
def test_bit_array_to_numpy_array(string, bits):
    """Test converting bit arrays to NumPy arrays."""
    assert np.all(
        string ==
        comp28512_utils.bit_array_to_numpy_array(bits, dtype=string.dtype)
    )


@pytest.mark.parametrize(
    "vals",
    [np.random.uniform(size=1000)]
)
def test_numpy_biconvert(vals):
    """Test np->bits and bits->np"""
    assert np.all(vals == comp28512_utils.bit_array_to_numpy_array(
        comp28512_utils.numpy_array_to_bit_array(vals), dtype=vals.dtype))
