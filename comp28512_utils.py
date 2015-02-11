"""Support utilities for COMP28512.

(C) University of Manchester 2015
Author: Andrew Mundy

Revisions
---------
31/01/2015:
    Included `get_pesq_scores` and generally neatened.
05/02/2015:
    Fixed Audio to avoid scaling altogether and instead warn if students fail
    to quantise their values.  If floats are passed then they are cast to int16
    and the student warned that this has occurred.
06/02/2015:
    a) Initialise a logging handler because IPython < 2.0 doesn't appear to do
       this.
    b) Call `display_html` to allow any number of Audio elements to appear in a
       block.  Also write WAV files to a temporary directory.
11/02/2015:
    - String to NumPy bool arrays
    - Option to display in Audio
    - Insert bit errors
"""
from __future__ import print_function

import base64
import collections
import logging
import numpy as np
import re
from scipy.io import wavfile
import struct
import tempfile

try:
    from IPython import display
except ImportError:
    display = None

logging.basicConfig()
logger = logging.getLogger(__name__)


def Audio(data, rate, filename=None, display_audio=True):
    """Save data to a file then play."""
    # Get a filename
    if filename is None:
        (_, filename) = tempfile.mkstemp(".wav", "comp28512_")

    # If none of the values are outside the range +/- 1 we remind the student
    # to scale!
    if data.dtype.kind == 'f':
        if np.max(data) <= 1.0 and np.min(data) >= -1.0:
            logger.warn("Data values fall in the range +/- 1.0, you need to "
                        "scale by 32767 (2^15 - 1) to listen to audio.")
        else:
            logger.warn("Casting data to int16.")
            data = np.int16(data)

    # Write the data
    wavfile.write(filename, rate, data)
    print("Data written to {}.".format(filename))

    # Read the data back in, display as HTML
    if display_audio:
        display.display_html(audio_from_file(filename))


def get_audio_from_file(filename):
    with open(filename, 'rb') as f:
        return "data:audio/wav;base64," + base64.encodestring(f.read())


def audio_from_file(filename):
    """Play the data from a file"""
    # Return an HTML audio object with base64 encoded audio data included.
    # If file:// could return correct MIMEtypes this would be much nicer.
    return display.HTML(
        "<p><audio controls=\"controls\">"
        "<source src=\"{}\" type=\"audio/wav\" />"
        "Your browser does not support the AUDIO element. "
        "Open the relevant file to listen."
        "</audio></p>".format(get_audio_from_file(filename)))


def get_pesq_scores(filename="pesq_results.txt"):
    """Get the PESQ scores as a double dictionary mapping ref to deg to result.

    e.g., if you ran "pesq +xxxx ref.wav deg.wav" running "get_pesq_scores()"
    would return a dictionary.  To extract the result one would type:

        results = get_pesq_scores()
        print results["ref.wav"]["deg.wav"]
    """
    results = collections.defaultdict(dict)
    pesq_re = re.compile(
        r"^(?P<ref>[^+]+\.wav)\s+(?P<deg>[^+]+\.wav)\s+(?P<score>\d+\.\d+)")

    with open(filename, "rb") as pesq_file:
        # Ignore the first line (it's just a header)
        pesq_file.readline()

        # Get each result and add it to the dictionary
        for result in pesq_file:
            m = pesq_re.match(result)

            if m is None:
                print(result)
                continue

            ref = m.group('ref')
            deg = m.group('deg')
            if deg in results[ref]:
                logger.warn("'{}' is compared to '{}' multiple times, "
                            "only the most recent comparison will be stored.".
                            format(deg, ref))

            results[ref][deg] = float(m.group('score'))

    return results


def bytes_to_bit_array(bytestring):
    """Converts a string of bytes to a NumPy array of boolean types.

    Example:

        >>> bytes_to_bit_array(b"BBC")
        array([False,  True, False, False, False, False,  True, False, False,
                True, False, False, False, False,  True, False, False,  True,
               False, False, False, False,  True,  True], dtype=bool)
    """
    def byte_to_bits(byte):
        """Convert a single byte to bits."""
        bits = struct.unpack("B", byte)[0]
        bit_vals = []

        for i in range(8):
            bit_vals.append((1 << (7 - i)) & bits != 0)

        return bit_vals

    return np.array(
        [byte_to_bits(b) for b in bytestring]).reshape(8 * len(bytestring))


def bit_array_to_bytes(bitarray):
    """Converts a bit array to a string of bytes.

    Example:

        >>> b = [False, True, False, False, False, False, True, False]
        >>> bit_array_to_bytes(np.array(b))
        'B'
    """
    def bits_to_byte(bits):
        """Convert an array of bits to a byte."""
        byte = 0

        for (i, b) in enumerate(bits):
            if b:
                byte |= (1 << (7 - i))

        return struct.pack('B', byte)

    # Reshape the bit array, then call `bits_to_byte` on each row.
    assert bitarray.size % 8 == 0
    bits = bitarray.reshape((bitarray.size / 8, 8))
    bytearray = []

    for bs in bits:
        bytearray.append(bits_to_byte(bs))

    return ''.join(bytearray)


def insert_bit_errors(data, bit_error_probability):
    """Insert bit errors into some data with given probability."""
    errors = np.random.uniform(size=data.size) < bit_error_probability
    return errors ^ data


def insert_bursty_errors(data, bit_error_probability):
    """Insert bursty errors into some data."""
    raise NotImplementedError
