#
# InjectTF2
# A fault injection framework for TensorFlow 2
#
# TU-Dresden, Institute of Automation (IfA)
#

import struct
from codecs import decode


def float_to_bin32(value):
    """Convert float to 32-bit binary string."""
    [d] = struct.unpack(">L", struct.pack(">f", value))
    return "{:032b}".format(d)


def bin_to_float32(b):
    """Convert binary string to a float."""
    bf = int_to_bytes(int(b, 2), 4)  # 4 bytes needed for IEEE 754 binary32.
    return struct.unpack(">f", bf)[0]


def float_to_bin64(value):
    """Convert float to 64-bit binary string."""
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return "{:064b}".format(d)


def bin_to_float64(b):
    """Convert binary string to a float."""
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack(">d", bf)[0]


# TODO refactor: use int.to_bytes() function instead
def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.
        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode("%%0%dx" % (length << 1) % n, "hex")[-length:]
