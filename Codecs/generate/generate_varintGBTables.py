#!/usr/bin/env python
#
# Generate tables for GroupVarint32
# Copyright 2011 Facebook
#
# @author Tudor Bosman (tudorb@fb.com)
#
# Reference: http://www.stepanovpapers.com/CIKM_2011.pdf
#
# From 17 encoded bytes, we may use between 5 and 17 bytes to encode 4
# integers.  The first byte is a key that indicates how many bytes each of
# the 4 integers takes:
#
# bit 0..1: length-1 of first integer
# bit 2..3: length-1 of second integer
# bit 4..5: length-1 of third integer
# bit 6..7: length-1 of fourth integer
#
# The value of the first byte is used as the index in a table which returns
# a mask value for the SSSE3 PSHUFB instruction, which takes an XMM register
# (16 bytes) and shuffles bytes from it into a destination XMM register
# (optionally setting some of them to 0)
#
# For example, if the key has value 4, that means that the first integer
# uses 1 byte, the second uses 2 bytes, the third and fourth use 1 byte each,
# so we set the mask value so that
#
# r[0] = a[0]
# r[1] = 0
# r[2] = 0
# r[3] = 0
#
# r[4] = a[1]
# r[5] = a[2]
# r[6] = 0
# r[7] = 0
#
# r[8] = a[3]
# r[9] = 0
# r[10] = 0
# r[11] = 0
#
# r[12] = a[4]
# r[13] = 0
# r[14] = 0
# r[15] = 0

import os
from optparse import OptionParser

OUTPUT_FILE = "varintGBTables.cpp"

# The length of the i-th encoded integer in the group described by desc.
def len(desc, i):
    return 1 + ((desc >> (2 * i)) & 3)

def generate(f):
    f.write("""#include <cstdint>
#include "Portability.h"

namespace Codecs {
namespace varintTables {

#if CODECS_X64 || defined(__i386__) || CODECS_PPC64
#if CODECS_AVX_PREREQ(2, 0) || CODECS_SSE_PREREQ(3, 1) 
extern const __m128i varintGB_SSSE3_shfl_msk_m128i[256] = {
""")

	# Compute shuffle control masks.
    for desc in range(0, 256):
        shf = [0xff] * 16
        offset = 0
        for i in range(0, 4):
            length = len(desc, i)
            # The i-th integer uses n bytes, consume them.
            for j in range(0, length):
                shf[4 * i + j] = offset + j
            offset += length
        f.write("\t// desc = {0}\n".format(desc))
        f.write("\t_mm_set_epi8(\n"
                "\t\t0x{15:02X}, 0x{14:02X}, 0x{13:02X}, 0x{12:02X},\n"
                "\t\t0x{11:02X}, 0x{10:02X}, 0x{9:02X}, 0x{8:02X},\n"
                "\t\t0x{7:02X}, 0x{6:02X}, 0x{5:02X}, 0x{4:02X},\n"
                "\t\t0x{3:02X}, 0x{2:02X}, 0x{1:02X}, 0x{0:02X}),\n".format(*shf))
    f.write("};\n"
            "#endif /* __AVX2__ || __SSSE3__ */\n")

	# Also compute total encoded lengths, including key byte.
    f.write("\n"
            "// Number of bytes for the group described by desc.\n"
            "extern const uint8_t varintGBInputOffsets[256] = {")
    for desc in range(0, 256):
        offset = 1  # include key byte
        for i in range(0, 4):
            offset += len(desc, i)
        if desc % 16 == 0:
            f.write("\n\t")
        f.write("{0:2}, ".format(offset))
    f.write("\n"
            "};\n")

	# Also compute individual encoded lengths.
    f.write("\n"
            "// Length of each encoded integer in the group described by desc.\n"
            "extern const uint8_t varintGBLengths[256][4] = {\n")
    for desc in range(0, 256):
        lengths = [0] * 4
        for i in range(0, 4):
            lengths[i] = len(desc, i)
        f.write("\t{")
        f.write("{0}, {1}, {2}, {3}".format(*lengths))
        f.write("}}, // desc = {0}\n".format(desc))
    f.write("};\n")

    f.write("#endif /* x86[_64] */\n"
            "\n"
            "} // namespace varintTables\n"
            "} // namespace Codecs\n")

def main():
    parser = OptionParser()
    parser.add_option("--install_dir", dest="install_dir", default=".",
                      help="write output to DIR", metavar="DIR")
    parser.add_option("--fbcode_dir")
    (options, args) = parser.parse_args()
    f = open(os.path.join(options.install_dir, OUTPUT_FILE), "w")
    generate(f)
    f.close()

if __name__ == "__main__":
    main()
