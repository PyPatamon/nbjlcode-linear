#!/usr/bin/env python
import os
from optparse import OptionParser

def clz(x):
    assert 0 <= x <= 255
    if x == 0:
        return 8
    n = 0
    if x <= 0x0F:
        n += 4
        x <<= 4
    if x <= 0x3F:
        n += 2
        x <<= 2
    if x <= 0x7F:
        n += 1
    return n

def ctz(x):
    assert 0 <= x <= 255
    if x == 0:
        return 8
    n = 0
    if x & 0x0F == 0:
        n += 4
        x >>= 4
    if x & 0x03 == 0:
        n += 2
        x >>= 2
    if x & 0x01 == 0:
        n += 1
    return n

def num(desc):
    return bin(desc).count('1')

def len(desc, i):
    length = 0
    for j in range(0, i + 1):
        desc >>= length
        length = 1 + ctz(desc)
    return length

def rem(desc):
    return clz(desc)

def constructShfMsk(desc, state):
    shf = [0xff] * 32
    offset = 0
    k = 0
    s = 4 - state
    for i in range(0, num(desc)):
        for j in range(0, len(desc, i)):
            shf[k + j] = offset + j
        offset += len(desc, i)
        k += s
        s = 4
    for j in range(0, rem(desc)):
        shf[k + j] = offset + j
    return shf


def generate_varintGUTables(f):
    f.write("""#include <cstdint>
#include "Portability.h"

namespace Codecs {
namespace varintTables {

#if CODECS_X64 || defined(__i386__) || CODECS_PPC64
#if CODECS_SSE_PREREQ(3, 1)
extern const __m128i varintGU_SSSE3_shfl_msk_m128i[4][256][2] = {
""")

    # Construct shuffle constrol masks.
    for state in range(0, 4):
        f.write("\t{{ // state = {0}\n".format(state))
        for desc in range(0, 256):
            shf = constructShfMsk(desc, state)
            f.write("\t  {{ // desc = {0}\n".format(desc))
            f.write("\t\t_mm_set_epi8(\n"
                    "\t\t\t0x{15:02X}, 0x{14:02X}, 0x{13:02X}, 0x{12:02X},\n"
                    "\t\t\t0x{11:02X}, 0x{10:02X}, 0x{9:02X}, 0x{8:02X},\n"
                    "\t\t\t0x{7:02X}, 0x{6:02X}, 0x{5:02X}, 0x{4:02X},\n"
                    "\t\t\t0x{3:02X}, 0x{2:02X}, 0x{1:02X}, 0x{0:02X}),\n".format(*shf))
            f.write("\t\t_mm_set_epi8(\n"
                    "\t\t\t0x{15:02X}, 0x{14:02X}, 0x{13:02X}, 0x{12:02X},\n"
                    "\t\t\t0x{11:02X}, 0x{10:02X}, 0x{9:02X}, 0x{8:02X},\n"
                    "\t\t\t0x{7:02X}, 0x{6:02X}, 0x{5:02X}, 0x{4:02X},\n"
                    "\t\t\t0x{3:02X}, 0x{2:02X}, 0x{1:02X}, 0x{0:02X})\n".format(*shf[16:]))
            f.write("\t  },\n")
        f.write("\t},\n")
    f.write("};\n"
            "#endif /* __SSSE3__ */\n")

    f.write("""
#if CODECS_AVX_PREREQ(2, 0)
extern const __m256i varintGU_AVX2_shfl_msk_m256i[4][256] = {
""")
    for state in range(0, 4):
        f.write("\t{{ // state = {0}\n".format(state))
        for desc in range(0, 256):
            shf = constructShfMsk(desc, state)
            f.write("\t\t// desc = {0}\n".format(desc))
            f.write("\t\t_mm256_set_epi8(\n"
                    "\t\t\t0x{31:02X}, 0x{30:02X}, 0x{29:02X}, 0x{28:02X},\n"
                    "\t\t\t0x{27:02X}, 0x{26:02X}, 0x{25:02X}, 0x{24:02X},\n"
                    "\t\t\t0x{23:02X}, 0x{22:02X}, 0x{21:02X}, 0x{20:02X},\n"
                    "\t\t\t0x{19:02X}, 0x{18:02X}, 0x{17:02X}, 0x{16:02X},\n"
                    "\t\t\t0x{15:02X}, 0x{14:02X}, 0x{13:02X}, 0x{12:02X},\n"
                    "\t\t\t0x{11:02X}, 0x{10:02X}, 0x{9:02X}, 0x{8:02X},\n"
                    "\t\t\t0x{7:02X}, 0x{6:02X}, 0x{5:02X}, 0x{4:02X},\n"
                    "\t\t\t0x{3:02X}, 0x{2:02X}, 0x{1:02X}, 0x{0:02X}),\n".format(*shf))
        f.write("\t},\n")
    f.write("};\n"
            "#endif /* __AVX2__ */\n")

    # Also compute encoded lengths
    f.write("\n"
            "// Length of each encoded integer in the group described by desc.\n"
            "extern const uint8_t varintGULengths[256][8] = {\n")
    for desc in range(0, 256):
        lengths = [0] * 8
        for i in range(0, num(desc)):
            lengths[i] = len(desc, i)
        if rem(desc) > 0:
            lengths[num(desc)] = rem(desc)
        f.write("\t{")
        f.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}".format(*lengths))
        f.write("},\n")
    f.write("};\n")

    # Also compute number of integers whose encoding is complete in the group described by desc.
    f.write("\n"
            "// Number of integers whose encoding is complete in the group described by desc.\n"
            "extern const uint8_t varintG8IUOutputOffsets[256] = {")
    for desc in range(0, 256):
        if desc % 16 == 0:
            f.write("\n\t")
        f.write("{0}, ".format(num(desc)))
    f.write("\n"
            "};\n")

	# Also compute number of encoded bytes in the group described by the descriptor desc.
    f.write("\n"
            "// Number of encoded bytes in the group described by desc.\n"
            "extern const uint8_t varintG8CUOutputOffsets[4][256] = {\n")
    for state in range(0, 4):
        f.write("\t{{ // state = {0}".format(state))
        for desc in range(0, 256):
            if desc % 16 == 0:
                f.write("\n\t ")
            outputoffset = 4 * num(desc) - state + rem(desc)
            f.write("{0:2}, ".format(outputoffset))
        f.write("\n"
                "\t},\n")
    f.write("};\n")

    # Also compute number of leading 0s in the descriptor desc.
    f.write("\n"
            "// Number of leading 0s in desc.\n"
            "extern const uint8_t varintG8CUStates[256] = {")
    for desc in range(0, 256):
        if desc % 16 == 0:
            f.write("\n\t")
        f.write("{0}, ".format(rem(desc)))
    f.write("\n"
            "};\n")

    f.write("\n"
            "extern const uint32_t kMask[4] = {\n"
            "\t0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF\n")
    f.write("};\n")

    f.write("#endif /* x86[_64] */\n"
            "\n"
            "} // varintTables\n"
            "} // namespace Codecs\n")

def main():
    f = open("varintGUTables.cpp", "w")
    generate_varintGUTables(f)
    f.close()

if __name__ == "__main__":
	main()
