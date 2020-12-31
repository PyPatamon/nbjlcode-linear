#!/usr/bin/env python
import os
from optparse import OptionParser

def generate_SIMDMasks(f):
    f.write("""#include <cstdint>
#include "Portability.h"

namespace Codecs {
namespace SIMDMasks {
""")

    # Compute SSE AND masks.
    f.write("\n"
            "#if CODECS_SSE_PREREQ(2, 0)\n"
            "extern const __m128i SSE2_and_msk_m128i[33] = {\n")
    for b in range(0, 33):
        f.write("\t_mm_set1_epi32(0x{0:X}),\t// {1}-bit\n".format((1<<b) - 1, b))
    f.write("};\n"
            "#endif /* __SSE2__ */\n")

    # Compute AVX AND masks.
    f.write("\n"
            "#if CODECS_AVX_PREREQ(2, 0)\n"
            "extern const __m256i AVX2_and_msk_m256i[33] = {\n")
    for b in range(0, 33):
        f.write("\t_mm256_set1_epi32(0x{0:X}),\t// {1}-bit\n".format((1<<b) - 1, b))
    f.write("};\n"
            "#endif /* __AVX2__ */\n")

    bexceptions = [27, 29, 30, 31]
    # Compute shuffle control masks.
    shfl = [[[-1]*16 for i in range(2)] for i in range(33)]
    shflexceptions = dict()
    shflexceptions[27] = [[0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13],
                          [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13]];
    shflexceptions[29] = [[0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14],
                          [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14]];
    shflexceptions[30] = [[0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14],
                          [0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14]];
    shflexceptions[31] = [[-1, -1, -1, -1, -1, -1, -1, -1, 7, 8, 9, 10, 11, 12, 13, 14],
                          [1, 2, 3, 4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1]];
    f.write("\n"
            "#if CODECS_SSE_PREREQ(4, 1)\n"
            "extern const char Hor_shfl_msk_char[33][2][16] = {\n")
    for b in range(0, 33):
        f.write("\t{{ // {0}-bit\n".format(b))
        if b in bexceptions:
            shfl[b] = shflexceptions[b]
        else:
            m = 4*b if b <= 16 else 4*b % 8
            for i in range(0, 2):
                for k in range(0, 4):
                    j = 4 * k
                    p = (i*m + k*b) / 8
                    r = (i*m + (k+1)*b - 1) / 8
                    for q in range(p, r + 1):
                        shfl[b][i][j] = q
                        j += 1
        for i in range(0, 2):
            f.write("\t\t{\n")
            f.write("\t\t  {15:>2}, {14:>2}, {13:>2}, {12:>2},\n"
                    "\t\t  {11:>2}, {10:>2}, {9:>2}, {8:>2},\n"
                    "\t\t  {7:>2}, {6:>2}, {5:>2}, {4:>2},\n"
                    "\t\t  {3:>2}, {2:>2}, {1:>2}, {0:>2},\n".format(*shfl[b][i]))
            f.write("\t\t},\n")
        f.write("\t},\n")
    f.write("};\n")

    # Compute:
    # 1. SSE multiplication masks;
    # 2. SSE logical shift right imms;
    # 3. AVX logical shift right vectors.
    # and write out SSE multiplication masks.
    mul = [[[0x01]*4 for i in range(2)]  for i in range(33)]
    mulexceptions = dict()
    mulexceptions[27] = [[0x08, 0x01, 0x08, 0x04], [0x02, 0x20, 0x08, 0x01]]
    mulexceptions[29] = [[0x04, 0x04, 0x01, 0x01], [0x04, 0x04, 0x01, 0x01]]
    mulexceptions[30] = [[0x04, 0x04, 0x01, 0x01], [0x04, 0x04, 0x01, 0x01]]
    mulexceptions[31] = [[0x01, 0x01, 0x01, 0x01], [0x02, 0x02, 0x02, 0x01]]

    srli = [[0]*2 for i in range(33)]
    srliexceptions = dict()
    srliexceptions[27] = [3, 5]
    srliexceptions[29] = [2, 3]
    srliexceptions[30] = [2, 2]
    srliexceptions[31] = [0, 1]

    srlv = [[[0]*4 for i in range(2)]  for i in range(33)]
    srlvexceptions = dict()
    srlvexceptions[27] = [[0, 3, 5, 0], [5, 0, 2, 5]]
    srlvexceptions[29] = [[3, 0, 3, 0], [3, 0, 3, 0]]
    srlvexceptions[30] = [[2, 0, 2, 0], [2, 0, 2, 0]]
    srlvexceptions[31] = [[1, 0, 1, 0], [1, 0, 1, 0]]

    f.write("\n"
            "extern const __m128i Hor_SSE4_mul_msk_m128i[33][2] = {\n")
    for b in range(0, 33):
        if b in bexceptions:
            mul[b] = mulexceptions[b]
            srli[b] = srliexceptions[b]
            srlv[b] = srlvexceptions[b]
        else:
            for i in range(0, 2):
                for j in range(0, 4):
                    k = 4*i + j
                    srlv[b][i][j] = k*b % 8
                srli[b][i] = max(srlv[b][i][0:4])
                for j in range(0, 4):
                    mul[b][i][j] = 1 << (srli[b][i] - srlv[b][i][j])
        f.write("\t{{ _mm_set_epi32(0x{3:02X}, 0x{2:02X}, 0x{1:02X}, 0x{0:02X}), ".format(*mul[b][0]))
        f.write("_mm_set_epi32(0x{3:02X}, 0x{2:02X}, 0x{1:02X}, 0x{0:02X}) }},".format(*mul[b][1]))
        f.write("\t// {0}-bit\n".format(b))
    f.write("};\n")

    # Write out SSE logical shift right imms.
    f.write("\n"
            "extern const int Hor_SSE4_srli_imm_int[33][2] = {\n")
    for b in range(0, 33):
        f.write("\t{{ {0}, {1} }},".format(*srli[b]))
        f.write("\t// {0}-bit\n".format(b))
    f.write("};\n"
            "#endif /* __SSE4_1__ */\n")

    # Write out AVX shuffle control masks.
    f.write("\n"
            "#if CODECS_AVX_PREREQ(2, 0)\n"
            "extern const __m256i Hor_AVX2_shfl_msk_m256i[33] = {\n")
    for b in range(0, 33):
        f.write("\t // {0}-bit\n".format(b))
        f.write("\t_mm256_set_epi8(\n"
                "\t\t{31:>2}, {30:>2}, {29:>2}, {28:>2},\n"
                "\t\t{27:>2}, {26:>2}, {25:>2}, {24:>2},\n"
                "\t\t{23:>2}, {22:>2}, {21:>2}, {20:>2},\n"
                "\t\t{19:>2}, {18:>2}, {17:>2}, {16:>2},\n"
                "\t\t{15:>2}, {14:>2}, {13:>2}, {12:>2},\n"
                "\t\t{11:>2}, {10:>2}, {9:>2}, {8:>2},\n"
                "\t\t{7:>2}, {6:>2}, {5:>2}, {4:>2},\n"
                "\t\t{3:>2}, {2:>2}, {1:>2}, {0:>2}),\n".format(*shfl[b][0]+shfl[b][1]))
    f.write("};\n")

    # Write out AVX logical shift right vectors.
    f.write("\n"
            "extern const __m256i Hor_AVX2_srlv_msk_m256i[33] = {\n")
    for b in range(0, 33):
        f.write("\t_mm256_set_epi32({7}, {6}, {5}, {4}, {3}, {2}, {1}, {0}), ".format(*srlv[b][0]+srlv[b][1]))
        f.write("// {0}-bit\n".format(b))
    f.write("};\n"
            "#endif /* __AVX2__ */\n")

    # Compute Simple family SSE multiplication masks.
    f.write("\n"
            "#if CODECS_SSE_PREREQ(4, 1)\n"
            "extern const __m128i Simple_SSE4_mul_msk_m128i[9] = {\n")
    cases = [1, 2, 3, 4, 5, 7, 9, 14, 28]
    for b in cases:
        mul = [0x01] * 4
        n = min(4, 28 / b)
        for i in range(0, n):
            mul[i] = 1 << (i * b)
        f.write("\t_mm_set_epi32(0x{3:06X}, 0x{2:06X}, 0x{1:06X}, 0x{0:06X}),".format(*mul))
        f.write("\t// {0} * {1}-bit\n".format(28 / b, b))
    f.write("};\n"
            "#endif /* __SSE4_1__ */\n")


    # Compute Simple family AVX logical shift right vectors.
    f.write("\n"
            "#if CODECS_AVX_PREREQ(2, 0)\n"
            "extern const __m256i Simple_AVX2_srlv_msk_m256i[9][4] = {\n")
    for b in cases:
        srlv = [0] * 32
        for i in range(0, 28 / b):
            srlv[i] = 28 - (i + 1) * b
        f.write("\t{{ // {0} * {1}\n".format(28 / b, b))
        f.write("\t  _mm256_set_epi32({7:>2}, {6:>2}, {5:>2}, {4:>2}, {3:>2}, {2:>2}, {1:>2}, {0:>2}),\n".format(*srlv))
        f.write("\t  _mm256_set_epi32({7:>2}, {6:>2}, {5:>2}, {4:>2}, {3:>2}, {2:>2}, {1:>2}, {0:>2}),\n".format(*srlv[8:]))
        f.write("\t  _mm256_set_epi32({7:>2}, {6:>2}, {5:>2}, {4:>2}, {3:>2}, {2:>2}, {1:>2}, {0:>2}),\n".format(*srlv[16:]))
        f.write("\t  _mm256_set_epi32({7:>2}, {6:>2}, {5:>2}, {4:>2}, {3:>2}, {2:>2}, {1:>2}, {0:>2}),\n".format(*srlv[24:]))
        f.write("\t},\n")
    f.write("};\n"
            "#endif /* __AVX2__ */\n")

    f.write("\n"
            "} // namespace SIMDMasks\n"
            "} // namespace Codecs\n")

def main():
    f = open("SIMDMasks.cpp", "w")
    generate_SIMDMasks(f)
    f.close()

if __name__ == "__main__":
	main()
