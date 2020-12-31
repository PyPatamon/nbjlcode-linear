/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Mar 10, 2015
 */

#ifndef CODECS_VARINTG8IU_H_
#define CODECS_VARINTG8IU_H_ 

#if !defined(__GNUC__) && !defined(_MSC_VER)
#error varintG8IU.h requires GCC or MSVC.
#endif

#include "Bits.h"
#include "IntegerCodec.h"
#include "Portability.h"

#if !CODECS_X64 && !defined(__i386__) && !CODECS_PPC64
#error varintG8IU.h requires x86[_64].
#endif


#if CODECS_SSE_PREREQ(3, 1)
namespace Codecs {
namespace varintTables {
extern const __m128i varintGU_SSSE3_shfl_msk_m128i[4][256][2];
} // namespace varintTables
} // namespace Codecs
#endif /* __SSSE3__ */

#if CODECS_AVX_PREREQ(2, 0)
namespace Codecs {
namespace varintTables {
extern const __m256i varintGU_AVX2_shfl_msk_m256i[4][256];
} // namespace varintTables
} // namespace Codecs
#endif /* __AVX2__ */

namespace Codecs {
namespace varintTables {
extern const uint8_t varintGULengths[256][8];
extern const uint8_t varintG8IUOutputOffsets[256];
extern const uint32_t kMask[4];
} // namespace varintTables
} // namespace Codecs


namespace Codecs {

template <typename InstructionSet, bool AllowOverShooting>
class varintG8IU;

using varintG8IU_Scalar = varintG8IU<Scalar, false>;

#if CODECS_SSE_PREREQ(3, 1)
using varintG8IU_SSE = varintG8IU<SSE, false>;
#endif /* __SSSE3__ */

#if CODECS_AVX_PREREQ(2, 0)
using varintG8IU_AVX = varintG8IU<AVX, false>;
#endif /* __AVX2__ */

/**
 * varintG8IU encoding for 32-bit integers.
 *
 * varintGU Groups 8 data bytes together along with 1 descriptor byte
 * containing the unary representations of the lengths of each encoded 
 * integer. (1=1 byte 10=2 bytes .. 1000=4 bytes)  The 8 data bytes may
 * encode as few as 2 and as many as 8 integers, depending on their size.
 * In the incomplete block variation, which we call varintG8IU, we store
 * only as many integers as fit in 8 bytes, leaving the data block 
 * incomplete if necessary.
 *
 * This implementation assumes little-endian and does unaligned 32-bit
 * accesses, so it's basically not portable outside of the x86[_64] world.
 *
 * Follows
 *
 * A. A. Stepanov, A. R. Gangolli, D. E. Rose, R. J. Ernst, and P. S. Oberoi.
 * SIMD-based decoding of posting lists. In Proc. Information and Knowledge
 * Management, CIKM, pages 317–326, 2011.
 */
template <typename InstructionSet, bool AllowOverShooting>
class varintG8IUBase : public IntegerCodec {
public:
    enum {
    	kHeaderSize = 1, // Number of bytes for the descriptor.
    	kGroupSize = 8,  // Number of bytes for the data, which is also
                         // the maximum possible number of encoded values.
		kFullGroupSize = kHeaderSize + kGroupSize // Number of bytes for the group.
    };

    static void encodeGroup(const uint32_t * &src, const uint32_t *const endsrc, uint8_t *dst) {
        // Reserve 1 byte for the descriptor.
    	uint8_t *const initdst = dst;
    	++dst; 

    	uint8_t desc = 0;
    	uint8_t totalLength = 0;
		while (endsrc > src) {
			uint8_t length = bytes(src[0]);
			totalLength += length;
			if (totalLength > kGroupSize) {
				break;
			}

			// Flip the corresponding termination bit in desc.
			desc |= static_cast<uint8_t>(1 << (totalLength - 1));

			// Write out the least significant length bytes
            // of src[0] to the buffer pointed-to by dst.
			*reinterpret_cast<uint32_t *>(dst) = src[0];
			dst += length;
			++src;
		}

		initdst[0] = desc;
    }

    virtual void encodeArray(const uint32_t *in, uint64_t nvalue,
                uint32_t *out, uint64_t &csize) {
        const uint32_t *const endin = in + nvalue;
        uint8_t *dst = reinterpret_cast<uint8_t *>(out);
        const uint8_t *const initdst = dst;
        while (endin > in) {
            encodeGroup(in, endin, dst);
            dst += kFullGroupSize;
        }

        // Align to 32-bit word boundary.
        csize = ((dst - initdst) + 3) / 4;
    }

    // varintG8IU_SSE and varintG8IU_AVX inherit decodeArray whereas
    // varintG8IU_Scalar define its own.
    virtual const uint32_t * decodeArray(const uint32_t *in, uint64_t csize,
                uint32_t *out, uint64_t nvalue) {
        const uint8_t *src = reinterpret_cast<const uint8_t *>(in);
        const uint8_t *const initsrc = src;
        const uint32_t *const endout = out + nvalue;

        if (AllowOverShooting) {
            // This implementation is more efficient but
            // can overshoot, which might result in disaster, 
            // e.g., if out points to an array of size nvalue.
            while (endout > out) {
                Derived::decodeGroup(src, out);
                src += kFullGroupSize;
            }
        }
        else {
            // When decoding a group, varintG8IU_SSE and
            // varintG8IU_AVX always output kGroupSize 
            // values. Therefore when there're >= kGroupSize
            // values to be decoded, it's safe to write
            // the decoded values directly to out.
            while (endout - out >= kGroupSize) {
                Derived::decodeGroup(src, out);
                src += kFullGroupSize;
            }

            // For the remaining (< kGroupSize) values, 
            // to avoid overshooting, we write the 
            // decoded values to the array buf first,
            // and then copy only those we need to out.
            uint32_t valuesRemaining = endout - out;
            uint32_t *ptr = buf;
            const uint32_t *const endptr = ptr + valuesRemaining;
            while (endptr > ptr) {
                Derived::decodeGroup(src, ptr);
                src += kFullGroupSize;
            }
            memcpy(out, buf, sizeof(uint32_t) * valuesRemaining);
        }

        csize = ((src - initsrc) + 3) / 4;
        return in + csize;
    }

    virtual std::string name() const {
        return Derived::codecname();
    }

private:
    using Derived = varintG8IU<InstructionSet, AllowOverShooting>;

    // We determine the size of buf based on:
    // 1) There're at most kGroupSize-1 values remaining to be decoded;
    // 2) varintG8IU_SSE and varintG8IU_AVX can write out kGroupSize-1
    //    garbage 0s when decoding the last group.
    uint32_t buf[(kGroupSize - 1) * 2];
};

template <bool AllowOverShooting>
class varintG8IU<Scalar, AllowOverShooting> : public varintG8IUBase<Scalar, AllowOverShooting> {
public:
    using varintG8IUBase<Scalar, AllowOverShooting>::kHeaderSize;
    using varintG8IUBase<Scalar, AllowOverShooting>::kFullGroupSize;

    // Note that varintG8IU_Scalar only decode those values we need,
    // so we are guaranteed that there won't be overshooting.
    virtual const uint32_t * decodeArray(const uint32_t *in, uint64_t csize,
                uint32_t *out, uint64_t nvalue) {
        const uint8_t *src = reinterpret_cast<const uint8_t *>(in);
        const uint8_t *const initsrc = src;
        const uint32_t *const endout = out + nvalue;

        while (endout > out) {
            decodeGroup(src, out);
            src += kFullGroupSize;
        }

        csize = ((src - initsrc) + 3) / 4;
        return in + csize;
    }

    static const std::string codecname() {
        return "varintG8IU_Scalar";
    }

    static void decodeGroup(const uint8_t *src, uint32_t * &dst) {
    	uint8_t desc = src[0];
    	src += kHeaderSize;
        // Table lookup is faster than __builtin_popcount
        // as in "const uint8_t num = __builtin_popcount(desc);".
    	const uint8_t num = varintTables::varintG8IUOutputOffsets[desc]; 
        for (uint8_t i = 0; i < num; ++i) {
            uint8_t length = varintTables::varintGULengths[desc][i];
            assert(length <= 4);
            dst[i] = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[length - 1];
            src += length;
        }

        /*
        // slower alternative
        uint64_t codeword = *reinterpret_cast<const uint64_t *>(src);
        for (uint8_t i = 0; i < num; ++i) {
            uint8_t length = varintTables::varintGULengths[desc][i];
            assert(length <= 4);
            dst[i] = codeword & varintTables::kMask[length - 1];
            codeword >>= length * 8;
        }
        */

        dst += num;
    }
};

#if CODECS_SSE_PREREQ(3, 1)
template <bool AllowOverShooting>
class varintG8IU<SSE, AllowOverShooting> : public varintG8IUBase<SSE, AllowOverShooting> {
public:
    using varintG8IUBase<SSE, AllowOverShooting>::kHeaderSize;

    static const std::string codecname() {
        return "varintG8IU_SSE";
    }

    template <bool branchless = true>
    static void decodeGroup(const uint8_t *src, uint32_t * &dst) {
    	uint8_t desc = src[0];
    	src += kHeaderSize;
    	const uint8_t num = varintTables::varintG8IUOutputOffsets[desc]; 

        const __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src));
        __m128i result = _mm_shuffle_epi8(val, varintTables::varintGU_SSSE3_shfl_msk_m128i[0][desc][0]);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), result);

        // If-conversion
        if (branchless) { 
            result = _mm_shuffle_epi8(val, varintTables::varintGU_SSSE3_shfl_msk_m128i[0][desc][1]);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 4), result);
        }
        else {
            if (num > 4) { // Branch misprediction is costly.
                result = _mm_shuffle_epi8(val, varintTables::varintGU_SSSE3_shfl_msk_m128i[0][desc][1]);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 4), result);
            }
        }

        dst += num;
    }
};
#endif /* __SSSE3__ */

#if CODECS_AVX_PREREQ(2, 0)
template <bool AllowOverShooting>
class varintG8IU<AVX, AllowOverShooting> : public varintG8IUBase<AVX, AllowOverShooting> {
public:
    using varintG8IUBase<AVX, AllowOverShooting>::kHeaderSize;

    static const std::string codecname() {
        return "varintG8IU_AVX";
    }

    static void decodeGroup(const uint8_t *src, uint32_t * &dst) {
    	uint8_t desc = src[0];
    	src += kHeaderSize;
    	const uint8_t num = varintTables::varintG8IUOutputOffsets[desc]; 

        const __m128i data = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src));
        const __m256i val = _mm256_broadcastsi128_si256(data);
//        const __m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(data), data, 1); // Slower alternative.
        __m256i result = _mm256_shuffle_epi8(val, varintTables::varintGU_AVX2_shfl_msk_m256i[0][desc]);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), result);

        dst += num;
    }
};
#endif /* __AVX2__ */

} // namespace Codecs

#endif // CODECS_VARINTG8IU_H_
