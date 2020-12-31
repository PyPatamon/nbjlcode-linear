/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Jan 18, 2015
 */
/**
 * Based on code by
 *     Facebook, https://github.com/facebook/folly
 * which was available under the Apache License, Version 2.0.
 */

#ifndef CODECS_VARINTGB_H_
#define CODECS_VARINTGB_H_ 

#if !defined(__GNUC__) && !defined(_MSC_VER)
#error varintG8IU.h requires GCC or MSVC.
#endif

#include "IntegerCodec.h"
#include "Portability.h"

#if !CODECS_X64 && !defined(__i386__) && !CODECS_PPC64
#error varintG8IU.h requires x86[_64].
#endif


#if CODECS_AVX_PREREQ(2, 0) || CODECS_SSE_PREREQ(3, 1)
namespace Codecs {
namespace varintTables {
extern const __m128i varintGB_SSSE3_shfl_msk_m128i[256];
} // namespace varintTables
} // namespace Codecs
#endif /* __AVX2__ || __SSSE3__ */

namespace Codecs {
namespace varintTables {
extern const uint8_t varintGBInputOffsets[256];
extern const uint8_t varintGBLengths[256][4];
extern const uint32_t kMask[4];
} // namespace varintTables
} // namespace Codecs


namespace Codecs {

struct G4B {
    enum {
        kHeaderSize = 1, // Number of bytes for the descriptor.
        kGroupSize = 4   // Number of encoded integers in a group.
    };
};

struct G8B {
    enum {
        kHeaderSize = 2, // Number of bytes for the descriptor.
        kGroupSize = 8   // Number of encoded integers in a groups.
    };
};


template<typename InstructionSet>
class varintG4B;

using varintG4B_Scalar = varintG4B<Scalar>;

#if CODECS_SSE_PREREQ(3, 1)
using varintG4B_SSE = varintG4B<SSE>;
#endif /* __SSSE3__ */

#if CODECS_AVX_PREREQ(2, 0)
using varintG4B_AVX = varintG4B<AVX>;
#endif /* __AVX2__ */


template<typename InstructionSet>
class varintG8B;

using varintG8B_Scalar = varintG8B<Scalar>;

#if CODECS_SSE_PREREQ(3, 1)
using varintG8B_SSE = varintG8B<SSE>;
#endif /* __SSSE3__ */

#if CODECS_AVX_PREREQ(2, 0)
using varintG8B_AVX = varintG8B<AVX>;
#endif /* __AVX2__ */


/**
 * varintGB encoding for 32-bit integers.
 *
 * Encodes kGroupSize 32-bit integers at once, each using 1-4 bytes depending on size.
 * There is kHeaderSize bytes of overhead.  (The first kHeaderSize bytes contain the
 * lengths of the kGroupSize integers encoded as two bits each; 00=1 byte .. 11=4 bytes)
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
template <typename Format>
class varintGBBase : public IntegerCodec {
public:
	/**
	 * Number of bytes for the descriptor.
	 */
	enum { kHeaderSize = Format::kHeaderSize };

    /**
     * Number of integers encoded / decoded in one pass.
     */
    enum { kGroupSize = Format::kGroupSize };

    /**
     * Maximum encoded size.
     */
    enum { kMaxSize = kHeaderSize + sizeof(uint32_t) * kGroupSize };

    /**
     * Maximum size for n values.
     */
    static uint64_t maxSize(uint64_t n) {
    	// Full groups
    	uint64_t total = (n / kGroupSize) * kFullGroupSize;

        n %= kGroupSize;
        // Incomplete last group, if any
        if (n) {
            total += kHeaderSize + n * sizeof(uint32_t);
        }
        return total;
    }


protected:
    static uint8_t key(uint32_t x) {
        // __builtin_clz is undefined for the x==0 case
        return 3 - (__builtin_clz(x|1) / 8);
    }

    static uint8_t b0key(uint8_t x) { return x & 3; }
    static uint8_t b1key(uint8_t x) { return (x >> 2) & 3; }
    static uint8_t b2key(uint8_t x) { return (x >> 4) & 3; }
    static uint8_t b3key(uint8_t x) { return (x >> 6) & 3; }

private:
    enum { kFullGroupSize = kHeaderSize + kGroupSize * sizeof(uint32_t) };
};


template <typename InstructionSet>
class varintG4BBase : public varintGBBase<G4B> {
public:
	varintG4BBase() : buf(Derived::kMaxDecodingCount) { }

	/**
	 * Return the number of bytes used to encode these four values.
	 */
	static uint64_t size(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
		return kHeaderSize + kGroupSize + key(a) + key(b) + key(c) + key(d);
	}

	/**
	 * Return the number of bytes used to encode four uint32_t values stored
	 * at consecutive positions in an array.
	 */
	static uint64_t size(const uint32_t *p) {
		return size(p[0], p[1], p[2], p[3]);
	}

	/**
	 * Return the number of bytes used to encode count (<= 4) values.
	 * If you clip a buffer after these many bytes, you can still decode
	 * the first "count" values correctly (if the remaining size() -
	 * partialSize() bytes are filled with garbage).
	 */
    static uint64_t partialSize(const uint32_t *p, uint64_t count) {
        assert(count <= kGroupSize);
        uint64_t s = kHeaderSize + count;
        for ( ; count; --count, ++p) {
        	s += key(*p);
        }
        return s;
    }

    /**
     * Size of n values starting at p.
     */
    static uint64_t totalSize(const uint32_t *p, uint64_t n) {
        uint64_t s = 0;
        for ( ; n >= kGroupSize; n -= kGroupSize, p += kGroupSize) {
            s += size(p);
        }
        if (n) {
            s += partialSize(p, n);
        }
        return s;
    }

    /**
     * Return the number of values from *p that are valid from an encoded
     * buffer of size bytes.
     */
    static uint64_t partialCount(const uint8_t *p, uint64_t size) {
        uint8_t v = *p;
        uint64_t s = kHeaderSize;
        s += 1 + b0key(v);
        if (s > size) return 0;
        s += 1 + b1key(v);
        if (s > size) return 1;
        s += 1 + b2key(v);
        if (s > size) return 2;
        s += 1 + b3key(v);
        if (s > size) return 3;
        return 4;
    }

    /**
     * Given a pointer to the beginning of an varintG4B-encoded block,
     * return the number of bytes used by the encoding.
     */
    static uint64_t encodedSize(const uint8_t *p) {
    	return (kHeaderSize + kGroupSize +
    			b0key(*p) + b1key(*p) + b2key(*p) + b3key(*p));
    }

    /**
     * Encode a group of four uint32_t values into the buffer pointed-to by p, 
     * and return the next position in the buffer (that is, one character past 
     * the last encoded byte).  p needs to have at least size() bytes available.
     */
    static uint8_t * encodeGroup(uint8_t *dst, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    	uint8_t k0 = key(a);
    	uint8_t k1 = key(b);
    	uint8_t k2 = key(c);
    	uint8_t k3 = key(d);
    	*dst++ = (k3 << 6) | (k2 << 4) | (k1 << 2) | k0;
    	*reinterpret_cast<uint32_t *>(dst) = a;
    	dst += k0+1;
    	*reinterpret_cast<uint32_t *>(dst) = b;
    	dst += k1+1;
    	*reinterpret_cast<uint32_t *>(dst) = c;
    	dst += k2+1;
    	*reinterpret_cast<uint32_t *>(dst) = d;
    	dst += k3+1;
    	return dst;
    }

    /**
     * Encode a group of four uint32_t values from the array pointed-to by src
     * into the buffer pointed-to by dst, similar to encode(dst,a,b,c,d) above.
     */
    static uint8_t * encodeGroup(uint8_t *dst, const uint32_t *src) {
    	return encodeGroup(dst, src[0], src[1], src[2], src[3]);
    }

    /**
     * Encode into the buffer pointed-to by dst with count (< 4) uint32_t values
     * from the array pointed-to by src and (4 - count) 0s.
     */
    static uint8_t * encodeGroup(uint8_t *dst, const uint32_t *src, uint64_t count) {
    	assert(count < kGroupSize);

        switch (count) {
        case 3: return encodeGroup(dst, src[0], src[1], src[2], 0);
        case 2: return encodeGroup(dst, src[0], src[1], 0, 0);
        case 1: return encodeGroup(dst, src[0], 0, 0, 0);
        default: return dst;
        }
    }

    /**
     * Encode nvalue values from the array pointed-to by in into the buffer pointed-to
     * by out, and return the number of 32-bit words consumed (that is, there might be 
     * some unused bytes in the end).
     */
    virtual void encodeArray(const uint32_t *in, uint64_t nvalue,
            uint32_t *out, uint64_t &csize) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out);
        const uint8_t *const initdst = dst;

        // Full groups.
        for ( ; nvalue >= kGroupSize; nvalue -= kGroupSize, in += kGroupSize) {
            dst = encodeGroup(dst, in);
        }
        // Incomplete last group, if any.
        if (nvalue) {
            dst = encodeGroup(dst, in, nvalue);
        }

        // Align to 32-bit word boundary.
        csize = ((dst - initdst) + 3) / 4;
    }

    /**
     * decode nvalue values from the buffer pointed-to to by in
     * into the array pointed-to by out. 
     */
    virtual const uint32_t * decodeArray(const uint32_t *in, uint64_t csize,
            uint32_t *out, uint64_t nvalue) {
        const uint8_t *src = reinterpret_cast<const uint8_t *>(in);
        const uint8_t *const initsrc = src;
        const uint32_t *const endout = out + nvalue;

        while (endout - out >= Derived::kMaxDecodingCount) {
            Derived::decodeGroup(src, out);
        }

        uint32_t valuesRemaining = endout - out;
        uint32_t *ptr = buf.data();
        const uint32_t *const endptr = ptr + valuesRemaining;
        while (endptr > ptr) {
        	Derived::decodeGroup(src, ptr);
        }
        memcpy(out, buf.data(), sizeof(uint32_t) * valuesRemaining);

        csize = ((src - initsrc) + 3) / 4;
        return in + csize;
    }

    virtual std::string name() const {
        return Derived::codecname();
    }

protected:
    std::vector<uint32_t> buf;

private:
    using Derived = varintG4B<InstructionSet>;
};


template <>
class varintG4B<Scalar> : public varintG4BBase<Scalar> {
public:
	enum { kMaxDecodingCount = kGroupSize };

    static const std::string codecname() {
        return "varintG4B_Scalar";
    }

    static void decodeGroup(const uint8_t * &src, uint32_t * &dst) {
        src = decodeGroup(src, dst, dst + 1, dst + 2, dst + 3);
        dst += kGroupSize;
    }

    /**
     * Decode four uint32_t values from a buffer, and return the next position
     * in the buffer (that is, one character past the last encoded byte).
     * The buffer needs to have at least 3 extra bytes available (they
     * may be read but ignored).
     */
    static const uint8_t * decodeGroup(const uint8_t *src, uint32_t *a, uint32_t *b,
                                                           uint32_t *c, uint32_t *d) {
    	uint8_t k = *reinterpret_cast<const uint8_t*>(src);
    	const uint8_t *end = src + varintTables::varintGBInputOffsets[k];
    	++src;

    	uint8_t k0 = b0key(k);
    	*a = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k0];
    	src += k0 + 1;
    	uint8_t k1 = b1key(k);
    	*b = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k1];
    	src += k1 + 1;
    	uint8_t k2 = b2key(k);
    	*c = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k2];
    	src += k2 + 1;
    	uint8_t k3 = b3key(k);
    	*d = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k3];
    	src += k3 + 1;

    	return end;
    }
};


#if CODECS_SSE_PREREQ(3, 1)
template <>
class varintG4B<SSE> : public varintG4BBase<SSE> {
public:
	enum { kMaxDecodingCount = kGroupSize };

    static const std::string codecname() {
        return "varintG4B_SSE";
    }

    static void decodeGroup(const uint8_t * &src, uint32_t * &dst) {
        uint8_t k = src[0];
        __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src + 1));
        __m128i mask = varintTables::varintGB_SSSE3_shfl_msk_m128i[k];
        __m128i result = _mm_shuffle_epi8(val, mask);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), result);

        src += varintTables::varintGBInputOffsets[k];
        dst += kGroupSize;
    }

#if CODECS_SSE_PREREQ(3, 1)
    static const uint8_t * decode(const uint8_t *src, uint32_t *a, uint32_t *b,
    		                                          uint32_t *c, uint32_t *d) {
        uint8_t k = src[0];
        __m128i val = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src + 1));
        __m128i mask = varintTables::varintGB_SSSE3_shfl_msk_m128i[k];
        __m128i result = _mm_shuffle_epi8(val, mask);

        // Extracting 32 bits at a time out of an XMM register is a SSE4 feature
#if CODECS_SSE_PREREQ(4, 1)
        *a = _mm_extract_epi32(result, 0);
        *b = _mm_extract_epi32(result, 1);
        *c = _mm_extract_epi32(result, 2);
        *d = _mm_extract_epi32(result, 3);
#else  /* !__SSE4_1__ */
        *a = _mm_extract_epi16(result, 0) + (_mm_extract_epi16(result, 1) << 16);
        *b = _mm_extract_epi16(result, 2) + (_mm_extract_epi16(result, 3) << 16);
        *c = _mm_extract_epi16(result, 4) + (_mm_extract_epi16(result, 5) << 16);
        *d = _mm_extract_epi16(result, 6) + (_mm_extract_epi16(result, 7) << 16);
#endif  /* __SSE4_1__ */
        return src + varintTables::varintGBInputOffsets[k];
    }
#endif  /* __SSSE3 */
};
#endif /* __SSSE3__ */


#if CODECS_AVX_PREREQ(2, 0)
template <>
class varintG4B<AVX> : public varintG4BBase<AVX> {
public:
	enum { kMaxDecodingCount = 2 * kGroupSize };

    static const std::string codecname() {
        return "varintG4B_AVX";
    }

    static void decodeGroup(const uint8_t * &src, uint32_t * &dst) {
        uint8_t k0 = src[0];
        __m128i val0 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src + 1));
        uint8_t k1 = src[varintTables::varintGBInputOffsets[k0]];
        __m128i val1 = _mm_lddqu_si128(reinterpret_cast<const __m128i *>(src + 1 + varintTables::varintGBInputOffsets[k0]));
        __m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(val0), val1, 1);

        __m128i mask0 = varintTables::varintGB_SSSE3_shfl_msk_m128i[k0];
        __m128i mask1 = varintTables::varintGB_SSSE3_shfl_msk_m128i[k1];
        __m256i mask = _mm256_inserti128_si256(_mm256_castsi128_si256(mask0), mask1, 1);

        __m256i result = _mm256_shuffle_epi8(val, mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), result);

        src += varintTables::varintGBInputOffsets[k0] + varintTables::varintGBInputOffsets[k1];
        dst += 2 * kGroupSize;
    }

    /**
     * decode nvalue values from the buffer pointed-to to by in
     * into the array pointed-to by out.
     */
    virtual const uint32_t * decodeArray(const uint32_t *in, uint64_t csize,
            uint32_t *out, uint64_t nvalue) {
        const uint8_t *src = reinterpret_cast<const uint8_t *>(in);
        const uint8_t *const initsrc = src;
        const uint32_t *const endout = out + nvalue;

        while (endout - out >= kMaxDecodingCount) {
            decodeGroup(src, out);
        }

        uint32_t valuesRemaining = endout - out;
        uint32_t *ptr = buf.data();
        const uint32_t *const endptr = ptr + valuesRemaining;
        while (endptr > ptr) {
        	varintG4B<SSE>::decodeGroup(src, ptr);
        }
        memcpy(out, buf.data(), sizeof(uint32_t) * valuesRemaining);

        csize = ((src - initsrc) + 3) / 4;
        return in + csize;
    }
};
#endif /* __AVX2__ */


template <typename InstructionSet>
class varintG8BBase : public varintGBBase<G8B> {
public:
	/**
	 * Return the number of bytes used to encode these eight values.
	 */
    static uint64_t size(uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                         uint32_t e, uint32_t f, uint32_t g, uint32_t h) {
        return kHeaderSize + kGroupSize + key(a) + key(b) + key(c) + key(d)
                                        + key(e) + key(f) + key(g) + key(h);
    }

    /**
     * Return the number of bytes used to encode eight uint32_t values stored
     * at consecutive positions in an array.
     */
    static uint64_t size(const uint32_t *p) {
        return size(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    }

    /**
     * Return the number of bytes used to encode count (<= 8) values.
     * If you clip a buffer after these many bytes, you can still decode
     * the first "count" values correctly (if the remaining size() -
     * partialSize() bytes are filled with garbage).
     */
    static uint64_t partialSize(const uint32_t *p, uint64_t count) {
        assert(count <= kGroupSize);
        uint64_t s = kHeaderSize + count;
        for ( ; count; --count, ++p) {
        	s += key(*p);
        }
        return s;
    }

    /**
     * Size of n values starting at p.
     */
    static uint64_t totalSize(const uint32_t *p, uint64_t n) {
        uint64_t s = 0;
        for ( ; n >= kGroupSize; n -= kGroupSize, p += kGroupSize) {
            s += size(p);
        }
        if (n) {
            s += partialSize(p, n);
        }
        return s;
    }

    /**
     * Return the number of values from *p that are valid from an encoded
     * buffer of size bytes.
     */
    static uint64_t partialCount(const uint8_t *p, uint64_t size) {
        uint64_t s = kHeaderSize;
        s += 1 + b0key(p[0]);
        if (s > size) return 0;
        s += 1 + b1key(p[0]);
        if (s > size) return 1;
        s += 1 + b2key(p[0]);
        if (s > size) return 2;
        s += 1 + b3key(p[0]);
        if (s > size) return 3;
        s += 1 + b0key(p[1]);
        if (s > size) return 4;
        s += 1 + b1key(p[1]);
        if (s > size) return 5;
        s += 1 + b2key(p[1]);
        if (s > size) return 6;
        s += 1 + b3key(p[1]);
        if (s > size) return 7;
        return 8;
    }

    /**
     * Given a pointer to the beginning of an varintG8B-encoded block,
     * return the number of bytes used by the encoding.
     */
    static uint64_t encodedSize(const uint8_t *p) {
        return (kHeaderSize + kGroupSize +
                b0key(p[0]) + b1key(p[0]) + b2key(p[0]) + b3key(p[0]) +
                b0key(p[1]) + b1key(p[1]) + b2key(p[1]) + b3key(p[1]));
    }

    /**
     * Encode a group of eight uint32_t values into the buffer pointed-to by dst,
     * and return the next position in the buffer (that is, one character past
     * the last encoded byte).  dst needs to have at least size() bytes available.
     */
    static uint8_t * encodeGroup(uint8_t *dst, uint32_t a, uint32_t b, uint32_t c, uint32_t d,
                                          uint32_t e, uint32_t f, uint32_t g, uint32_t h) {
        uint8_t k00 = key(a);
        uint8_t k01 = key(b);
        uint8_t k02 = key(c);
        uint8_t k03 = key(d);
        *reinterpret_cast<uint8_t *>(dst++) = (k03 << 6) | (k02 << 4) | (k01 << 2) | k00;

        uint8_t k10 = key(e);
        uint8_t k11 = key(f);
        uint8_t k12 = key(g);
        uint8_t k13 = key(h);
        *reinterpret_cast<uint8_t *>(dst++) = (k13 << 6) | (k12 << 4) | (k11 << 2) | k10;

        *reinterpret_cast<uint32_t *>(dst) = a;
        dst += k00 + 1;
        *reinterpret_cast<uint32_t *>(dst) = b;
        dst += k01 + 1;
        *reinterpret_cast<uint32_t *>(dst) = c;
        dst += k02 + 1;
        *reinterpret_cast<uint32_t *>(dst) = d;
        dst += k03 + 1;

        *reinterpret_cast<uint32_t *>(dst) = e;
        dst += k10 + 1;
        *reinterpret_cast<uint32_t *>(dst) = f;
        dst += k11 + 1;
        *reinterpret_cast<uint32_t *>(dst) = g;
        dst += k12 + 1;
        *reinterpret_cast<uint32_t *>(dst) = h;
        dst += k13 + 1;
        return dst;
    }

    /**
     * Encode a group of eight uint32_t values from the array pointed-to by src into 
     * the buffer pointed-to by dst, similar to encode(dst,a,b,c,d,e,f,g,h) above.
     */
    static uint8_t * encodeGroup(uint8_t *dst, const uint32_t *src) {
        return encodeGroup(dst, src[0], src[1], src[2], src[3],
                                src[4], src[5], src[6], src[7]);
    }

    /**
     * Encode into the buffer pointed-to by dst with count (< 8) uint32_t values
     * from the array pointed-to by src and (8 - count) 0s.
     */
    static uint8_t * encodeGroup(uint8_t *dst, const uint32_t *src, uint64_t count) {
    	assert(count < kGroupSize);

        switch (count) {
        case 7: return encodeGroup(dst, src[0], src[1], src[2], src[3], src[4], src[5], src[6], 0);
        case 6: return encodeGroup(dst, src[0], src[1], src[2], src[3], src[4], src[5], 0, 0);
        case 5: return encodeGroup(dst, src[0], src[1], src[2], src[3], src[4], 0, 0, 0);
        case 4: return encodeGroup(dst, src[0], src[1], src[2], src[3], 0, 0, 0, 0);
        case 3: return encodeGroup(dst, src[0], src[1], src[2], 0, 0, 0, 0, 0);
        case 2: return encodeGroup(dst, src[0], src[1], 0, 0, 0, 0, 0, 0);
        case 1: return encodeGroup(dst, src[0], 0, 0, 0, 0, 0, 0, 0);
        default: return dst;
        }
    }

    virtual void encodeArray(const uint32_t *in, uint64_t nvalue,
            uint32_t *out, uint64_t &csize) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out);
        const uint8_t *const initdst = dst;

        // Full groups.
        for ( ; nvalue >= kGroupSize; nvalue -= kGroupSize, in += kGroupSize) {
        	dst = encodeGroup(dst, in);
        }
        // Incomplete last group, if any.
        if (nvalue) {
        	dst = encodeGroup(dst, in, nvalue);
        }

        // Align to 32-bit word boundary.
        csize = ((dst - initdst) + 3) / 4;
    }

    virtual const uint32_t * decodeArray(const uint32_t *in, uint64_t csize,
            uint32_t *out, uint64_t nvalue) {
        const uint32_t *const endout = out + nvalue;
        const uint8_t *src = reinterpret_cast<const uint8_t *>(in);
        const uint8_t *const initsrc = src;

        while (endout - out >= kGroupSize) {
            Derived::decodeGroup(src, out);
        }

        uint32_t valuesRemaining = endout - out;
        uint32_t *ptr = buf;
        const uint32_t *const endptr = buf + valuesRemaining;
        while (endptr > ptr) {
        	Derived::decodeGroup(src, ptr);
        }
        memcpy(out, buf, sizeof(uint32_t) * valuesRemaining);

        csize = ((src - initsrc) + 3) / 4;
        return in + csize;
    }

    virtual std::string name() const {
        return Derived::codecname();
    }

private:
    using Derived = varintG8B<InstructionSet>;

    uint32_t buf[kGroupSize];
};


template <>
class varintG8B<Scalar> : public varintG8BBase<Scalar> {
public:
    static const std::string codecname() {
        return "varintG8B_Scalar";
    }

    static void decodeGroup(const uint8_t * &src, uint32_t * &dst) {
        src = decodeGroup(src, dst, dst + 1, dst + 2, dst + 3,
                           dst + 4, dst + 5, dst + 6, dst + 7);
        dst += kGroupSize;
    }

    /**
     * Decode eight uint32_t values from a buffer, and return the next position
     * in the buffer (that is, one character past the last encoded byte).
     * The buffer needs to have at least 3 extra bytes available (they
     * may be read but ignored).
     */
    static const uint8_t * decodeGroup(const uint8_t *src, uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d,
    		                                               uint32_t *e, uint32_t *f, uint32_t *g, uint32_t *h) {
        uint16_t k = *reinterpret_cast<const uint16_t*>(src);
        uint8_t k0 = k & 0xff, k1 = k >> 8;
//        // Slower alternative.
//        uint8_t k0  = *reinterpret_cast<const uint8_t*>(src);
//        uint8_t k1 = *reinterpret_cast<const uint8_t*>(src + 1);

        const uint8_t *end = src + varintTables::varintGBInputOffsets[k0] + varintTables::varintGBInputOffsets[k1];
        src += kHeaderSize;

        uint64_t k00 = b0key(k0);
        *a = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k00];
        src += k00 + 1;
        uint64_t k01 = b1key(k0);
        *b = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k01];
        src += k01 + 1;
        uint64_t k02 = b2key(k0);
        *c = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k02];
        src += k02 + 1;
        uint64_t k03 = b3key(k0);
        *d = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k03];
        src += k03 + 1;

        uint64_t k10 = b0key(k1);
        *e = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k10];
        src += k10 + 1;
        uint64_t k11 = b1key(k1);
        *f = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k11];
        src += k11 + 1;
        uint64_t k12 = b2key(k1);
        *g = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k12];
        src += k12 + 1;
        uint64_t k13 = b3key(k1);
        *h = *reinterpret_cast<const uint32_t *>(src) & varintTables::kMask[k13];
        src += k13 + 1;

        return end;
    }
};


#if CODECS_SSE_PREREQ(3, 1)
template <>
class varintG8B<SSE> : public varintG8BBase<SSE> {
public:
    static const std::string codecname() {
        return "varintG8B_SSE";
    }

    static void decodeGroup(const uint8_t * &src, uint32_t * &dst) {
        uint16_t k = *reinterpret_cast<const uint16_t *>(src);
        uint8_t k0 = k & 0xff, k1 = k >> 8;
//        uint8_t k0 = src[0], k1 = src[1]; // Slower alternative.

        const __m128i val0 = _mm_lddqu_si128((const __m128i *)(src + 2));
        const __m128i val1 = _mm_lddqu_si128((const __m128i *)(src + 1 + varintTables::varintGBInputOffsets[k0]));
        const __m128i mask0 = varintTables::varintGB_SSSE3_shfl_msk_m128i[k0];
        const __m128i mask1 = varintTables::varintGB_SSSE3_shfl_msk_m128i[k1];

        __m128i result0 = _mm_shuffle_epi8(val0, mask0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), result0);
        __m128i result1 = _mm_shuffle_epi8(val1, mask1);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 4), result1);

        src += varintTables::varintGBInputOffsets[k0] + varintTables::varintGBInputOffsets[k1];
        dst += kGroupSize;
    }
};
#endif /* __SSSE3__ */


#if CODECS_AVX_PREREQ(2, 0)
template <>
class varintG8B<AVX> : public varintG8BBase<AVX> {
public:
    static const std::string codecname() {
        return "varintG8B_AVX";
    }

    static void decodeGroup(const uint8_t * &src, uint32_t * &dst) {
        uint16_t k = *reinterpret_cast<const uint16_t *>(src);
        uint8_t k0 = k & 0xff, k1 = k >> 8;
//        uint8_t k0 = src[0], k1 = src[1]; // Slower alternative.

        const __m128i val0 = _mm_lddqu_si128((const __m128i *)(src + 2));
        const __m128i val1 = _mm_lddqu_si128((const __m128i *)(src + 1 + varintTables::varintGBInputOffsets[k0]));
        __m256i val = _mm256_inserti128_si256(_mm256_castsi128_si256(val0), val1, 1);
        const __m128i mask0 = varintTables::varintGB_SSSE3_shfl_msk_m128i[k0];
        const __m128i mask1 = varintTables::varintGB_SSSE3_shfl_msk_m128i[k1];
        // Note that _mm256_loadu2_m128i isn't supported by gcc 4.8.3; use _mm256_inserti128_si256 instead.
        __m256i mask = _mm256_inserti128_si256(_mm256_castsi128_si256(mask0), mask1, 1);

        __m256i result = _mm256_shuffle_epi8(val, mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), result);

        src += varintTables::varintGBInputOffsets[k0] + varintTables::varintGBInputOffsets[k1];
        dst += kGroupSize;
    }
};
#endif /* __AVX2__ */


} // namespace Codecs


#endif // CODECS_VARINTGB_H_ 
