/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Mar 16, 2016
 */


#ifndef CODECS_HORUNPACKER_H_
#define CODECS_HORUNPACKER_H_

#if !defined(__GNUC__) && !defined(_MSC_VER)
#error HorUnpacker.h requires GCC or MSVC
#endif

#include "Portability.h"
#include "util.h"

#if CODECS_SSE_PREREQ(4, 1)
namespace Codecs {
namespace SIMDMasks {
extern const __m128i Hor_SSE4_mul_msk_m128i[33][2];
extern const int Hor_SSE4_srli_imm_int[33][2];
extern const __m128i SSE2_and_msk_m128i[33];
} // namespace SIMDMasks
} // namespace Codecs
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
namespace Codecs {
namespace SIMDMasks {
extern const __m256i Hor_AVX2_shfl_msk_m256i[33];
extern const __m256i Hor_AVX2_srlv_msk_m256i[33];
extern const __m256i AVX2_and_msk_m256i[33];
} // namespace SIMDMasks
} // namespace Codecs
#endif /* __AVX2__ */

namespace Codecs {

template <typename InstructionSet, bool IsRiceCoding>
class HorUnpacker;

template <bool IsRiceCoding>
using HorScalarUnpacker = HorUnpacker<Scalar, IsRiceCoding>;

#if CODECS_SSE_PREREQ(4, 1)
template <bool IsRiceCoding>
using HorSSEUnpacker = HorUnpacker<SSE, IsRiceCoding>;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
template <bool IsRiceCoding>
using HorAVXUnpacker = HorUnpacker<AVX, IsRiceCoding>;
#endif /* __AVX2__ */

template <typename InstructionSet, bool IsRiceCoding>
class HorUnpackerBase {
public:
	enum { BASEPACKSIZE = 64 };

	using datatype = typename InstructionSet::datatype;

	HorUnpackerBase() {
		checkifdivisibleby(Derived::PACKSIZE, BASEPACKSIZE);
	}

	/* assumes that integers fit in the prescribed number of bits */
	static void packwithoutmask(datatype *  __restrict__  outPtr,
			const datatype *  __restrict__  inPtr, uint32_t b) {
		uint32_t *out = reinterpret_cast<uint32_t *>(outPtr);
		const uint32_t *in = reinterpret_cast<const uint32_t *>(inPtr);
		for (uint32_t valuesPacked = 0; valuesPacked < Derived::PACKSIZE; valuesPacked += BASEPACKSIZE) {
			// Could have used function pointers instead of switch.
			// Switch calls do offer the compiler more opportunities for
			// optimization in theory. In this case, it makes no difference
			// with a good compiler.
			switch(b) {
			case 0: break;
			case 1: horizontalpackwithoutmask_c1(out, in); break;
			case 2: horizontalpackwithoutmask_c2(out, in); break;
			case 3: horizontalpackwithoutmask_c3(out, in); break;
			case 4: horizontalpackwithoutmask_c4(out, in); break;
			case 5: horizontalpackwithoutmask_c5(out, in); break;
			case 6: horizontalpackwithoutmask_c6(out, in); break;
			case 7: horizontalpackwithoutmask_c7(out, in); break;
			case 8: horizontalpackwithoutmask_c8(out, in); break;
			case 9: horizontalpackwithoutmask_c9(out, in); break;
			case 10: horizontalpackwithoutmask_c10(out, in); break;
			case 11: horizontalpackwithoutmask_c11(out, in); break;
			case 12: horizontalpackwithoutmask_c12(out, in); break;
			case 13: horizontalpackwithoutmask_c13(out, in); break;
			case 14: horizontalpackwithoutmask_c14(out, in); break;
			case 15: horizontalpackwithoutmask_c15(out, in); break;
			case 16: horizontalpackwithoutmask_c16(out, in); break;
			case 17: horizontalpackwithoutmask_c17(out, in); break;
			case 18: horizontalpackwithoutmask_c18(out, in); break;
			case 19: horizontalpackwithoutmask_c19(out, in); break;
			case 20: horizontalpackwithoutmask_c20(out, in); break;
			case 21: horizontalpackwithoutmask_c21(out, in); break;
			case 22: horizontalpackwithoutmask_c22(out, in); break;
			case 23: horizontalpackwithoutmask_c23(out, in); break;
			case 24: horizontalpackwithoutmask_c24(out, in); break;
			case 25: horizontalpackwithoutmask_c25(out, in); break;
			case 26: horizontalpackwithoutmask_c26(out, in); break;
			case 27: horizontalpackwithoutmask_c27(out, in); break;
			case 28: horizontalpackwithoutmask_c28(out, in); break;
			case 29: horizontalpackwithoutmask_c29(out, in); break;
			case 30: horizontalpackwithoutmask_c30(out, in); break;
			case 31: horizontalpackwithoutmask_c31(out, in); break;
			case 32: horizontalpackwithoutmask_c32(out, in); break;
			default: break;
			}

            in += BASEPACKSIZE;
            out += (BASEPACKSIZE * b) / 32;
		}
	}

	static void pack(uint32_t *  __restrict__  out,
			const uint32_t *  __restrict__  in, uint32_t b) {
		for (uint32_t valuesPacked = 0; valuesPacked < Derived::PACKSIZE; valuesPacked += BASEPACKSIZE) {
			// Could have used function pointers instead of switch.
			// Switch calls do offer the compiler more opportunities
			// for optimization in theory. In this case, it makes no
			// difference with a good compiler.
			switch(b) {
			case 0: break;
			case 1: horizontalpack_c1(out, in); break;
			case 2: horizontalpack_c2(out, in); break;
			case 3: horizontalpack_c3(out, in); break;
			case 4: horizontalpack_c4(out, in); break;
			case 5: horizontalpack_c5(out, in); break;
			case 6: horizontalpack_c6(out, in); break;
			case 7: horizontalpack_c7(out, in); break;
			case 8: horizontalpack_c8(out, in); break;
			case 9: horizontalpack_c9(out, in); break;
			case 10: horizontalpack_c10(out, in); break;
			case 11: horizontalpack_c11(out, in); break;
			case 12: horizontalpack_c12(out, in); break;
			case 13: horizontalpack_c13(out, in); break;
			case 14: horizontalpack_c14(out, in); break;
			case 15: horizontalpack_c15(out, in); break;
			case 16: horizontalpack_c16(out, in); break;
			case 17: horizontalpack_c17(out, in); break;
			case 18: horizontalpack_c18(out, in); break;
			case 19: horizontalpack_c19(out, in); break;
			case 20: horizontalpack_c20(out, in); break;
			case 21: horizontalpack_c21(out, in); break;
			case 22: horizontalpack_c22(out, in); break;
			case 23: horizontalpack_c23(out, in); break;
			case 24: horizontalpack_c24(out, in); break;
			case 25: horizontalpack_c25(out, in); break;
			case 26: horizontalpack_c26(out, in); break;
			case 27: horizontalpack_c27(out, in); break;
			case 28: horizontalpack_c28(out, in); break;
			case 29: horizontalpack_c29(out, in); break;
			case 30: horizontalpack_c30(out, in); break;
			case 31: horizontalpack_c31(out, in); break;
			case 32: horizontalpack_c32(out, in); break;
			default: break;
			}

            in += BASEPACKSIZE;
            out += (BASEPACKSIZE * b) / 32;
		}
	}


	using packer = void (uint32_t *  __restrict__  out, const uint32_t *  __restrict__  in);
	static packer horizontalpackwithoutmask_c1;  static packer horizontalpackwithoutmask_c2;
	static packer horizontalpackwithoutmask_c3;  static packer horizontalpackwithoutmask_c4;
	static packer horizontalpackwithoutmask_c5;  static packer horizontalpackwithoutmask_c6;
	static packer horizontalpackwithoutmask_c7;  static packer horizontalpackwithoutmask_c8;
	static packer horizontalpackwithoutmask_c9;  static packer horizontalpackwithoutmask_c10;
	static packer horizontalpackwithoutmask_c11; static packer horizontalpackwithoutmask_c12;
	static packer horizontalpackwithoutmask_c13; static packer horizontalpackwithoutmask_c14;
	static packer horizontalpackwithoutmask_c15; static packer horizontalpackwithoutmask_c16;
	static packer horizontalpackwithoutmask_c17; static packer horizontalpackwithoutmask_c18;
	static packer horizontalpackwithoutmask_c19; static packer horizontalpackwithoutmask_c20;
	static packer horizontalpackwithoutmask_c21; static packer horizontalpackwithoutmask_c22;
	static packer horizontalpackwithoutmask_c23; static packer horizontalpackwithoutmask_c24;
	static packer horizontalpackwithoutmask_c25; static packer horizontalpackwithoutmask_c26;
	static packer horizontalpackwithoutmask_c27; static packer horizontalpackwithoutmask_c28;
	static packer horizontalpackwithoutmask_c29; static packer horizontalpackwithoutmask_c30;
	static packer horizontalpackwithoutmask_c31; static packer horizontalpackwithoutmask_c32;
    static packer horizontalpack_c1;  static packer horizontalpack_c2;
	static packer horizontalpack_c3;  static packer horizontalpack_c4;
	static packer horizontalpack_c5;  static packer horizontalpack_c6;
	static packer horizontalpack_c7;  static packer horizontalpack_c8;
	static packer horizontalpack_c9;  static packer horizontalpack_c10;
	static packer horizontalpack_c11; static packer horizontalpack_c12;
	static packer horizontalpack_c13; static packer horizontalpack_c14;
	static packer horizontalpack_c15; static packer horizontalpack_c16;
	static packer horizontalpack_c17; static packer horizontalpack_c18;
	static packer horizontalpack_c19; static packer horizontalpack_c20;
	static packer horizontalpack_c21; static packer horizontalpack_c22;
	static packer horizontalpack_c23; static packer horizontalpack_c24;
	static packer horizontalpack_c25; static packer horizontalpack_c26;
	static packer horizontalpack_c27; static packer horizontalpack_c28;
	static packer horizontalpack_c29; static packer horizontalpack_c30;
	static packer horizontalpack_c31; static packer horizontalpack_c32;


	/* assumes that integers fit in the prescribed number of bits */
	static void packwithoutmask_generic(uint32_t *  __restrict__  out,
			const uint32_t *  __restrict__  in, uint32_t b, uint32_t nvalue) {
		uint32_t nwords = div_roundup(nvalue * b, 32);
		memset(out, 0, nwords * sizeof(uint32_t));
		for (uint32_t valuesPacked = 0; valuesPacked < nvalue; ++valuesPacked) {
			uint32_t idx = (valuesPacked * b) >> 5;
			uint32_t shift = (valuesPacked * b) & 0x1f;
			uint64_t &codeword = (reinterpret_cast<uint64_t *>(out + idx))[0];
			codeword |= static_cast<uint64_t>(in[valuesPacked]) << shift;
		}
	}

	static void pack_generic(uint32_t *  __restrict__  out,
			const uint32_t *  __restrict__  in, uint32_t b, uint32_t nvalue) {
		uint64_t mask = (1ULL << b) - 1;
		uint32_t nwords = div_roundup(nvalue * b, 32);
		memset(out, 0, nwords * sizeof(uint32_t));
		for (uint32_t valuesPacked = 0; valuesPacked < nvalue; ++valuesPacked) {
			uint32_t idx = (valuesPacked * b) >> 5;
			uint32_t shift = (valuesPacked * b) & 0x1f;
			uint64_t &codeword = (reinterpret_cast<uint64_t *>(out + idx))[0];
			codeword |= static_cast<uint64_t>(in[valuesPacked] & mask) << shift;
		}
	}

	static void unpack_generic(uint32_t *  __restrict__  out,
			const uint32_t *  __restrict__  in, uint32_t b, uint32_t nvalue) {
		uint64_t mask = (1ULL << b) - 1;
		for (uint32_t valuesUnpacked = 0; valuesUnpacked < nvalue; ++valuesUnpacked) {
			uint32_t idx = (valuesUnpacked * b) >> 5;
			uint32_t shift = (valuesUnpacked * b) & 0x1f;
			const uint64_t codeword = (reinterpret_cast<const uint64_t *>(in + idx))[0];
			out[valuesUnpacked] = static_cast<uint32_t>((codeword >> shift) & mask);
		}
	}

private:
	using Derived = HorUnpacker<InstructionSet, IsRiceCoding>;
};


template <bool IsRiceCoding>
class HorUnpacker<Scalar, IsRiceCoding> : public HorUnpackerBase<Scalar, IsRiceCoding> {
	template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
	friend class Rice;
public:
	enum { PACKSIZE = 64 }; // number of elements to be unpacked for each invocation of unpack
	using datatype = Scalar::datatype;

	HorUnpacker(const uint32_t *q = nullptr): quotient(reinterpret_cast<const datatype *>(q)) {
		assert(PACKSIZE == 64);
	}

	~HorUnpacker() {
		quotient = nullptr;
	}

	static const std::string name() {
		std::ostringstream unpackername;
		unpackername << "HorScalarUnpacker<" << PACKSIZE << ">";
		return unpackername.str();
	}

	void unpack(datatype *  __restrict__  out,
			const datatype *  __restrict__  in, uint32_t b) {
	    // Could have used function pointers instead of switch.
	    // Switch calls do offer the compiler more opportunities
		// for optimization in theory. In this case, it makes no
		// difference with a good compiler.
	    switch(b) {
	    case 0: horizontalunpack_c0(out, in); return;
		case 1: horizontalunpack_c1(out, in); return;
		case 2: horizontalunpack_c2(out, in); return;
		case 3: horizontalunpack_c3(out, in); return;
		case 4: horizontalunpack_c4(out, in); return;
		case 5: horizontalunpack_c5(out, in); return;
		case 6: horizontalunpack_c6(out, in); return;
		case 7: horizontalunpack_c7(out, in); return;
		case 8: horizontalunpack_c8(out, in); return;
		case 9: horizontalunpack_c9(out, in); return;
		case 10: horizontalunpack_c10(out, in); return;
		case 11: horizontalunpack_c11(out, in); return;
		case 12: horizontalunpack_c12(out, in); return;
		case 13: horizontalunpack_c13(out, in); return;
		case 14: horizontalunpack_c14(out, in); return;
		case 15: horizontalunpack_c15(out, in); return;
		case 16: horizontalunpack_c16(out, in); return;
		case 17: horizontalunpack_c17(out, in); return;
		case 18: horizontalunpack_c18(out, in); return;
		case 19: horizontalunpack_c19(out, in); return;
		case 20: horizontalunpack_c20(out, in); return;
		case 21: horizontalunpack_c21(out, in); return;
		case 22: horizontalunpack_c22(out, in); return;
		case 23: horizontalunpack_c23(out, in); return;
		case 24: horizontalunpack_c24(out, in); return;
		case 25: horizontalunpack_c25(out, in); return;
		case 26: horizontalunpack_c26(out, in); return;
		case 27: horizontalunpack_c27(out, in); return;
		case 28: horizontalunpack_c28(out, in); return;
		case 29: horizontalunpack_c29(out, in); return;
		case 30: horizontalunpack_c30(out, in); return;
		case 31: horizontalunpack_c31(out, in); return;
		case 32: horizontalunpack_c32(out, in); return;
		default: return;
	    }
	}

	using unpacker = void (datatype *  __restrict__  out, const datatype *  __restrict__  in);
	unpacker horizontalunpack_c0;  unpacker horizontalunpack_c1;  unpacker horizontalunpack_c2;
	unpacker horizontalunpack_c3;  unpacker horizontalunpack_c4;  unpacker horizontalunpack_c5;
	unpacker horizontalunpack_c6;  unpacker horizontalunpack_c7;  unpacker horizontalunpack_c8;
	unpacker horizontalunpack_c9;  unpacker horizontalunpack_c10; unpacker horizontalunpack_c11;
	unpacker horizontalunpack_c12; unpacker horizontalunpack_c13; unpacker horizontalunpack_c14;
	unpacker horizontalunpack_c15; unpacker horizontalunpack_c16; unpacker horizontalunpack_c17;
	unpacker horizontalunpack_c18; unpacker horizontalunpack_c19; unpacker horizontalunpack_c20;
	unpacker horizontalunpack_c21; unpacker horizontalunpack_c22; unpacker horizontalunpack_c23;
	unpacker horizontalunpack_c24; unpacker horizontalunpack_c25; unpacker horizontalunpack_c26;
	unpacker horizontalunpack_c27; unpacker horizontalunpack_c28; unpacker horizontalunpack_c29;
	unpacker horizontalunpack_c30; unpacker horizontalunpack_c31; unpacker horizontalunpack_c32;

private:
	const datatype *quotient = nullptr;
};

#include "HorScalarUnpacker.h"


#if CODECS_SSE_PREREQ(4, 1)
/**
 * SSE4-based bit unpacking for horizontal layout.
 *
 * Follows
 *
 * T. Willhalm, N. Popovici, Y. Boshmaf, H. Plattner, A. Zeier, and J. Schaffner.
 * SIMD-scan: ultra fast in-memory table scan using on-chip vector processing units.
 * Proc. VLDB Endow., 2(1):385–394, 2009.
 */
template <bool IsRiceCoding>
class HorUnpacker<SSE, IsRiceCoding> : public HorUnpackerBase<SSE, IsRiceCoding> {
	template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
	friend class Rice;
public:
	enum { PACKSIZE = 128 }; // number of elements to be unpacked for each invocation of unpack
	using datatype = SSE::datatype;

	HorUnpacker(const uint32_t *q = nullptr): quotient(reinterpret_cast<const datatype *>(q)) {
		assert(PACKSIZE == 128);
	}

	~HorUnpacker() {
		quotient = nullptr;
	}

	static const std::string name() {
		std::ostringstream unpackername;
		unpackername << "HorSSEUnpacker<" << PACKSIZE << ">";
		return unpackername.str();
	}

	void unpack(datatype *  __restrict__  out,
			const datatype *  __restrict__  in, uint32_t b) {
	    // Could have used function pointers instead of switch.
	    // Switch calls do offer the compiler more opportunities
		// for optimization in theory. In this case, it makes no
		// difference with a good compiler.
	    switch(b) {
	    case 0: horizontalunpack_c0(out, in); return;
		case 1: horizontalunpack_c1(out, in); return;
		case 2: horizontalunpack_c2(out, in); return;
		case 3: horizontalunpack_c3(out, in); return;
		case 4: horizontalunpack_c4(out, in); return;
		case 5: horizontalunpack_c5(out, in); return;
		case 6: horizontalunpack_c6(out, in); return;
		case 7: horizontalunpack_c7(out, in); return;
		case 8: horizontalunpack_c8(out, in); return;
		case 9: horizontalunpack_c9(out, in); return;
		case 10: horizontalunpack_c10(out, in); return;
		case 11: horizontalunpack_c11(out, in); return;
		case 12: horizontalunpack_c12(out, in); return;
		case 13: horizontalunpack_c13(out, in); return;
		case 14: horizontalunpack_c14(out, in); return;
		case 15: horizontalunpack_c15(out, in); return;
		case 16: horizontalunpack_c16(out, in); return;
		case 17: horizontalunpack_c17(out, in); return;
		case 18: horizontalunpack_c18(out, in); return;
		case 19: horizontalunpack_c19(out, in); return;
		case 20: horizontalunpack_c20(out, in); return;
		case 21: horizontalunpack_c21(out, in); return;
		case 22: horizontalunpack_c22(out, in); return;
		case 23: horizontalunpack_c23(out, in); return;
		case 24: horizontalunpack_c24(out, in); return;
		case 25: horizontalunpack_c25(out, in); return;
		case 26: horizontalunpack_c26(out, in); return;
		case 27: horizontalunpack_c27(out, in); return;
		case 28: horizontalunpack_c28(out, in); return;
		case 29: horizontalunpack_c29(out, in); return;
		case 30: horizontalunpack_c30(out, in); return;
		case 31: horizontalunpack_c31(out, in); return;
		case 32: horizontalunpack_c32(out, in); return;
		default: return;
	    }
	}

	using unpacker = void (datatype *  __restrict__  out, const datatype *  __restrict__  in);
	unpacker horizontalunpack_c0;  unpacker horizontalunpack_c1;  unpacker horizontalunpack_c2;
	unpacker horizontalunpack_c3;  unpacker horizontalunpack_c4;  unpacker horizontalunpack_c5;
	unpacker horizontalunpack_c6;  unpacker horizontalunpack_c7;  unpacker horizontalunpack_c8;
	unpacker horizontalunpack_c9;  unpacker horizontalunpack_c10; unpacker horizontalunpack_c11;
	unpacker horizontalunpack_c12; unpacker horizontalunpack_c13; unpacker horizontalunpack_c14;
	unpacker horizontalunpack_c15; unpacker horizontalunpack_c16; unpacker horizontalunpack_c17;
	unpacker horizontalunpack_c18; unpacker horizontalunpack_c19; unpacker horizontalunpack_c20;
	unpacker horizontalunpack_c21; unpacker horizontalunpack_c22; unpacker horizontalunpack_c23;
	unpacker horizontalunpack_c24; unpacker horizontalunpack_c25; unpacker horizontalunpack_c26;
	unpacker horizontalunpack_c27; unpacker horizontalunpack_c28; unpacker horizontalunpack_c29;
	unpacker horizontalunpack_c30; unpacker horizontalunpack_c31; unpacker horizontalunpack_c32;

private:
	using unpackerhelper = void (datatype *  __restrict__  &out, const datatype &InReg);

	template <int imm8>
	unpackerhelper hor_sse4_unpack32_c1;
	template <int byte>
	unpackerhelper hor_sse4_unpack16_c1;
	template <int bit>
	unpackerhelper hor_sse4_unpack4_c1;
	template <int bit>
	unpackerhelper hor_sse4_unpackwithoutmask4_c1;

	template <int imm8>
    unpackerhelper hor_sse4_unpack16_c2;
	template <int bit>
    unpackerhelper hor_sse4_unpack4_c2;
	template <int bit>
	unpackerhelper hor_sse4_unpackwithoutmask4_c2;

	template <int byte>
	unpackerhelper hor_sse4_unpack8_c3;
	template <int bit>
	unpackerhelper hor_sse4_unpack4_c3;

	unpackerhelper hor_sse4_unpack32_c4;
	template <int byte>
	unpackerhelper hor_sse4_unpack4_c4_f1;
	template <int byte>
	unpackerhelper hor_sse4_unpack4_c4_f2;

	template <int imm8>
    unpackerhelper hor_sse4_unpack8_c4;
	template <int bit>
    unpackerhelper hor_sse4_unpack4_c4;
	template <int bit>
    unpackerhelper hor_sse4_unpackwithoutmask4_c4;

	template <int byte>
	unpackerhelper hor_sse4_unpack8_c5;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c6;
	template <int byte>
	unpackerhelper hor_sse4_unpack8_c7;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c8;
	template <int byte>
	unpackerhelper hor_sse4_unpack8_c9;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c10;
	template <int byte>
	unpackerhelper hor_sse4_unpack8_c11;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c12;
	template <int byte>
	unpackerhelper hor_sse4_unpack8_c13;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c14;
	template <int byte>
	unpackerhelper hor_sse4_unpack8_c15;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c16;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c17_f1;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c17_f2;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c18;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c19_f1;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c19_f2;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c20;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c21_f1;
	template <int byte>
	unpackerhelper hor_sse4_unpack4_c21_f2;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c22;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c23_f1;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c23_f2;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c24;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c25_f1;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c25_f2;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c26;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c27_f1;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c27_f2;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c28;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c29_f1;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c29_f2;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c30;

	template <int byte>
	unpackerhelper hor_sse4_unpack4_c31_f1;
	template <int byte>
    unpackerhelper hor_sse4_unpack4_c31_f2;

	const datatype *quotient = nullptr;
};

#include "HorSSEUnpacker.h"

#endif /* __SSE4_1__ */


#if CODECS_AVX_PREREQ(2, 0)

/**
 * AVX2-based bit unpacking for horizontal layout.
 *
 * Follows
 *
 * T. Willhalm, I. Oukid, I. Muller, and F. Faerber. Vectorizing database column scans
 * with complex predicates. In Proc. Workshop on Accelerating Data Management Systems
 * using Modern Processor and Storage Architectures, ADMS, pages 1–12, 2013.
 */
template <bool IsRiceCoding>
class HorUnpacker<AVX, IsRiceCoding> : public HorUnpackerBase<AVX, IsRiceCoding> {
	template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
	friend class Rice;
public:
	enum { PACKSIZE = 128 }; // number of elements to be unpacked for each invocation of unpack
	using datatype = AVX::datatype;

	HorUnpacker(const uint32_t *q = nullptr): quotient(reinterpret_cast<const datatype *>(q)) {
		assert(PACKSIZE == 128);
	}

	~HorUnpacker() {
		quotient = nullptr;
	}

	static const std::string name() {
		std::ostringstream unpackername;
		unpackername << "HorAVXUnpacker<" << PACKSIZE << ">";
		return unpackername.str();
	}

	void unpack(datatype *  __restrict__  out,
			const datatype *  __restrict__  in, uint32_t b) {
		const __m128i *in_m128i = reinterpret_cast<const __m128i *>(in);
	    // Could have used function pointers instead of switch.
	    // Switch calls do offer the compiler more opportunities
		// for optimization in theory. In this case, it makes no
		// difference with a good compiler.
	    switch(b) {
	    case 0: horizontalunpack_c0(out, in_m128i); return;
		case 1: horizontalunpack_c1(out, in_m128i); return;
		case 2: horizontalunpack_c2(out, in_m128i); return;
		case 3: horizontalunpack_c3(out, in_m128i); return;
		case 4: horizontalunpack_c4(out, in_m128i); return;
		case 5: horizontalunpack_c5(out, in_m128i); return;
		case 6: horizontalunpack_c6(out, in_m128i); return;
		case 7: horizontalunpack_c7(out, in_m128i); return;
		case 8: horizontalunpack_c8(out, in_m128i); return;
		case 9: horizontalunpack_c9(out, in_m128i); return;
		case 10: horizontalunpack_c10(out, in_m128i); return;
		case 11: horizontalunpack_c11(out, in_m128i); return;
		case 12: horizontalunpack_c12(out, in_m128i); return;
		case 13: horizontalunpack_c13(out, in_m128i); return;
		case 14: horizontalunpack_c14(out, in_m128i); return;
		case 15: horizontalunpack_c15(out, in_m128i); return;
		case 16: horizontalunpack_c16(out, in_m128i); return;
		case 17: horizontalunpack_c17(out, in_m128i); return;
		case 18: horizontalunpack_c18(out, in_m128i); return;
		case 19: horizontalunpack_c19(out, in_m128i); return;
		case 20: horizontalunpack_c20(out, in_m128i); return;
		case 21: horizontalunpack_c21(out, in_m128i); return;
		case 22: horizontalunpack_c22(out, in_m128i); return;
		case 23: horizontalunpack_c23(out, in_m128i); return;
		case 24: horizontalunpack_c24(out, in_m128i); return;
		case 25: horizontalunpack_c25(out, in_m128i); return;
		case 26: horizontalunpack_c26(out, in_m128i); return;
		case 27: horizontalunpack_c27(out, in_m128i); return;
		case 28: horizontalunpack_c28(out, in_m128i); return;
		case 29: horizontalunpack_c29(out, in_m128i); return;
		case 30: horizontalunpack_c30(out, in_m128i); return;
		case 31: horizontalunpack_c31(out, in_m128i); return;
		case 32: horizontalunpack_c32(out, in_m128i); return;
		default: return;
	    }
	}

	using unpacker = void (datatype *  __restrict__  out, const __m128i *  __restrict__  in);
	unpacker horizontalunpack_c0;  unpacker horizontalunpack_c1;  unpacker horizontalunpack_c2;
	unpacker horizontalunpack_c3;  unpacker horizontalunpack_c4;  unpacker horizontalunpack_c5;
	unpacker horizontalunpack_c6;  unpacker horizontalunpack_c7;  unpacker horizontalunpack_c8;
	unpacker horizontalunpack_c9;  unpacker horizontalunpack_c10; unpacker horizontalunpack_c11;
	unpacker horizontalunpack_c12; unpacker horizontalunpack_c13; unpacker horizontalunpack_c14;
	unpacker horizontalunpack_c15; unpacker horizontalunpack_c16; unpacker horizontalunpack_c17;
	unpacker horizontalunpack_c18; unpacker horizontalunpack_c19; unpacker horizontalunpack_c20;
	unpacker horizontalunpack_c21; unpacker horizontalunpack_c22; unpacker horizontalunpack_c23;
	unpacker horizontalunpack_c24; unpacker horizontalunpack_c25; unpacker horizontalunpack_c26;
	unpacker horizontalunpack_c27; unpacker horizontalunpack_c28; unpacker horizontalunpack_c29;
	unpacker horizontalunpack_c30; unpacker horizontalunpack_c31; unpacker horizontalunpack_c32;

private:
	using unpackerhelper = void (datatype *  __restrict__  &out, const datatype &InReg);

	template <int imm8>
	unpackerhelper hor_avx2_unpack32_c1;
	template <int bit>
	unpackerhelper hor_avx2_unpack8_c1;

	template <int byte>
	unpackerhelper hor_avx2_unpack32_c2;
	template <int bit>
	unpackerhelper hor_avx2_unpack8_c2;

	template <int byte>
	unpackerhelper hor_avx2_unpack16_c3;
	template <int bit>
	unpackerhelper hor_avx2_unpack8_c3;

	template <int bit>
	unpackerhelper hor_avx2_unpack8_c4;

	template <int byte>
	unpackerhelper hor_avx2_unpack16_c5;
	template <int bit>
	unpackerhelper hor_avx2_unpack8_c5;

	template <int byte>
	unpackerhelper hor_avx2_unpack16_c6;
	template <int bit>
	unpackerhelper hor_avx2_unpack8_c6;

	template <int byte>
	unpackerhelper hor_avx2_unpack16_c7;
	template <int bit>
	unpackerhelper hor_avx2_unpack8_c7;

	template <int byte>
	unpackerhelper hor_avx2_unpack8_c8;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack16_c9;
	template <int bit>
	unpackerhelper hor_avx2_unpack8_c9;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack16_c10;
	template <int bit>
	unpackerhelper hor_avx2_unpack8_c10;

	template <int byte>
	unpackerhelper hor_avx2_unpack8_c11;

	template <int byte>
	unpackerhelper hor_avx2_unpack8_c12;

	template <int byte>
	unpackerhelper hor_avx2_unpack8_c13;

	template <int byte>
	unpackerhelper hor_avx2_unpack8_c14;

	template <int byte>
	unpackerhelper hor_avx2_unpack8_c15;

	unpackerhelper hor_avx2_unpack8_c16;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c17;
	unpackerhelper hor_avx2_unpack8_c17;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c18;
	unpackerhelper hor_avx2_unpack8_c18;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c19;
	unpackerhelper hor_avx2_unpack8_c19;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c20;
	unpackerhelper hor_avx2_unpack8_c20;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c21;
	unpackerhelper hor_avx2_unpack8_c21;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c22;
	unpackerhelper hor_avx2_unpack8_c22;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c23;
	unpackerhelper hor_avx2_unpack8_c23;

	template <int byte>
	unpackerhelper hor_avx2_unpack8_c24;
	unpackerhelper hor_avx2_unpack8_c24;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c25;
	unpackerhelper hor_avx2_unpack8_c25;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c26;
	unpackerhelper hor_avx2_unpack8_c26;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c27;
	unpackerhelper hor_avx2_unpack8_c27;

	template <int byte>
	unpackerhelper hor_avx2_unpack8_c28;
	unpackerhelper hor_avx2_unpack8_c28;

	template <int byte1, int byte2>
	unpackerhelper hor_avx2_unpack8_c29;
	unpackerhelper hor_avx2_unpack8_c29;

	template <int byte>
	unpackerhelper hor_avx2_unpack8_c30;
	unpackerhelper hor_avx2_unpack8_c30;

	unpackerhelper hor_avx2_unpack8_c31;

	const datatype *quotient = nullptr;
};

#include "HorAVXUnpacker.h"

#endif /* __AVX2__ */

} // namespace Codecs

#endif // CODECS_HORUNPACKER_H_
