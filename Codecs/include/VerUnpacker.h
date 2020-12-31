/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Mar 16, 2016
 */

#ifndef CODECS_VERUNPACKER_H_
#define CODECS_VERUNPACKER_H_

#if !defined(__GNUC__) && !defined(_MSC_VER)
#error VerUnpacker.h requires GCC or MSVC
#endif

#include "Portability.h"
#include "util.h"

#if CODECS_SSE_PREREQ(2, 0)
namespace Codecs {
namespace SIMDMasks {
extern const __m128i SSE2_and_msk_m128i[33];
} // namespace SIMDMasks
} // namespace Codecs
#endif /* __SSE2 */

#if CODECS_AVX_PREREQ(2, 0)
namespace Codecs {
namespace SIMDMasks {
extern const __m256i AVX2_and_msk_m256i[33];
} // namespace SIMDMasks
} // namespace Codecs
#endif /* __AVX2__ */

namespace Codecs {

template <typename InstructionSet, bool IsRiceCoding>
class VerUnpacker;

template <bool IsRiceCoding>
using VerScalarUnpacker = VerUnpacker<Scalar, IsRiceCoding>;

#if CODECS_SSE_PREREQ(2, 0)
template <bool IsRiceCoding>
using VerSSEUnpacker = VerUnpacker<SSE, IsRiceCoding>;
#endif /* __SSE2__ */

#if CODECS_AVX_PREREQ(2, 0)
template <bool IsRiceCoding>
using VerAVXUnpacker = VerUnpacker<AVX, IsRiceCoding>;
#endif /* __AVX2__ */

template <typename InstructionSet, bool IsRiceCoding>
class VerUnpackerBase {
public:
	using datatype = typename InstructionSet::datatype;

	/**
	 * Pack into out Derived::PACKSIZE integers from in.
	 * Assumes that they fit in the prescribed number of bits.
	 */
	static void packwithoutmask(datatype *  __restrict__  out,
			const datatype *  __restrict__  in, uint32_t b) {
		// Could have used function pointers instead of switch.
		// Switch calls do offer the compiler more opportunities
		// for optimization in theory. In this case, it makes no
		// difference with a good compiler.
		switch(b) {
		case 0: return;
		case 1: Derived::verticalpackwithoutmask_c1(out, in); return;
		case 2: Derived::verticalpackwithoutmask_c2(out, in); return;
		case 3: Derived::verticalpackwithoutmask_c3(out, in); return;
		case 4: Derived::verticalpackwithoutmask_c4(out, in); return;
		case 5: Derived::verticalpackwithoutmask_c5(out, in); return;
		case 6: Derived::verticalpackwithoutmask_c6(out, in); return;
		case 7: Derived::verticalpackwithoutmask_c7(out, in); return;
		case 8: Derived::verticalpackwithoutmask_c8(out, in); return;
		case 9: Derived::verticalpackwithoutmask_c9(out, in); return;
		case 10: Derived::verticalpackwithoutmask_c10(out, in); return;
		case 11: Derived::verticalpackwithoutmask_c11(out, in); return;
		case 12: Derived::verticalpackwithoutmask_c12(out, in); return;
		case 13: Derived::verticalpackwithoutmask_c13(out, in); return;
		case 14: Derived::verticalpackwithoutmask_c14(out, in); return;
		case 15: Derived::verticalpackwithoutmask_c15(out, in); return;
		case 16: Derived::verticalpackwithoutmask_c16(out, in); return;
		case 17: Derived::verticalpackwithoutmask_c17(out, in); return;
		case 18: Derived::verticalpackwithoutmask_c18(out, in); return;
		case 19: Derived::verticalpackwithoutmask_c19(out, in); return;
		case 20: Derived::verticalpackwithoutmask_c20(out, in); return;
		case 21: Derived::verticalpackwithoutmask_c21(out, in); return;
		case 22: Derived::verticalpackwithoutmask_c22(out, in); return;
		case 23: Derived::verticalpackwithoutmask_c23(out, in); return;
		case 24: Derived::verticalpackwithoutmask_c24(out, in); return;
		case 25: Derived::verticalpackwithoutmask_c25(out, in); return;
		case 26: Derived::verticalpackwithoutmask_c26(out, in); return;
		case 27: Derived::verticalpackwithoutmask_c27(out, in); return;
		case 28: Derived::verticalpackwithoutmask_c28(out, in); return;
		case 29: Derived::verticalpackwithoutmask_c29(out, in); return;
		case 30: Derived::verticalpackwithoutmask_c30(out, in); return;
		case 31: Derived::verticalpackwithoutmask_c31(out, in); return;
		case 32: Derived::verticalpackwithoutmask_c32(out, in); return;
		default: return;
		}
	}

	/**
	 * Pack into out Derived::PACKSIZE integers from in.
	 * Assumes that they don't fit in the prescribed number of bits.
	 */
	static void pack(datatype *  __restrict__  out,
			const datatype *  __restrict__  in, uint32_t b) {
		// Could have used function pointers instead of switch.
		// Switch calls do offer the compiler more opportunities
		// for optimization in theory. In this case, it makes no
		// difference with a good compiler.
		switch(b) {
		case 0: return;
		case 1: Derived::verticalpack_c1(out, in); return;
		case 2: Derived::verticalpack_c2(out, in); return;
		case 3: Derived::verticalpack_c3(out, in); return;
		case 4: Derived::verticalpack_c4(out, in); return;
		case 5: Derived::verticalpack_c5(out, in); return;
		case 6: Derived::verticalpack_c6(out, in); return;
		case 7: Derived::verticalpack_c7(out, in); return;
		case 8: Derived::verticalpack_c8(out, in); return;
		case 9: Derived::verticalpack_c9(out, in); return;
		case 10: Derived::verticalpack_c10(out, in); return;
		case 11: Derived::verticalpack_c11(out, in); return;
		case 12: Derived::verticalpack_c12(out, in); return;
		case 13: Derived::verticalpack_c13(out, in); return;
		case 14: Derived::verticalpack_c14(out, in); return;
		case 15: Derived::verticalpack_c15(out, in); return;
		case 16: Derived::verticalpack_c16(out, in); return;
		case 17: Derived::verticalpack_c17(out, in); return;
		case 18: Derived::verticalpack_c18(out, in); return;
		case 19: Derived::verticalpack_c19(out, in); return;
		case 20: Derived::verticalpack_c20(out, in); return;
		case 21: Derived::verticalpack_c21(out, in); return;
		case 22: Derived::verticalpack_c22(out, in); return;
		case 23: Derived::verticalpack_c23(out, in); return;
		case 24: Derived::verticalpack_c24(out, in); return;
		case 25: Derived::verticalpack_c25(out, in); return;
		case 26: Derived::verticalpack_c26(out, in); return;
		case 27: Derived::verticalpack_c27(out, in); return;
		case 28: Derived::verticalpack_c28(out, in); return;
		case 29: Derived::verticalpack_c29(out, in); return;
		case 30: Derived::verticalpack_c30(out, in); return;
		case 31: Derived::verticalpack_c31(out, in); return;
		case 32: Derived::verticalpack_c32(out, in); return;
		default: return;
		}
	}

private:
	using Derived = VerUnpacker<InstructionSet, IsRiceCoding>;
};


/**
 * Scalar bit packing and unpacking for 4-way vertical layout.
 */
template <bool IsRiceCoding>
class VerUnpacker<Scalar, IsRiceCoding> : public VerUnpackerBase<Scalar, IsRiceCoding> {
	template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
	friend class Rice;
public:
	enum { PACKSIZE = 128 };
	using datatype = Scalar::datatype;

	VerUnpacker(const uint32_t *q = nullptr) : quotient(reinterpret_cast<const datatype *>(q)) {
		assert(PACKSIZE == 128);
	}

	~VerUnpacker() {
		quotient = nullptr;
	}

	static const std::string name() {
		std::ostringstream unpackername;
		unpackername << "VerScalarUnpacker<" << PACKSIZE << ">";
		return unpackername.str();
	}

	/**
	 * Unpack PACKSIZE b-bit values from in, and write them to out.
	 */
	void unpack(datatype *  __restrict__  out,
			const datatype *  __restrict__  in, uint32_t b) {
	    // Could have used function pointers instead of switch.
	    // Switch calls do offer the compiler more opportunities
		// for optimization in theory. In this case, it makes no
		// difference with a good compiler.
	    switch(b) {
	    case 0: verticalunpack_c0(out, in); return;
		case 1: verticalunpack_c1(out, in); return;
		case 2: verticalunpack_c2(out, in); return;
		case 3: verticalunpack_c3(out, in); return;
		case 4: verticalunpack_c4(out, in); return;
		case 5: verticalunpack_c5(out, in); return;
		case 6: verticalunpack_c6(out, in); return;
		case 7: verticalunpack_c7(out, in); return;
		case 8: verticalunpack_c8(out, in); return;
		case 9: verticalunpack_c9(out, in); return;
		case 10: verticalunpack_c10(out, in); return;
		case 11: verticalunpack_c11(out, in); return;
		case 12: verticalunpack_c12(out, in); return;
		case 13: verticalunpack_c13(out, in); return;
		case 14: verticalunpack_c14(out, in); return;
		case 15: verticalunpack_c15(out, in); return;
		case 16: verticalunpack_c16(out, in); return;
		case 17: verticalunpack_c17(out, in); return;
		case 18: verticalunpack_c18(out, in); return;
		case 19: verticalunpack_c19(out, in); return;
		case 20: verticalunpack_c20(out, in); return;
		case 21: verticalunpack_c21(out, in); return;
		case 22: verticalunpack_c22(out, in); return;
		case 23: verticalunpack_c23(out, in); return;
		case 24: verticalunpack_c24(out, in); return;
		case 25: verticalunpack_c25(out, in); return;
		case 26: verticalunpack_c26(out, in); return;
		case 27: verticalunpack_c27(out, in); return;
		case 28: verticalunpack_c28(out, in); return;
		case 29: verticalunpack_c29(out, in); return;
		case 30: verticalunpack_c30(out, in); return;
		case 31: verticalunpack_c31(out, in); return;
		case 32: verticalunpack_c32(out, in); return;
		default: return;
	    }
	}

	using unpacker = void (datatype *  __restrict__  out, const datatype *  __restrict__  in);
	unpacker verticalunpack_c0;  unpacker verticalunpack_c1;  unpacker verticalunpack_c2;
	unpacker verticalunpack_c3;  unpacker verticalunpack_c4;  unpacker verticalunpack_c5;
	unpacker verticalunpack_c6;  unpacker verticalunpack_c7;  unpacker verticalunpack_c8;
	unpacker verticalunpack_c9;  unpacker verticalunpack_c10; unpacker verticalunpack_c11;
	unpacker verticalunpack_c12; unpacker verticalunpack_c13; unpacker verticalunpack_c14;
	unpacker verticalunpack_c15; unpacker verticalunpack_c16; unpacker verticalunpack_c17;
	unpacker verticalunpack_c18; unpacker verticalunpack_c19; unpacker verticalunpack_c20;
	unpacker verticalunpack_c21; unpacker verticalunpack_c22; unpacker verticalunpack_c23;
	unpacker verticalunpack_c24; unpacker verticalunpack_c25; unpacker verticalunpack_c26;
	unpacker verticalunpack_c27; unpacker verticalunpack_c28; unpacker verticalunpack_c29;
	unpacker verticalunpack_c30; unpacker verticalunpack_c31; unpacker verticalunpack_c32;

	using packer = void (datatype *  __restrict__  out, const datatype *  __restrict__  in);
	static packer verticalpackwithoutmask_c1;  static packer verticalpackwithoutmask_c2;
	static packer verticalpackwithoutmask_c3;  static packer verticalpackwithoutmask_c4;
	static packer verticalpackwithoutmask_c5;  static packer verticalpackwithoutmask_c6;
	static packer verticalpackwithoutmask_c7;  static packer verticalpackwithoutmask_c8;
	static packer verticalpackwithoutmask_c9;  static packer verticalpackwithoutmask_c10;
	static packer verticalpackwithoutmask_c11; static packer verticalpackwithoutmask_c12;
	static packer verticalpackwithoutmask_c13; static packer verticalpackwithoutmask_c14;
	static packer verticalpackwithoutmask_c15; static packer verticalpackwithoutmask_c16;
	static packer verticalpackwithoutmask_c17; static packer verticalpackwithoutmask_c18;
	static packer verticalpackwithoutmask_c19; static packer verticalpackwithoutmask_c20;
	static packer verticalpackwithoutmask_c21; static packer verticalpackwithoutmask_c22;
	static packer verticalpackwithoutmask_c23; static packer verticalpackwithoutmask_c24;
	static packer verticalpackwithoutmask_c25; static packer verticalpackwithoutmask_c26;
	static packer verticalpackwithoutmask_c27; static packer verticalpackwithoutmask_c28;
	static packer verticalpackwithoutmask_c29; static packer verticalpackwithoutmask_c30;
	static packer verticalpackwithoutmask_c31; static packer verticalpackwithoutmask_c32;
	static packer verticalpack_c1;  static packer verticalpack_c2;
	static packer verticalpack_c3;  static packer verticalpack_c4;
	static packer verticalpack_c5;  static packer verticalpack_c6;
	static packer verticalpack_c7;  static packer verticalpack_c8;
	static packer verticalpack_c9;  static packer verticalpack_c10;
	static packer verticalpack_c11; static packer verticalpack_c12;
	static packer verticalpack_c13; static packer verticalpack_c14;
	static packer verticalpack_c15; static packer verticalpack_c16;
	static packer verticalpack_c17; static packer verticalpack_c18;
	static packer verticalpack_c19; static packer verticalpack_c20;
	static packer verticalpack_c21; static packer verticalpack_c22;
	static packer verticalpack_c23; static packer verticalpack_c24;
	static packer verticalpack_c25; static packer verticalpack_c26;
	static packer verticalpack_c27; static packer verticalpack_c28;
	static packer verticalpack_c29; static packer verticalpack_c30;
	static packer verticalpack_c31; static packer verticalpack_c32;

private:
	const datatype *quotient = nullptr; // For use with Rice and OptRice.
};

#include "VerScalarUnpacker.h"


#if CODECS_SSE_PREREQ(2, 0)
/**
 * SSE2-based bit packing and unpacking for 4-way vertical layout.
 *
 * Follows
 *
 * D. Lemire and L. Boytsov. Decoding billions of integers per second
 * through vectorization. Softw. Pract. Exper., 45(1):1–29, 2015.
 */
template <bool IsRiceCoding>
class VerUnpacker<SSE, IsRiceCoding> : public VerUnpackerBase<SSE, IsRiceCoding> {
	template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
	friend class Rice;
public:
	enum { PACKSIZE = 128 };
	using datatype = SSE::datatype;

	VerUnpacker(const uint32_t *q = nullptr) : quotient(reinterpret_cast<const datatype *>(q)) {
		assert(PACKSIZE == 128);
	}

	~VerUnpacker() {
		quotient = nullptr;
	}

	static const std::string name() {
		std::ostringstream unpackername;
		unpackername << "VerSSEUnpacker<" << PACKSIZE << ">";
		return unpackername.str();
	}

	/**
	 * Unpack PACKSIZE b-bit values from in, and write them to out.
	 */
	void unpack(datatype *  __restrict__  out,
			const datatype *  __restrict__  in, uint32_t b) {
	    // Could have used function pointers instead of switch.
	    // Switch calls do offer the compiler more opportunities
		// for optimization in theory. In this case, it makes no
		// difference with a good compiler.
	    switch(b) {
	    case 0: verticalunpack_c0(out, in); return;
		case 1: verticalunpack_c1(out, in); return;
		case 2: verticalunpack_c2(out, in); return;
		case 3: verticalunpack_c3(out, in); return;
		case 4: verticalunpack_c4(out, in); return;
		case 5: verticalunpack_c5(out, in); return;
		case 6: verticalunpack_c6(out, in); return;
		case 7: verticalunpack_c7(out, in); return;
		case 8: verticalunpack_c8(out, in); return;
		case 9: verticalunpack_c9(out, in); return;
		case 10: verticalunpack_c10(out, in); return;
		case 11: verticalunpack_c11(out, in); return;
		case 12: verticalunpack_c12(out, in); return;
		case 13: verticalunpack_c13(out, in); return;
		case 14: verticalunpack_c14(out, in); return;
		case 15: verticalunpack_c15(out, in); return;
		case 16: verticalunpack_c16(out, in); return;
		case 17: verticalunpack_c17(out, in); return;
		case 18: verticalunpack_c18(out, in); return;
		case 19: verticalunpack_c19(out, in); return;
		case 20: verticalunpack_c20(out, in); return;
		case 21: verticalunpack_c21(out, in); return;
		case 22: verticalunpack_c22(out, in); return;
		case 23: verticalunpack_c23(out, in); return;
		case 24: verticalunpack_c24(out, in); return;
		case 25: verticalunpack_c25(out, in); return;
		case 26: verticalunpack_c26(out, in); return;
		case 27: verticalunpack_c27(out, in); return;
		case 28: verticalunpack_c28(out, in); return;
		case 29: verticalunpack_c29(out, in); return;
		case 30: verticalunpack_c30(out, in); return;
		case 31: verticalunpack_c31(out, in); return;
		case 32: verticalunpack_c32(out, in); return;
		default: return;
	    }
	}

	using unpacker = void (datatype *  __restrict__  out, const datatype *  __restrict__  in);
	unpacker verticalunpack_c0;  unpacker verticalunpack_c1;  unpacker verticalunpack_c2;
	unpacker verticalunpack_c3;  unpacker verticalunpack_c4;  unpacker verticalunpack_c5;
	unpacker verticalunpack_c6;  unpacker verticalunpack_c7;  unpacker verticalunpack_c8;
	unpacker verticalunpack_c9;  unpacker verticalunpack_c10; unpacker verticalunpack_c11;
	unpacker verticalunpack_c12; unpacker verticalunpack_c13; unpacker verticalunpack_c14;
	unpacker verticalunpack_c15; unpacker verticalunpack_c16; unpacker verticalunpack_c17;
	unpacker verticalunpack_c18; unpacker verticalunpack_c19; unpacker verticalunpack_c20;
	unpacker verticalunpack_c21; unpacker verticalunpack_c22; unpacker verticalunpack_c23;
	unpacker verticalunpack_c24; unpacker verticalunpack_c25; unpacker verticalunpack_c26;
	unpacker verticalunpack_c27; unpacker verticalunpack_c28; unpacker verticalunpack_c29;
	unpacker verticalunpack_c30; unpacker verticalunpack_c31; unpacker verticalunpack_c32;


	using packer = void (datatype *  __restrict__  out, const datatype *  __restrict__  in);
	static packer verticalpackwithoutmask_c1;  static packer verticalpackwithoutmask_c2;
	static packer verticalpackwithoutmask_c3;  static packer verticalpackwithoutmask_c4;
	static packer verticalpackwithoutmask_c5;  static packer verticalpackwithoutmask_c6;
	static packer verticalpackwithoutmask_c7;  static packer verticalpackwithoutmask_c8;
	static packer verticalpackwithoutmask_c9;  static packer verticalpackwithoutmask_c10;
	static packer verticalpackwithoutmask_c11; static packer verticalpackwithoutmask_c12;
	static packer verticalpackwithoutmask_c13; static packer verticalpackwithoutmask_c14;
	static packer verticalpackwithoutmask_c15; static packer verticalpackwithoutmask_c16;
	static packer verticalpackwithoutmask_c17; static packer verticalpackwithoutmask_c18;
	static packer verticalpackwithoutmask_c19; static packer verticalpackwithoutmask_c20;
	static packer verticalpackwithoutmask_c21; static packer verticalpackwithoutmask_c22;
	static packer verticalpackwithoutmask_c23; static packer verticalpackwithoutmask_c24;
	static packer verticalpackwithoutmask_c25; static packer verticalpackwithoutmask_c26;
	static packer verticalpackwithoutmask_c27; static packer verticalpackwithoutmask_c28;
	static packer verticalpackwithoutmask_c29; static packer verticalpackwithoutmask_c30;
	static packer verticalpackwithoutmask_c31; static packer verticalpackwithoutmask_c32;
	static packer verticalpack_c1;  static packer verticalpack_c2;
	static packer verticalpack_c3;  static packer verticalpack_c4;
	static packer verticalpack_c5;  static packer verticalpack_c6;
	static packer verticalpack_c7;  static packer verticalpack_c8;
	static packer verticalpack_c9;  static packer verticalpack_c10;
	static packer verticalpack_c11; static packer verticalpack_c12;
	static packer verticalpack_c13; static packer verticalpack_c14;
	static packer verticalpack_c15; static packer verticalpack_c16;
	static packer verticalpack_c17; static packer verticalpack_c18;
	static packer verticalpack_c19; static packer verticalpack_c20;
	static packer verticalpack_c21; static packer verticalpack_c22;
	static packer verticalpack_c23; static packer verticalpack_c24;
	static packer verticalpack_c25; static packer verticalpack_c26;
	static packer verticalpack_c27; static packer verticalpack_c28;
	static packer verticalpack_c29; static packer verticalpack_c30;
	static packer verticalpack_c31; static packer verticalpack_c32;

private:
	const datatype *quotient = nullptr;
};

#include "VerSSEUnpacker.h"

#endif /* __SSE2__ */


#if CODECS_AVX_PREREQ(2, 0)
/**
 * AVX2-based bit packing and unpacking for 8-way vertical layout.
 */
template <bool IsRiceCoding>
class VerUnpacker<AVX, IsRiceCoding> : public VerUnpackerBase<AVX, IsRiceCoding> {
	template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
	friend class Rice;
public:
	enum { PACKSIZE = 256 };
	using datatype = AVX::datatype;

	VerUnpacker(const uint32_t *q = nullptr) : quotient(reinterpret_cast<const datatype *>(q)) {
		assert(PACKSIZE == 256);
	}

	~VerUnpacker() {
		quotient = nullptr;
	}

	static const std::string name() {
		std::ostringstream unpackername;
		unpackername << "VerAVXUnpacker<" << PACKSIZE << ">";
		return unpackername.str();
	}

	/**
	 * Unpack PACKSIZE b-bit values from in, and write them to out.
	 */
	void unpack(datatype *  __restrict__  out,
			const datatype *  __restrict__  in, uint32_t b) {
		    // Could have used function pointers instead of switch.
		    // Switch calls do offer the compiler more opportunities
			// for optimization in theory. In this case, it makes no
			// difference with a good compiler.
		    switch(b) {
		    case 0: verticalunpack_c0(out, in); return;
			case 1: verticalunpack_c1(out, in); return;
			case 2: verticalunpack_c2(out, in); return;
			case 3: verticalunpack_c3(out, in); return;
			case 4: verticalunpack_c4(out, in); return;
			case 5: verticalunpack_c5(out, in); return;
			case 6: verticalunpack_c6(out, in); return;
			case 7: verticalunpack_c7(out, in); return;
			case 8: verticalunpack_c8(out, in); return;
			case 9: verticalunpack_c9(out, in); return;
			case 10: verticalunpack_c10(out, in); return;
			case 11: verticalunpack_c11(out, in); return;
			case 12: verticalunpack_c12(out, in); return;
			case 13: verticalunpack_c13(out, in); return;
			case 14: verticalunpack_c14(out, in); return;
			case 15: verticalunpack_c15(out, in); return;
			case 16: verticalunpack_c16(out, in); return;
			case 17: verticalunpack_c17(out, in); return;
			case 18: verticalunpack_c18(out, in); return;
			case 19: verticalunpack_c19(out, in); return;
			case 20: verticalunpack_c20(out, in); return;
			case 21: verticalunpack_c21(out, in); return;
			case 22: verticalunpack_c22(out, in); return;
			case 23: verticalunpack_c23(out, in); return;
			case 24: verticalunpack_c24(out, in); return;
			case 25: verticalunpack_c25(out, in); return;
			case 26: verticalunpack_c26(out, in); return;
			case 27: verticalunpack_c27(out, in); return;
			case 28: verticalunpack_c28(out, in); return;
			case 29: verticalunpack_c29(out, in); return;
			case 30: verticalunpack_c30(out, in); return;
			case 31: verticalunpack_c31(out, in); return;
			case 32: verticalunpack_c32(out, in); return;
			default: return;
		    }
		}

		using unpacker = void (datatype *  __restrict__  out, const datatype *  __restrict__  in);
		unpacker verticalunpack_c0;  unpacker verticalunpack_c1;  unpacker verticalunpack_c2;
		unpacker verticalunpack_c3;  unpacker verticalunpack_c4;  unpacker verticalunpack_c5;
		unpacker verticalunpack_c6;  unpacker verticalunpack_c7;  unpacker verticalunpack_c8;
		unpacker verticalunpack_c9;  unpacker verticalunpack_c10; unpacker verticalunpack_c11;
		unpacker verticalunpack_c12; unpacker verticalunpack_c13; unpacker verticalunpack_c14;
		unpacker verticalunpack_c15; unpacker verticalunpack_c16; unpacker verticalunpack_c17;
		unpacker verticalunpack_c18; unpacker verticalunpack_c19; unpacker verticalunpack_c20;
		unpacker verticalunpack_c21; unpacker verticalunpack_c22; unpacker verticalunpack_c23;
		unpacker verticalunpack_c24; unpacker verticalunpack_c25; unpacker verticalunpack_c26;
		unpacker verticalunpack_c27; unpacker verticalunpack_c28; unpacker verticalunpack_c29;
		unpacker verticalunpack_c30; unpacker verticalunpack_c31; unpacker verticalunpack_c32;

		using packer = void (datatype *  __restrict__  out, const datatype *  __restrict__  in);
		static packer verticalpackwithoutmask_c1;  static packer verticalpackwithoutmask_c2;
		static packer verticalpackwithoutmask_c3;  static packer verticalpackwithoutmask_c4;
		static packer verticalpackwithoutmask_c5;  static packer verticalpackwithoutmask_c6;
		static packer verticalpackwithoutmask_c7;  static packer verticalpackwithoutmask_c8;
		static packer verticalpackwithoutmask_c9;  static packer verticalpackwithoutmask_c10;
		static packer verticalpackwithoutmask_c11; static packer verticalpackwithoutmask_c12;
		static packer verticalpackwithoutmask_c13; static packer verticalpackwithoutmask_c14;
		static packer verticalpackwithoutmask_c15; static packer verticalpackwithoutmask_c16;
		static packer verticalpackwithoutmask_c17; static packer verticalpackwithoutmask_c18;
		static packer verticalpackwithoutmask_c19; static packer verticalpackwithoutmask_c20;
		static packer verticalpackwithoutmask_c21; static packer verticalpackwithoutmask_c22;
		static packer verticalpackwithoutmask_c23; static packer verticalpackwithoutmask_c24;
		static packer verticalpackwithoutmask_c25; static packer verticalpackwithoutmask_c26;
		static packer verticalpackwithoutmask_c27; static packer verticalpackwithoutmask_c28;
		static packer verticalpackwithoutmask_c29; static packer verticalpackwithoutmask_c30;
		static packer verticalpackwithoutmask_c31; static packer verticalpackwithoutmask_c32;
		static packer verticalpack_c1;  static packer verticalpack_c2;
		static packer verticalpack_c3;  static packer verticalpack_c4;
		static packer verticalpack_c5;  static packer verticalpack_c6;
		static packer verticalpack_c7;  static packer verticalpack_c8;
		static packer verticalpack_c9;  static packer verticalpack_c10;
		static packer verticalpack_c11; static packer verticalpack_c12;
		static packer verticalpack_c13; static packer verticalpack_c14;
		static packer verticalpack_c15; static packer verticalpack_c16;
		static packer verticalpack_c17; static packer verticalpack_c18;
		static packer verticalpack_c19; static packer verticalpack_c20;
		static packer verticalpack_c21; static packer verticalpack_c22;
		static packer verticalpack_c23; static packer verticalpack_c24;
		static packer verticalpack_c25; static packer verticalpack_c26;
		static packer verticalpack_c27; static packer verticalpack_c28;
		static packer verticalpack_c29; static packer verticalpack_c30;
		static packer verticalpack_c31; static packer verticalpack_c32;

private:
	const datatype *quotient = nullptr;
};

#include "VerAVXUnpacker.h"

#endif /* __AVX2__ */

} // namespace Codecs


#endif // CODECS_VERUNPACKER_H_
