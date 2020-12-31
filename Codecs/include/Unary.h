/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Dec 31, 2014
 */

#ifndef CODECS_UNARY_H_
#define CODECS_UNARY_H_ 

#include "IntegerCodec.h"
#include "Portability.h"

namespace Codecs {

// Note: 0 indicates continuation, whereas 1 indicates termination.
// For example, 0 is represented as 1, 1 as 10, 2 as 100, and so on.
template <bool AllowOverShooting>
class Unary : public IntegerCodec {
public:
	virtual void encodeArray(const uint32_t *in, uint64_t nvalue,
			uint32_t *out, uint64_t &csize) {
		// Compute sum[i] = in[0] + in[1] + ... + in[i] + i, for i = 0, 1, ..., nvalue-1.
		std::vector<uint64_t> sum(in, in + nvalue);
		std::partial_sum(sum.begin(), sum.end(), sum.begin(),
				[](uint64_t x, uint64_t y) { return x + y + 1; } );

		// Compute the total number of words needed and set those words to 0
		csize = div_roundup(sum[nvalue - 1] + 1, 32);
		memset(out, 0, sizeof(uint32_t) * csize); // FIXME: memset is not fast enough.

		// Flip termination bits.
		uint64_t idx = 0;
		uint32_t shift = 0;
		for (uint64_t i = 0; i < nvalue; ++i) {
			idx = sum[i] >> 5;
			shift = static_cast<uint32_t>(sum[i] & 0x1f);
			out[idx] |= 1 << shift;
		}

		// Flip all unused bits.
		out[idx] |= ~((1 << shift) - 1);
	}

	// For use with OptRice.
	// Like encodeArray, but does not actually write out the data.
	void fakeencodeArray(const uint32_t *in, uint64_t nvalue,
			uint64_t &csize) {
		// Compute sum[i] = in[0] + in[1] + ... + in[i] + i, for i = 0, 1, ..., nvalue-1
		std::vector<uint64_t> sum(in, in + nvalue);
		std::partial_sum(sum.begin(), sum.end(), sum.begin(),
				[](uint64_t x, uint64_t y) { return x + y + 1; } );

		// Compute the total number of words needed.
		csize = div_roundup(sum[nvalue - 1] + 1, 32);
	}

	static void decodeWord(uint32_t *&out, uint32_t dword, uint32_t &carry) {
	    uint32_t tmpdword = dword;
		const int ones = __builtin_popcount(dword);
		switch (ones) {
		case 0:
			carry += 32;

			break;
		case 32:
            memset32(out);
			out[0] += carry;
			carry = 0;

			break;
		default:
			for (int i = 0; i < ones; ++i) {
				int zeros = __builtin_ctz(dword);
				out[i] = zeros;
				dword >>= zeros + 1;
			}
			out[0] += carry;
			carry = __builtin_clz(tmpdword);

			break;
		}

		out += ones;
	}

	virtual const uint32_t *decodeArray(const uint32_t *in, uint64_t csize,
			uint32_t *out, uint64_t nvalue) {
		uint32_t carry = 0;
	    const uint32_t *const endout = out + nvalue;

	    if (AllowOverShooting) {
	    	while (endout > out) {
	    		uint32_t dword = in[0];
				++in;
				decodeWord(out, dword, carry);
			}
	    }
	    else {
	    	while (endout - out >= 32) {
	    		uint32_t dword = in[0];
				++in;
				decodeWord(out, dword, carry);
	    	}

	    	uint32_t valuesRemaining = endout - out;
	    	uint32_t *ptr = buf;
	    	const uint32_t *const endptr = buf + valuesRemaining;
	    	while (endptr > ptr) {
	    		uint32_t dword = in[0];
				++in;
				decodeWord(ptr, dword, carry);
	    	}
	    	memcpy(out, buf, sizeof(uint32_t) * valuesRemaining);
	    }

	    return in;
	}

	std::string name() const {
		return "Unary";
	}

private:
	// We determine the size of buf based on:
	// 1) there're at most 31 values remaining to be decoded;
	// 2) decodeWord can write out at most 31 garbage values
	//    when decoding the last 32-bit word.
	uint32_t buf[31 * 2];
};

//// Slower alternatives
//template <bool AllowOverShooting>
//const uint32_t * Unary<AllowOverShooting>::decodeArray(const uint32_t *in, uint64_t csize,
//		uint32_t *out, uint64_t nvalue) {
//	const uint64_t *in64 = reinterpret_cast<const uint64_t *>(in);
//	uint32_t carry = 0;
//	const uint32_t *const endout(out + nvalue);
//
//	while (endout > out) {
//		uint64_t dword = in64[0], tmpdword = dword;
//		++in64;
//		const int ones = __builtin_popcountll(dword);
//		switch (ones) {
//		case 0:
//			carry += 64;
//
//			break;
//		case 64:
//			memset(out, 0, sizeof(uint32_t) * 64);
//			out[0] += carry;
//			carry = 0;
//
//			break;
//		default: // FIXME: eliminate for loop
//			for (int i = 0; i < ones; ++i) {
//				out[i] = __builtin_ctzll(dword);
//				dword >>= out[i] + 1;
//			}
//			out[0] += carry;
//			carry = __builtin_clzll(tmpdword);
//
//			break;
//		}
//
//		out += ones;
//	}
//
//	return reinterpret_cast<const uint32_t *>(in64);
//}
//
//template <bool AllowOverShooting>
//const uint32_t * Unary<AllowOverShooting>::decodeArray(const uint32_t *in, uint64_t csize,
//		uint32_t *out, uint64_t nvalue) {
//	const uint16_t *in16 = reinterpret_cast<const uint16_t *>(in);
//	uint32_t carry = 0;
//	const uint32_t *const endout(out + nvalue);
//
//	while (endout > out) {
//		uint16_t dword = in16[0], tmpdword = dword;
//		++in16;
//		const int ones = __builtin_popcount(dword);
//		switch (ones) {
//		case 0:
//			carry += 16;
//
//			break;
//		case 16:
//			memset(out, 0, sizeof(uint32_t) * 16);
//			out[0] += carry;
//			carry = 0;
//
//			break;
//		default: // FIXME: eliminate for loop
//			for (int i = 0; i < ones; ++i) {
//				out[i] = __builtin_ctz(dword);
//				dword >>= out[i] + 1;
//			}
//			out[0] += carry;
//			carry = __builtin_clz(tmpdword) - 16;
//
//			break;
//		}
//
//		out += ones;
//	}
//
//	return reinterpret_cast<const uint32_t *>(in16);
//}
//
//template <bool AllowOverShooting>
//const uint32_t * Unary<AllowOverShooting>::decodeArray(const uint32_t *in, uint64_t csize,
//		uint32_t *out, uint64_t nvalue) {
//	uint32_t carry(0);
//	const uint32_t *const endout(out + nvalue);
//
//	while (endout > out) {
//		uint32_t *const beginout(out);
//		uint32_t dword = in[0], tmpdword = dword;
//		++in;
//		const int ones = __builtin_popcount(dword);
//		switch (ones) {
//		case 0:
//			carry += 32;
//
//			break;
//		case 32:
//			memset(out, 0, sizeof(uint32_t) * 32);
//			out[0] += carry;
//			carry = 0;
//
//			out += 32;
//
//			break;
//		case 31:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 30:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 29:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 28:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 27:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 26:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 25:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 24:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 23:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 22:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 21:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 20:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 19:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 18:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 17:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 16:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 15:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 14:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 13:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 12:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 11:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 10:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 9:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 8:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 7:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 6:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 5:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 4:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 3:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 2:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//			/* no break */
//		case 1:
//			out[0] = __builtin_ctz(dword);
//			dword >>= out[0] + 1;
//
//			++out;
//
//			beginout[0] += carry;
//			carry = __builtin_clz(tmpdword);
//
//			break;
//		default:
//			break;
//		}
//	}
//
//	return in;
//}

} // namespace Codecs

#endif // CODECS_UNARY_H_ 
