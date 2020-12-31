/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Jan 16, 2015
 */

#ifndef CODECS_RICE_H_
#define CODECS_RICE_H_ 

#include "Bits.h"
#include "common.h"
#include "util.h"

#include "IntegerCodec.h"
#include "Unary.h"

#include "HorUnpacker.h"
#include "VerUnpacker.h"

#include "Portability.h"


namespace Codecs {

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
class Rice;

template <uint32_t BlockSize>
using Rice_Hor_Scalar = Rice<BlockSize, HorUnpacker<Scalar, true>, HorUnpacker<Scalar, true>, Unary<true> >;

#if CODECS_SSE_PREREQ(4, 1)
template <uint32_t BlockSize>
using Rice_Hor_SSE = Rice<BlockSize, HorUnpacker<SSE, true>, HorUnpacker<SSE, true>, Unary<true> >;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
template <uint32_t BlockSize>
using Rice_Hor_AVX = Rice<BlockSize, HorUnpacker<AVX, true>, HorUnpacker<AVX, true>, Unary<true> >;
#endif /* __AVX2__ */


template <uint32_t BlockSize>
using Rice_Ver_Scalar = Rice<BlockSize, VerUnpacker<Scalar, true>, HorUnpacker<Scalar, true>, Unary<true> >;

#if CODECS_SSE_PREREQ(4, 1)
template <uint32_t BlockSize>
using Rice_Ver_SSE = Rice<BlockSize, VerUnpacker<SSE, true>, HorUnpacker<SSE, true>, Unary<true> >;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
template <uint32_t BlockSize>
using Rice_Ver_AVX = Rice<BlockSize, VerUnpacker<AVX, true>, HorUnpacker<AVX, true>, Unary<true> >;
#endif /* __AVX2__ */


template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
class Rice : public IntegerCodec {
public:
	Rice() : quotient(4 * BlockSize), remainder(BlockSize),
		  	 unpacker(&quotient[0]), tbunpacker(&quotient[0]), ucoder() {
		checkifdivisibleby(BlockSize, unpacker.PACKSIZE);
	}

    template <bool IsTailBlock = false>
    void encodeBlock(const uint32_t *in, uint32_t nvalue,
    		uint32_t *out, uint32_t &csize);

	virtual void encodeArray(const uint32_t *in, uint64_t nvalue,
			uint32_t *out, uint64_t &csize);

    template <bool IsTailBlock = false>
    const uint32_t * decodeBlock(const uint32_t* in, uint32_t csize,
    		uint32_t* out, uint32_t nvalue = BlockSize);

	virtual const uint32_t * decodeArray(const uint32_t *in, uint64_t csize,
			uint32_t *out, uint64_t nvalue);

	virtual std::string name() const {
		std::ostringstream codecname;
		codecname << basecodecname() << "<" << BlockSize << ", "
				  << unpacker.name() << ", " << tbunpacker.name() << ", "
				  << ucoder.name() << ">";
		return codecname.str();
	}

protected:
	std::vector<uint32_t> quotient;
	std::vector<uint32_t> remainder;

	UnaryCoder ucoder;

private:
	enum {
		RICE_B = 6,
		RICE_UNARYSZ = 26,
	};
	enum { RICE_RATIO = 69 };   // Expressed as a percent.

    enum { PACKSIZE = Unpacker::PACKSIZE };
    enum { TBPACKSIZE = TailBlockUnpacker::PACKSIZE };
	using datatype = typename Unpacker::datatype;
	using tbdatatype = typename TailBlockUnpacker::datatype;

    virtual std::string basecodecname() const { return "Rice"; }
	virtual uint32_t findBestB(const uint32_t *in, uint32_t nvalue) {
        double avg = std::accumulate(in, in + nvalue, 0.0) / nvalue;
        uint32_t b = gccbits(static_cast<uint32_t>(RICE_RATIO / 100.0 * avg));
        return b;
    }

	Unpacker unpacker;
	TailBlockUnpacker tbunpacker;
};

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
template <bool IsTailBlock>
void Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::encodeBlock(const uint32_t *in, uint32_t nvalue,
		uint32_t *out, uint32_t &csize) {
     assert(nvalue <= BlockSize);

	const uint32_t b = findBestB(in, nvalue);
	if (b < 32) {
		uint32_t *const initout(out); // We use this later.
		++out;

		for (uint32_t i = 0; i < nvalue; ++i) {
			quotient[i] = in[i] >> b;
			remainder[i] = in[i] & ((1U << b) - 1);
		}

		uint64_t encodedQuotientSize = 0;
		ucoder.encodeArray(&quotient[0], nvalue, out, encodedQuotientSize);
		out += encodedQuotientSize;

		if (!IsTailBlock) { // For all but the last block.
			for (uint32_t valuesPacked = 0; valuesPacked < BlockSize; valuesPacked += PACKSIZE) {
				Unpacker::packwithoutmask(reinterpret_cast<datatype *>(out),
						reinterpret_cast<const datatype *>(&remainder[valuesPacked]), b);
				out += (PACKSIZE * b) / 32;
			}
		}
		else { // For the last block.
			uint32_t valuesPacked = 0;
			for ( ; valuesPacked + TBPACKSIZE < nvalue; valuesPacked += TBPACKSIZE) {
				TailBlockUnpacker::packwithoutmask(reinterpret_cast<tbdatatype *>(out),
						reinterpret_cast<const tbdatatype *>(&remainder[valuesPacked]), b);
				out += (TBPACKSIZE * b) / 32;
			}

			uint32_t valuesRemaining = nvalue - valuesPacked;
			TailBlockUnpacker::packwithoutmask_generic(out, &remainder[valuesPacked], b, valuesRemaining);
			out += div_roundup(valuesRemaining * b, 32);
		}

		// Write descriptor.
		*initout = (b << (32 - RICE_B)) | static_cast<uint32_t>(encodedQuotientSize);

		csize = out - initout;
	}
	else { // b == 32 (quotient part will be all 0s).
        *out = (b << (32 - RICE_B));
        ++out;
        for (uint32_t i = 0; i < nvalue; ++i) {
            out[i] = in[i];
        }

        csize =  1 + nvalue;
	}
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
void Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::encodeArray(const uint32_t *in, uint64_t nvalue,
		uint32_t *out, uint64_t &csize) {
	const uint32_t *const initout(out);

	uint32_t blockcsize = 0;
    uint64_t numBlocks = div_roundup(nvalue, BlockSize); // Number of blocks in total; 
                                                         // No need to output it.
    // For all blocks except the last block.
    for (uint64_t i = 0; i < numBlocks - 1; ++i) {
        encodeBlock(in, BlockSize, out, blockcsize);
        in += BlockSize;
        out += blockcsize;
    }

    // For the last block.
    uint32_t tailBlockSize = static_cast<uint32_t>( nvalue - (numBlocks - 1) * BlockSize );
    encodeBlock<true>(in, tailBlockSize, out, blockcsize);
    out += blockcsize;

    csize = out - initout;
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
template <bool IsTailBlock>
const uint32_t * Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::decodeBlock(const uint32_t *in, uint32_t cszie,
		uint32_t *out, uint32_t nvalue) {
    assert(nvalue <= BlockSize);

	const uint32_t b = *in >> (32 - RICE_B);
	const uint64_t encodedQuotientSize = *in & ((1U << RICE_UNARYSZ) - 1);
	++in;

	if (b < 32) {
		ucoder.decodeArray(in, encodedQuotientSize, &quotient[0], nvalue);
		in += encodedQuotientSize;
	}

    const uint32_t *const beginout = out;

	if (!IsTailBlock) { // For all but the last block.
		for (uint32_t valuesUnpacked = 0; valuesUnpacked < BlockSize; valuesUnpacked += PACKSIZE) {
            unpacker.quotient = reinterpret_cast<const datatype *>(&quotient[valuesUnpacked]);
			unpacker.unpack(reinterpret_cast<datatype *>(out), reinterpret_cast<const datatype *>(in), b);
			in += (PACKSIZE * b) / 32;
			out += PACKSIZE;
		}
	}
	else { // For the last block.
		uint32_t valuesUnpacked = 0;
		for ( ; valuesUnpacked + TBPACKSIZE < nvalue; valuesUnpacked += TBPACKSIZE) {
            tbunpacker.quotient = reinterpret_cast<const tbdatatype *>(&quotient[valuesUnpacked]);
			tbunpacker.unpack(reinterpret_cast<tbdatatype *>(out), reinterpret_cast<const tbdatatype *>(in), b);
			in += (TBPACKSIZE * b) / 32;
			out += TBPACKSIZE;
		}

		uint32_t valuesRemaining = nvalue - valuesUnpacked;
        TailBlockUnpacker::unpack_generic(out, in, b, valuesRemaining);
		in += div_roundup(valuesRemaining * b, 32);

		if (b < 32) {
			for (uint32_t i = 0; i < valuesRemaining; ++i)
				out[i] |= quotient[valuesUnpacked + i] << b;
		}
	}

	return in;
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
const uint32_t * Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder>::decodeArray(const uint32_t *in, uint64_t csize,
		uint32_t *out, uint64_t nvalue) {
    const uint64_t numBlocks = div_roundup(nvalue, BlockSize); // Number of blocks in total.

    // For all but the last block.
    for (uint64_t i = 0; i < numBlocks - 1; ++i) {
        in = decodeBlock(in, 0, out);
        out += BlockSize;
    }

    // For the last block.
    uint32_t tailBlockSize = static_cast<uint64_t>(nvalue - (numBlocks - 1) * BlockSize);
    in = decodeBlock<true>(in, 0, out, tailBlockSize);

    return in;
}

} // namespace Codecs

#endif // CODECS_RICE_H_ 
