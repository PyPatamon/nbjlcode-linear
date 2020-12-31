/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Jan 14, 2015
 */
/**  
 * Based on code by
 *     Daniel Lemire, https://github.com/lemire/FastPFor
 * which was available under the Apache License, Version 2.0.
 */

#ifndef CODECS_NEWPFOR_H_
#define CODECS_NEWPFOR_H_ 

#include "Bits.h"
#include "common.h"
#include "util.h"
#include "Portability.h"

#include "IntegerCodec.h"
#include "Simple16.h"
#include "HorUnpacker.h"
#include "VerUnpacker.h"

namespace Codecs {

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
class NewPFor;

template <uint32_t BlockSize>
using NewPFor_Hor_Scalar = NewPFor<BlockSize, HorUnpacker<Scalar, false>, HorUnpacker<Scalar, false>, Simple16<Scalar, true> >;

#if CODECS_SSE_PREREQ(4, 1)
template <uint32_t BlockSize>
using NewPFor_Hor_SSE = NewPFor<BlockSize, HorUnpacker<SSE, false>, HorUnpacker<SSE, false>, Simple16<SSE, true> >;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
template <uint32_t BlockSize>
using NewPFor_Hor_AVX = NewPFor<BlockSize, HorUnpacker<AVX, false>, HorUnpacker<AVX, false>, Simple16<SSE, true> >;
#endif /* __AVX2__ */


template <uint32_t BlockSize>
using NewPFor_Ver_Scalar = NewPFor<BlockSize, VerUnpacker<Scalar, false>, HorUnpacker<Scalar, false>, Simple16<Scalar, true> >;

#if CODECS_SSE_PREREQ(4, 1)
template <uint32_t BlockSize>
using NewPFor_Ver_SSE = NewPFor<BlockSize, VerUnpacker<SSE, false>, HorUnpacker<SSE, false>, Simple16<SSE, true> >;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
template <uint32_t BlockSize>
using NewPFor_Ver_AVX = NewPFor<BlockSize, VerUnpacker<AVX, false>, HorUnpacker<AVX, false>, Simple16<SSE, true> >;
#endif /* __AVX2__ */

/**
 * NewPFD also known as NewPFor.
 *
 * In a multithreaded context, you may need one NewPFor per thread.
 *
 * Follows
 *
 * H. Yan, S. Ding, T. Suel, Inverted index compression and query processing with
 * optimized document ordering, in: WWW '09, 2009, pp. 401-410.
 */
template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
class NewPFor : public IntegerCodec {
public:
    NewPFor() : exceptionsPositions(BlockSize), exceptionsValues(BlockSize),
	            exceptions(4 * BlockSize), tobecoded(BlockSize),
			    unpacker(), tbunpacker(), ecoder() {
    	checkifdivisibleby(BlockSize, PACKSIZE);
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
				  << ecoder.name() << ">";
        return codecname.str();
    }

protected:
    static const std::vector<uint32_t> possLogs;
    std::vector<uint32_t> exceptionsPositions;
    std::vector<uint32_t> exceptionsValues;
    std::vector<uint32_t> exceptions;

    ExceptionCoder ecoder;         // coder for exceptions' positions & values

private:
    enum {
        PFORDELTA_B = 6,
        PFORDELTA_NEXCEPT = 10,
        PFORDELTA_EXCEPTSZ = 16
    };
    enum { PFORDELTA_RATIO = 10 }; // 10%; exception ratio (expressed as a percent).

    enum { PACKSIZE = Unpacker::PACKSIZE };
    enum { TBPACKSIZE = TailBlockUnpacker::PACKSIZE };
	using datatype = typename Unpacker::datatype;
	using tbdatatype = typename TailBlockUnpacker::datatype;

    virtual std::string basecodecname() const { return "NewPFor"; }
    virtual uint32_t tryB(uint32_t b, const uint32_t *in, uint32_t nvalue);
    virtual uint32_t findBestB(const uint32_t *in, uint32_t nvalue);

    std::vector<uint32_t> tobecoded;

    Unpacker unpacker;             // unpacker for all but the last block
    TailBlockUnpacker tbunpacker;  // unpacker for the last block
};

// nice compilers support this
template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
const std::vector<uint32_t> NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::possLogs =
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };

///// this is for brain dead compilers:
//static inline std::vector<uint32_t> __ihatestupidcompilers() {
//	std::vector<uint32_t> ans;
//	ans.push_back(0); // I
//	ans.push_back(1); // hate
//	ans.push_back(2); // stupid
//	ans.push_back(3); // compilers
//	ans.push_back(4);
//	ans.push_back(5);
//	ans.push_back(6);
//	ans.push_back(7);
//	ans.push_back(8);
//	ans.push_back(9);
//	ans.push_back(10);
//	ans.push_back(11);
//	ans.push_back(12);
//	ans.push_back(13);
//	ans.push_back(16);
//	ans.push_back(20);
//	ans.push_back(32);
//	return ans;
//}
//
//template <uint32_t BlockSize, typename Unpacker, typename ExceptionCoder>
//std::vector<uint32_t> NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::possLogs = __ihatestupidcompilers();

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
__attribute__ ((pure))
uint32_t NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::tryB(uint32_t b, const uint32_t *in, uint32_t nvalue) {
    assert(b <= 32);

    if (b == 32)
        return 0;

    uint32_t nExceptions = 0;
    for (uint32_t i = 0; i < nvalue; ++i) {
        if (in[i] >= (1U << b))
        	++nExceptions;
    }

    return nExceptions;
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
__attribute__ ((pure))
uint32_t NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::findBestB(const uint32_t *in, uint32_t nvalue) {
    // Some schemes such as Simple16 don't code numbers greater than (1 << 28) - 1.
    const uint32_t mb = maxbits(in, in + nvalue);
    uint32_t i = 0;
    while (mb > 28 + possLogs[i]) 
        ++i; 

    for ( ; i < possLogs.size() - 1; ++i) {
        const uint32_t nExceptions = tryB(possLogs[i], in, nvalue);
        if (nExceptions * 100 <= nvalue * PFORDELTA_RATIO)
            return possLogs[i];
    }
    return possLogs.back();
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
template <bool IsTailBlock>
void NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::encodeBlock(const uint32_t *in, uint32_t nvalue,
		uint32_t *out, uint32_t &csize) {
    if (!IsTailBlock) {
        assert(nvalue == BlockSize);
    }
    else {
        assert(nvalue <= BlockSize);
    }

    uint32_t b = findBestB(in, nvalue);
    if (b < 32) {
        // Reserve space for the descriptor.
    	uint32_t *const initout(out); // We use this later.
    	++out;  

    	uint32_t nExceptions = 0;
        for (uint32_t i = 0; i < nvalue; ++i) {
            if (in[i] >= (1U << b)) {
                tobecoded[i] = in[i] & ((1U << b) - 1);
                exceptionsPositions[nExceptions] = i;
                exceptionsValues[nExceptions] = (in[i] >> b);
                ++nExceptions;
            }
            else {
                tobecoded[i] = in[i];
            }
        }

        // Pack least significant b bits of all values.
        if (!IsTailBlock) { // For all but the last block.
        	for (uint32_t valuesPacked = 0; valuesPacked < BlockSize; valuesPacked += PACKSIZE) {
        		Unpacker::packwithoutmask(reinterpret_cast<datatype *>(out),
                        reinterpret_cast<const datatype *>(&tobecoded[valuesPacked]), b);
        		out += (PACKSIZE * b) / 32;
        	}
        }
        else {  // For the last block.
        	uint32_t valuesPacked = 0;
        	for ( ; valuesPacked + TBPACKSIZE < nvalue; valuesPacked += TBPACKSIZE) {
                TailBlockUnpacker::packwithoutmask(reinterpret_cast<tbdatatype *>(out),
                        reinterpret_cast<const tbdatatype *>(&tobecoded[valuesPacked]), b);
				out += (TBPACKSIZE * b) / 32;
        	}

            uint32_t valuesRemaining = nvalue - valuesPacked;
            TailBlockUnpacker::packwithoutmask_generic(out, &tobecoded[valuesPacked], b, valuesRemaining);
        	out += div_roundup(valuesRemaining * b, 32);
        }

        if (nExceptions > 0) {
            for (uint32_t i = nExceptions - 1; i > 0; --i) {
                const uint32_t cur = exceptionsPositions[i];
                const uint32_t prev = exceptionsPositions[i - 1];
                exceptionsPositions[i] = cur - prev - 1;
            }

            for (uint32_t i = 0; i < nExceptions; ++i) {
                exceptions[i] = exceptionsPositions[i];
                exceptions[i + nExceptions] = exceptionsValues[i] - 1;
            }
        }

        // Pack exceptions' positions and values.
        uint64_t encodedExceptionsSize = 0;
        if (nExceptions > 0)
            ecoder.encodeArray(&exceptions[0], 2 * nExceptions, out, encodedExceptionsSize);
        out += static_cast<uint32_t>(encodedExceptionsSize);

        // Write descriptor.
        *initout = (b << (32 - PFORDELTA_B)) |
        	   (nExceptions << PFORDELTA_EXCEPTSZ) |
				static_cast<uint32_t>(encodedExceptionsSize);

        csize = out - initout;
    }
    else { // b == 32
        *out = (b << (32 - PFORDELTA_B));
        ++out;
        for (uint32_t i = 0; i < nvalue; ++i)
            out[i] = in[i];

        csize =  1 + nvalue;
    }
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
void NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::encodeArray(const uint32_t *in, uint64_t nvalue,
		uint32_t *out, uint64_t &csize) {
	const uint32_t *const initout(out);

	uint32_t blockcsize = 0;
    uint64_t numBlocks = div_roundup(nvalue, BlockSize); // Number of blocks in total;
                                                         // No need to output it.
    // For all but the last block.
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

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
template <bool IsTailBlock>
const uint32_t * NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::decodeBlock(const uint32_t *in, uint32_t csize,
		uint32_t *out, uint32_t nvalue) {
    assert(nvalue <= BlockSize);

    const uint32_t b = *in >> (32 - PFORDELTA_B);
    const uint32_t nExceptions = (*in >> PFORDELTA_EXCEPTSZ) & ((1U << PFORDELTA_NEXCEPT) - 1);
    const uint32_t encodedExceptionsSize = *in & ((1U << PFORDELTA_EXCEPTSZ) - 1);
    ++in;
    assert(nExceptions <= nvalue);

    uint32_t *beginout(out); // We use this later.

    if (!IsTailBlock) { // For all but the last block.
    	for (uint32_t valuesUnpacked = 0; valuesUnpacked < BlockSize; valuesUnpacked += PACKSIZE) {
    		unpacker.unpack(reinterpret_cast<datatype *>(out),
    				reinterpret_cast<const datatype *>(in), b);
    		in += (PACKSIZE * b) / 32;
    		out += PACKSIZE;
    	}
    }
    else { // For the last block.
    	uint32_t valuesUnpacked = 0;
    	for ( ; valuesUnpacked + TBPACKSIZE < nvalue; valuesUnpacked += TBPACKSIZE) {
    		tbunpacker.unpack(reinterpret_cast<tbdatatype *>(out),
                    reinterpret_cast<const tbdatatype *>(in), b);
    		in += (TBPACKSIZE * b) / 32;
    		out += TBPACKSIZE;
    	}

        uint32_t valuesRemaining = nvalue - valuesUnpacked;
        TailBlockUnpacker::unpack_generic(out, in, b, valuesRemaining);
    	in += div_roundup(valuesRemaining * b, 32);
    }

    if (nExceptions > 0) {
        ecoder.decodeArray(in, encodedExceptionsSize, &exceptions[0], 2 * nExceptions);
		in += encodedExceptionsSize;
	}

    // Recover the exceptions.
    for (uint32_t e = 0, lpos = -1; e < nExceptions; ++e) {
        lpos += exceptions[e] + 1;
        beginout[lpos] |= (exceptions[e + nExceptions] + 1) << b;
    }

    return in;
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
const uint32_t * NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::decodeArray(const uint32_t *in, uint64_t csize,
		uint32_t *out, uint64_t nvalue) {
    const uint64_t numBlocks = div_roundup(nvalue, BlockSize); // Number of blocks in total.

    // For all but the last block.
    for (uint64_t i = 0; i < numBlocks - 1; ++i) {
        in = decodeBlock(in, 0, out);
        out += BlockSize;
    }

    // For the last block.
    uint32_t tailBlockSize = static_cast<uint32_t>(nvalue - (numBlocks - 1) * BlockSize);
    in = decodeBlock<true>(in, 0, out, tailBlockSize);

    return in;
}

} // namespace Codecs

#endif // CODECS_NEWPFOR_H_ 
