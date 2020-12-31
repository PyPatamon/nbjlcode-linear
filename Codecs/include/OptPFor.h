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

#ifndef CODECS_OPTPFOR_H_
#define CODECS_OPTPFOR_H_

#include "NewPFor.h"

namespace Codecs {

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
class OptPFor;

template <uint32_t BlockSize>
using OptPFor_Hor_Scalar = OptPFor<BlockSize, HorUnpacker<Scalar, false>, HorUnpacker<Scalar, false>, Simple16<Scalar, true> >;

#if CODECS_SSE_PREREQ(4, 1)
template <uint32_t BlockSize>
using OptPFor_Hor_SSE = OptPFor<BlockSize, HorUnpacker<SSE, false>, HorUnpacker<SSE, false>, Simple16<SSE, true> >;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
template <uint32_t BlockSize>
using OptPFor_Hor_AVX = OptPFor<BlockSize, HorUnpacker<AVX, false>, HorUnpacker<AVX, false>, Simple16<SSE, true> >;
#endif /* __AVX2__ */

template <uint32_t BlockSize>
using OptPFor_Ver_Scalar = OptPFor<BlockSize, VerUnpacker<Scalar, false>, HorUnpacker<Scalar, false>, Simple16<Scalar, true> >;

#if CODECS_SSE_PREREQ(4, 1)
template <uint32_t BlockSize>
using OptPFor_Ver_SSE = OptPFor<BlockSize, VerUnpacker<SSE, false>, HorUnpacker<SSE, false>, Simple16<SSE, true> >;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
template <uint32_t BlockSize>
using OptPFor_Ver_AVX = OptPFor<BlockSize, VerUnpacker<AVX, false>, HorUnpacker<AVX, false>, Simple16<SSE, true> >;
#endif /* __AVX2__ */

/**
 * OptPFD
 *
 * In a multithreaded context, you may need one OPTPFor per thread.
 *
 * Follows:
 *
 * H. Yan, S. Ding, T. Suel, Inverted index compression and query processing with
 * optimized document ordering, in: WWW '09, 2009, pp. 401-410.
 */
template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
class OptPFor : public NewPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder> {
private:
    virtual std::string basecodecname() const { return "OptPFor"; }
    virtual uint32_t tryB(uint32_t b, const uint32_t *in, uint32_t nvalue);
    virtual uint32_t findBestB(const uint32_t *in, uint32_t nvalue);
};

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
__attribute__ ((pure))
uint32_t OptPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::tryB(uint32_t b, const uint32_t *in, uint32_t nvalue) {
    assert(b <= 32);

    if (b == 32) {
    	return nvalue;
    }

    uint32_t size = div_roundup(nvalue * b, 32);
    uint32_t nExceptions = 0;
    for (uint32_t i = 0; i < nvalue; ++i) {
        if (in[i] >= (1U << b)) {
            this->exceptionsPositions[nExceptions] = i;
            this->exceptionsValues[nExceptions] = (in[i] >> b);
            ++nExceptions;
        }
    }

    if (nExceptions > 0) {
        for (uint32_t i = nExceptions - 1; i > 0; --i) {
            const uint32_t cur = this->exceptionsPositions[i];
            const uint32_t prev = this->exceptionsPositions[i - 1];
            this->exceptionsPositions[i] = cur - prev - 1;
        }

        for (uint32_t i = 0; i < nExceptions; i++) {
            this->exceptions[i] = this->exceptionsPositions[i];
            this->exceptions[i + nExceptions] = this->exceptionsValues[i] - 1;
        }

        uint64_t encodedExceptionsSize = 0;
        this->ecoder.fakeencodeArray(&this->exceptions[0], 2 * nExceptions, encodedExceptionsSize);

        size += static_cast<uint32_t>(encodedExceptionsSize);
    }

    return size;
}

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename ExceptionCoder>
__attribute__ ((pure))
uint32_t OptPFor<BlockSize, Unpacker, TailBlockUnpacker, ExceptionCoder>::findBestB(const uint32_t *in, uint32_t nvalue) {
    uint32_t b = this->possLogs.back();
    assert(b == 32);

    uint32_t bsize = tryB(b, in, nvalue);
    const uint32_t mb = maxbits(in, in+nvalue);
    uint32_t i = 0;
    // Some schemes such as Simple16 don't code numbers greater than ((1 << 28) - 1).
    while (mb > 28 + this->possLogs[i]) 
        ++i; 

    for ( ; i < this->possLogs.size() - 1; ++i) {
        const uint32_t csize = tryB(this->possLogs[i], in, nvalue);

        if (csize < bsize) {
            b = this->possLogs[i];
            bsize = csize;
        }
    }
    return b;
}

} // namespace Codecs

#endif // CODECS_OPTPFOR_H_ 
