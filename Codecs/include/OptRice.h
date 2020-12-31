/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Jan 18, 2015
 */

#ifndef CODECS_OPTRICE_H_
#define CODECS_OPTRICE_H_ 

#include "Rice.h"

namespace Codecs {

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
class OptRice;

template <uint32_t BlockSize>
using OptRice_Hor_Scalar = OptRice<BlockSize, HorUnpacker<Scalar, true>, HorUnpacker<Scalar, true>, Unary<true> >;

#if CODECS_SSE_PREREQ(4, 1)
template <uint32_t BlockSize>
using OptRice_Hor_SSE = OptRice<BlockSize, HorUnpacker<SSE, true>, HorUnpacker<SSE, true>, Unary<true> >;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
template <uint32_t BlockSize>
using OptRice_Hor_AVX = OptRice<BlockSize, HorUnpacker<AVX, true>, HorUnpacker<AVX, true>, Unary<true> >;
#endif /* __AVX2__ */


template <uint32_t BlockSize>
using OptRice_Ver_Scalar = OptRice<BlockSize, VerUnpacker<Scalar, true>, HorUnpacker<Scalar, true>, Unary<true> >;

#if CODECS_SSE_PREREQ(4, 1)
template <uint32_t BlockSize>
using OptRice_Ver_SSE = OptRice<BlockSize, VerUnpacker<SSE, true>, HorUnpacker<SSE, true>, Unary<true> >;
#endif /* __SSE4_1__ */

#if CODECS_AVX_PREREQ(2, 0)
template <uint32_t BlockSize>
using OptRice_Ver_AVX = OptRice<BlockSize, VerUnpacker<AVX, true>, HorUnpacker<AVX, true>, Unary<true> >;
#endif /* __AVX2__ */

template <uint32_t BlockSize, typename Unpacker, typename TailBlockUnpacker, typename UnaryCoder>
class OptRice : public Rice<BlockSize, Unpacker, TailBlockUnpacker, UnaryCoder> {
private:
    virtual std::string basecodecname() const { return "OptRice"; }

	virtual uint32_t findBestB(const uint32_t *in, uint32_t nvalue) {
        uint32_t b = 32;
        uint64_t bsize = nvalue;

        for (uint32_t c = 0; c < 32; ++c) {
            for (uint32_t i = 0; i < nvalue; ++i) {
                this->quotient[i] = in[i] >> c;
                this->remainder[i] = in[i] & ((1U << c) - 1);
            }

            uint64_t csize = 0;
            this->ucoder.fakeencodeArray(&this->quotient[0], nvalue, csize);
            csize += div_roundup(nvalue * c, 32);
            if (csize < bsize) {
                b = c;
                bsize = csize;
            }
        }

        return b;
    }
};

} // namespace Codecs

#endif // CODECS_OPTRICE_H_ 
