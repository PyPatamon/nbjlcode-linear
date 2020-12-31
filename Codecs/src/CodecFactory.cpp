/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 8, 2015
 */

#include "CodecFactory.h"

namespace Codecs {

// C++11 allows better than this, but neither Microsoft nor Intel support C++11 fully.
static inline CodecMap initializefactory() {
    CodecMap cmap;

	// Bit-oriented codecs.
	cmap["Rice_Hor_Scalar"] = std::shared_ptr<IntegerCodec>(new Rice_Hor_Scalar<256>);
	cmap["Rice_Ver_Scalar"] = std::shared_ptr<IntegerCodec>(new Rice_Ver_Scalar<256>);
	cmap["OptRice_Hor_Scalar"] = std::shared_ptr<IntegerCodec>(new OptRice_Hor_Scalar<256>);
	cmap["OptRice_Ver_Scalar"] = std::shared_ptr<IntegerCodec>(new OptRice_Ver_Scalar<256>);
#if CODECS_SSE_PREREQ(4, 1)
	cmap["Rice_Hor_SSE"] = std::shared_ptr<IntegerCodec>(new Rice_Hor_SSE<256>);
	cmap["Rice_Ver_SSE"] = std::shared_ptr<IntegerCodec>(new Rice_Ver_SSE<256>);
	cmap["OptRice_Hor_SSE"] = std::shared_ptr<IntegerCodec>(new OptRice_Hor_SSE<256>);
	cmap["OptRice_Ver_SSE"] = std::shared_ptr<IntegerCodec>(new OptRice_Ver_SSE<256>);
#endif /* __SSE4_1__ */
#if CODECS_AVX_PREREQ(2, 0)
	cmap["Rice_Hor_AVX"] = std::shared_ptr<IntegerCodec>(new Rice_Hor_AVX<256>);
	cmap["Rice_Ver_AVX"] = std::shared_ptr<IntegerCodec>(new Rice_Ver_AVX<256>);
	cmap["OptRice_Hor_AVX"] = std::shared_ptr<IntegerCodec>(new OptRice_Hor_AVX<256>);
	cmap["OptRice_Ver_AVX"] = std::shared_ptr<IntegerCodec>(new OptRice_Ver_AVX<256>);
#endif /* __AVX2__ */

    // Byte-aligned codecs.
    cmap["varint-G4B_Scalar"] = std::shared_ptr<IntegerCodec>(new varintG4B_Scalar);
    cmap["varint-G8B_Scalar"] = std::shared_ptr<IntegerCodec>(new varintG8B_Scalar);
    cmap["varint-G8IU_Scalar"] = std::shared_ptr<IntegerCodec>(new varintG8IU_Scalar);
    cmap["varint-G8CU_Scalar"] = std::shared_ptr<IntegerCodec>(new varintG8CU_Scalar);
#if CODECS_SSE_PREREQ(3, 1)
    cmap["varint-G4B_SSE"] = std::shared_ptr<IntegerCodec>(new varintG4B_SSE);
    cmap["varint-G8B_SSE"] = std::shared_ptr<IntegerCodec>(new varintG8B_SSE);
    cmap["varint-G8IU_SSE"] = std::shared_ptr<IntegerCodec>(new varintG8IU_SSE);
    cmap["varint-G8CU_SSE"] = std::shared_ptr<IntegerCodec>(new varintG8CU_SSE);
#endif /* __SSSE3__ */
#if CODECS_AVX_PREREQ(2, 0)
    cmap["varint-G4B_AVX"] = std::shared_ptr<IntegerCodec>(new varintG4B_AVX);
    cmap["varint-G8B_AVX"] = std::shared_ptr<IntegerCodec>(new varintG8B_AVX);
    cmap["varint-G8IU_AVX"] = std::shared_ptr<IntegerCodec>(new varintG8IU_AVX);
    cmap["varint-G8CU_AVX"] = std::shared_ptr<IntegerCodec>(new varintG8CU_AVX);
#endif /* __AVX2__ */

	// Word-aligned codecs.
	cmap["Simple-9_Scalar"] = std::shared_ptr<IntegerCodec>(new Simple9_Scalar);
	cmap["Simple-16_Scalar"] = std::shared_ptr<IntegerCodec>(new Simple16_Scalar);
#if CODECS_SSE_PREREQ(4, 1)
	cmap["Simple-9_SSE"] =  std::shared_ptr<IntegerCodec>(new Simple9_SSE);
	cmap["Simple-16_SSE"] =  std::shared_ptr<IntegerCodec>(new Simple16_SSE);
#endif /* __SSE4_1__ */
#if CODECS_AVX_PREREQ(2, 0)
	cmap["Simple-9_AVX"] = std::shared_ptr<IntegerCodec>(new Simple9_AVX);
	cmap["Simple-16_AVX"] = std::shared_ptr<IntegerCodec>(new Simple16_AVX);
#endif /* __AVX2__ */

	// Frame-based codecs.
	cmap["NewPFor_Hor_Scalar"] = std::shared_ptr<IntegerCodec>(new NewPFor_Hor_Scalar<256>);
	cmap["NewPFor_Ver_Scalar"] = std::shared_ptr<IntegerCodec>(new NewPFor_Ver_Scalar<256>);
	cmap["OptPFor_Hor_Scalar"] = std::shared_ptr<IntegerCodec>(new OptPFor_Hor_Scalar<256>);
	cmap["OptPFor_Ver_Scalar"] = std::shared_ptr<IntegerCodec>(new OptPFor_Ver_Scalar<256>);
#if CODECS_SSE_PREREQ(4, 1)
	cmap["NewPFor_Hor_SSE"] = std::shared_ptr<IntegerCodec>(new NewPFor_Hor_SSE<256>);
	cmap["NewPFor_Ver_SSE"] = std::shared_ptr<IntegerCodec>(new NewPFor_Ver_SSE<256>);
	cmap["OptPFor_Hor_SSE"] = std::shared_ptr<IntegerCodec>(new OptPFor_Hor_SSE<256>);
	cmap["OptPFor_Ver_SSE"] = std::shared_ptr<IntegerCodec>(new OptPFor_Ver_SSE<256>);
#endif /* __SSE4_1__ */
#if CODECS_AVX_PREREQ(2, 0)
	cmap["NewPFor_Hor_AVX"] = std::shared_ptr<IntegerCodec>(new NewPFor_Hor_AVX<256>);
	cmap["NewPFor_Ver_AVX"] = std::shared_ptr<IntegerCodec>(new NewPFor_Ver_AVX<256>);
	cmap["OptPFor_Hor_AVX"] = std::shared_ptr<IntegerCodec>(new OptPFor_Hor_AVX<256>);
	cmap["OptPFor_Ver_AVX"] = std::shared_ptr<IntegerCodec>(new OptPFor_Ver_AVX<256>);
#endif /* __AVX2__ */

    return cmap;
}

CodecMap CodecFactory::scodecmap = initializefactory();

} // namespace Codecs


