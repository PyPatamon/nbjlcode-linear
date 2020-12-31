/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 7, 2015
 */
/**  Based on code by
 *      Daniel Lemire, https://github.com/lemire/FastPFor
 *   which was available under the Apache License, Version 2.0.
 */

#ifndef DELTA_H_
#define DELTA_H_

#include "common.h"
#include "Portability.h"

template <typename T>
class Delta {
public:
	virtual ~Delta() = default;
	virtual void runDelta(T *data, uint64_t size) = 0;
	virtual void runPrefixSum(T *data, uint64_t size) = 0;
	virtual std::string name() const {
		return "Delta";
	};
};

template <typename T>
class RegularDelta : public Delta<T> {
public:
	virtual ~RegularDelta() = default;

	// this proves to be very fast
	virtual void runDelta(T *data, uint64_t size) {
        if (size == 0)
            throw std::runtime_error("delta coding impossible with no value!");

        for (uint64_t i = size - 1; i > 0; --i) {
            data[i] -= data[i - 1];
        }
	}

	virtual void runPrefixSum(T *data, uint64_t size) {
		for (uint64_t i = 1; i < size; ++i) {
			data[i] += data[i - 1];
		}
	}

	virtual std::string name() const {
		return "RegularDelta";
	}
};


template <typename T>
class RegularDeltaUnrolled : public RegularDelta<T> {
public:
	virtual ~RegularDeltaUnrolled() = default;

    //  Original designed by Vasily Volkov, improved by D. Lemire
	virtual void runPrefixSum(T *data, uint64_t size) {
		const uint64_t UnrollQty = 4;
        const uint64_t sz0 = size / UnrollQty * UnrollQty; // equal to 0, if size < UnrollQty
        uint64_t i = 1;
        if (sz0 > UnrollQty) {
            T a = data[0];
            for ( ; i < sz0 - UnrollQty; i += UnrollQty) {
                a = data[i] += a;
                a = data[i + 1] += a;
                a = data[i + 2] += a;
                a = data[i + 3] += a;
            }
        }

        for ( ; i != size; ++i) {
            data[i] += data[i - 1];
        }
	}

	virtual std::string name() const {
		return "RegularDeltaUnrolled";
	}
};

#if CODECS_SSE_PREREQ(2, 0)
class RegularDeltaSSE : public Delta<uint32_t> {
public:
	virtual void runDelta(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty == 0)
            throw std::runtime_error("delta coding impossible with no value!");

        for (uint64_t i = TotalQty - 1; i > 0; --i) {
            pData[i] -= pData[i - 1] + 1;
        }
	}

	virtual void runPrefixSum(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty < 4) {
			for (uint64_t i = 1; i < TotalQty; ++i) {
				pData[i] += pData[i - 1] + 1;
			}
            return;
        }

        const uint64_t Qty4 = TotalQty / 4;
        __m128i runningCount = _mm_set_epi32(3, 2, 1, 0);
		__m128i adjustment = _mm_set_epi32(4, 3, 2, 1); 
        __m128i *pCurr = reinterpret_cast<__m128i *>(pData);
        const __m128i *pEnd = pCurr + Qty4;
        while (pCurr < pEnd) {
            __m128i a0 = _mm_loadu_si128(pCurr);
            __m128i a1 = _mm_add_epi32(_mm_slli_si128(a0, 8), a0);
            __m128i a2 = _mm_add_epi32(_mm_slli_si128(a1, 4), a1);
            __m128i a3 = _mm_add_epi32(a2, runningCount);
            _mm_storeu_si128(pCurr++, a3);

            runningCount = _mm_add_epi32(_mm_shuffle_epi32(a3, 0xFF), adjustment); 
        }

        for (uint64_t i = Qty4 * 4; i < TotalQty; ++i) {
            pData[i] += pData[i - 1] + 1;
        }
	}

	virtual std::string name() const {
		return "RegularDeltaSSE";
	}
};
#endif /* __SSE2__ */

#if CODECS_AVX_PREREQ(2, 0)
class RegularDeltaAVX : public Delta<uint32_t> {
public:
	virtual void runDelta(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty == 0)
            throw std::runtime_error("delta coding impossible with no value!");

        for (uint64_t i = TotalQty - 1; i > 0; --i) {
            pData[i] -= pData[i - 1] + 1;
        }
	}

	virtual void runPrefixSum(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty < 8) {
			for (uint64_t i = 1; i < TotalQty; ++i) {
				pData[i] += pData[i - 1] + 1;
			}
            return;
        }

        const uint64_t Qty8 = TotalQty / 8;
        __m256i runningCount = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
		__m256i adjustment = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
        __m256i *pCurr = reinterpret_cast<__m256i *>(pData);
        const __m256i *pEnd = pCurr + Qty8;
        while (pCurr < pEnd) {
        	__m256i a0 = _mm256_loadu_si256(pCurr);
        	__m256i a1 = _mm256_add_epi32(_mm256_slli_si256(a0, 8), a0);
        	__m256i a2 = _mm256_add_epi32(_mm256_slli_si256(a1, 4), a1);
        	int sum = _mm256_extract_epi32(a2, 3);
        	__m256i a3 = _mm256_set_epi32(sum, sum, sum, sum, 0, 0, 0, 0);
        	__m256i a4 = _mm256_add_epi32(a2, _mm256_add_epi32(a3, runningCount));
            _mm256_storeu_si256(pCurr++, a4);

            runningCount = _mm256_add_epi32(_mm256_set1_epi32(_mm256_extract_epi32(a4, 7)), adjustment);
        }

        for (uint64_t i = Qty8 * 8; i < TotalQty; ++i) {
            pData[i] += pData[i - 1] + 1;
        }
	}

	virtual std::string name() const {
		return "RegularDeltaAVX";
	}
};
#endif /* __AVX2__ */

#if CODECS_SSE_PREREQ(2, 0)
class CoarseDelta4SSE : public RegularDelta<uint32_t> {
public:
	virtual void runDelta(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty < 4) {
        	RegularDelta<uint32_t>::runDelta(pData, TotalQty); // no need for SIMD
        	return;
        }

        const uint64_t Qty4 = TotalQty / 4;
        for (uint64_t i = 4 * Qty4; i < TotalQty; ++i) {
             pData[i] -= pData[i - 4];
        }

        __m128i *pCurr = reinterpret_cast<__m128i *>(pData) + Qty4 - 1;
        const __m128i *pStart = reinterpret_cast<__m128i *>(pData);
        __m128i a = _mm_loadu_si128(pCurr);
        while (pCurr > pStart) {
            __m128i b = _mm_loadu_si128(pCurr - 1);
            _mm_storeu_si128(pCurr-- , _mm_sub_epi32(a, b));
            a = b;
        }
     }

	virtual void runPrefixSum(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty < 4) {
        	RegularDelta<uint32_t>::runPrefixSum(pData, TotalQty); // no SIMD
            return;
        }

        const uint64_t Qty4 = TotalQty / 4;
        __m128i *pCurr = reinterpret_cast<__m128i *>(pData);
        const __m128i *pEnd = pCurr + Qty4;
        __m128i a = _mm_loadu_si128(pCurr++);
        while (pCurr < pEnd) {
            __m128i b = _mm_loadu_si128(pCurr);
            a = _mm_add_epi32(a, b);
            _mm_storeu_si128(pCurr++, a);
        }

        for (uint64_t i = Qty4 * 4; i < TotalQty; ++i) {
            pData[i] += pData[i - 4];
        }
    }

	virtual std::string name() const {
		return "CoarseDelta4SSE";
	}
};

class CoarseDelta2SSE : public RegularDelta<uint32_t> {
public:
	virtual void runDelta(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty < 4) {
        	RegularDelta<uint32_t>::runDelta(pData, TotalQty); // no need for SIMD
        	return;
        }

        const uint64_t Qty4 = TotalQty / 4;
        for (uint64_t i = TotalQty - 1; i >= 4 * Qty4; --i) {
             pData[i] -= pData[i - 2];
        }

        __m128i *pCurr = reinterpret_cast<__m128i *>(pData) + Qty4 - 1;
        const __m128i *pStart = reinterpret_cast<__m128i *>(pData);
        __m128i curr = _mm_loadu_si128(pCurr);
        while (pCurr > pStart) {
            __m128i prev = _mm_loadu_si128(pCurr - 1);
            curr = _mm_sub_epi32(curr, _mm_or_si128(_mm_slli_si128(curr, 8), _mm_srli_si128(prev, 8)));
            _mm_storeu_si128(pCurr-- , curr);

            curr = prev;
        }
     }

	virtual void runPrefixSum(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty < 4) {
        	RegularDelta<uint32_t>::runPrefixSum(pData, TotalQty); // no SIMD
            return;
        }

        const uint64_t Qty4 = TotalQty / 4;
        __m128i *pCurr = reinterpret_cast<__m128i *>(pData);
        const __m128i *pEnd = pCurr + Qty4;
        __m128i prev = _mm_loadu_si128(pCurr++);
        while (pCurr < pEnd) {
            __m128i curr = _mm_loadu_si128(pCurr);
            curr = _mm_add_epi32(curr, _mm_srli_si128(prev, 8));
            prev = _mm_add_epi32(curr, _mm_slli_si128(curr, 8));

            _mm_storeu_si128(pCurr++, prev);
        }

        for (uint64_t i = Qty4 * 4; i < TotalQty; ++i) {
            pData[i] += pData[i - 2];
        }
    }

	virtual std::string name() const {
		return "CoarseDelta2SSE";
	}
};
#endif /* __SSE2__ */

#if CODECS_AVX_PREREQ(2, 0)
class CoarseDelta8AVX : public RegularDelta<uint32_t> {
public:
	virtual void runDelta(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty < 9) {
        	RegularDelta<uint32_t>::runDelta(pData, TotalQty); // no need for SIMD
        	return;
        }

        const uint64_t Qty8 = TotalQty / 8;
        for (uint64_t i = 8 * Qty8; i < TotalQty; ++i) {
             pData[i] -= pData[i - 8];
        }

        __m256i *pCurr = reinterpret_cast<__m256i *>(pData) + Qty8 - 1;
        const __m256i *pStart = reinterpret_cast<__m256i *>(pData);
        __m256i a = _mm256_loadu_si256(pCurr);
        while (pCurr > pStart) {
            __m256i b = _mm256_loadu_si256(pCurr - 1);
            _mm256_storeu_si256(pCurr-- , _mm256_sub_epi32(a, b));
            a = b;
        }
     }

	virtual void runPrefixSum(uint32_t *pData, uint64_t TotalQty) {
        if (TotalQty < 9) {
        	RegularDelta<uint32_t>::runPrefixSum(pData, TotalQty); // no SIMD
            return;
        }

        const uint64_t Qty8 = TotalQty / 8;
        __m256i *pCurr = reinterpret_cast<__m256i *>(pData);
        const __m256i *pEnd = pCurr + Qty8;
        __m256i a = _mm256_loadu_si256(pCurr++);
        while (pCurr < pEnd) {
            __m256i b = _mm256_loadu_si256(pCurr);
            a = _mm256_add_epi32(a, b);
            _mm256_storeu_si256(pCurr++, a);
        }

        for (uint64_t i = Qty8 * 8; i < TotalQty; ++i) {
            pData[i] += pData[i - 8];
        }
    }

	virtual std::string name() const {
		return "CoarseDelta8AVX";
	}
};
#endif /* __AVX2__ */

#endif /* DELTA_H_ */
