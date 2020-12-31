/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 13, 2015
 */

#ifndef LINEARREGRESSION_H_
#define LINEARREGRESSION_H_

#include "Bits.h"
#include "common.h"
#include "displayHelper.h"
#include "util.h"

/**
 * To reduce both computation and space cost, we don't keep the minimum.
 * To that end, we have to represent the y-intercept as an int. For the 
 * reason, let's consider an input array (or block) with only one value.
 * If that value has too many digits to be represented precisely as a 
 * float, we have to keep the minimum, since otherwise the vertical 
 * deviation might be -1.
 */
struct RegressionInfo_t {
	RegressionInfo_t() : fSlope(0), iIntercept(0) { }

	float fSlope;
	int iIntercept;
};

struct RegressionStatInfo_t {
	RegressionStatInfo_t() : dRSquared(0), dContractionRatio(0), uSearchRange(0), dBitsPerInt(0) { }

	std::ostream &display(std::ostream &os = std::cout) {
        (stats_line(os))
            ("R^2", dRSquared)
            ("cr", dContractionRatio)
            ("bits/int", dBitsPerInt)
            ;
		return os;
	}

	double dRSquared;
	double dContractionRatio;
	uint32_t uSearchRange;
	double dBitsPerInt;
};


/**
 * Since we represent the y-intercept as an int, our input values
 * should fit in int, which is true for both baidu and gov2 datasets.
 */
class LR {
public:
	LR() : regressionInfo(1), statInfo(1) {
		regressionInfo.shrink_to_fit();
		statInfo.shrink_to_fit();
	}

	virtual ~LR() = default;

	virtual std::string name() const {
        std::ostringstream lrname;
        std::string platform;
#ifdef __SSE2__
        platform = "SSE";
#else  /* !__SSE2__ */
#ifdef __AVX2__
        platform = "AVX";
#else  /* !__SSE2__ && !__AVX2__ */
        platform = "Scalar";
#endif  /* __AVX2__ */
#endif  /* __SSE2__ */
        lrname << "LR_" << platform;
		return lrname.str();
	}

	/**
	 * We take into account the size of auxiliary info when computing compression ratio.
	 * For LR and SegLR, this function returns size of regression info.
	 */
	virtual uint64_t sizeOfAuxiliaryInfo() const {
		return sizeof(RegressionInfo_t) * regressionInfo.size();
	}

	static inline void getRegressionInfo(const uint32_t *data, uint64_t size,
			RegressionInfo_t &regInfo, RegressionStatInfo_t &stInfo);

	/**
	 * For x = 0...size-1, calculate vertical deviation (possibly negative):
	 * iVDs[x] = data[x] - static_cast<int>(fSlope*x) - iIntercept.
	 */
	template <bool IsTailBlock>
	static void getiVDs(const uint32_t *data, uint64_t size, 
			const RegressionInfo_t &regInfo, int *iVDs);

	/**
	 * For x = 0...size-1, calculate non-negative vertical deviation:
	 * uVDs[x] = VDs[x] - minimum, where minimum = min{VDs[x]|x=0...size-1}.
	 * Note that this can be performed either globally or locally, which is
	 * reflected through T (int - globally, uint32_t - locally).
	 */
	template <bool IsTailBlock, typename T>
	static void getuVDs(const T *VDs, uint64_t size,
			T &minimum, uint32_t *uVDs, double &dBitsPerInt);

	/**
	 * This is to reduce the space cost and computation cost
	 * of reconstructing docIDs from the corresponding VDs.
	 */
	static void shiftLine(int &iIntercept, int iMin) {
		iIntercept += iMin;
	}

	/**
	 * Convert input data to non-negative vertical deviations.
	 * This happens in place.
	 */
	template <bool IsTailBlock>
	static void convert(uint32_t *data, uint64_t size, 
			RegressionInfo_t &regInfo, RegressionStatInfo_t &stInfo);

	/**
	 * Reconstruct original input data from corresponding vertical deviations.
	 * This also happens in place.
	 */
	template <bool IsTailBlock>
	static void reconstruct(uint32_t *data, uint64_t size, 
			const RegressionInfo_t &regInfo);


	virtual void runConversion(uint32_t *data, uint64_t size) {
		convert<true>(data, size, regressionInfo[0], statInfo[0]);
	}

	virtual void runReconstruction(uint32_t *data, uint64_t size) {
		reconstruct<true>(data, size, regressionInfo[0]);
	}

	std::vector<RegressionInfo_t> regressionInfo;
	std::vector<RegressionStatInfo_t> statInfo;
};


template <uint32_t BlockSize>
class LRSeg : public LR {
public:
	LRSeg() : localMins() { }

	virtual ~LRSeg() = default;

	virtual std::string name() const {
        std::ostringstream lrname;
        std::string platform;
#ifdef __SSE2__
        platform = "SSE";
#else  /* !__SSE2__ */
#ifdef __AVX2__
        platform = "AVX";
#else  /* !__SSE2__ && !__AVX2__ */
        platform = "Scalar";
#endif  /* __AVX2__ */
#endif  /* __SSE2__ */
        lrname << "LRSeg<" << BlockSize << ">_" << platform;
		return lrname.str();
	}

	/**
	 * For LRSeg, returns size of regression info plus local minimums.
	 */
	virtual uint64_t sizeOfAuxiliaryInfo() const {
		return sizeof(RegressionInfo_t) * regressionInfo.size() +
			   sizeof(uint32_t) * localMins.size();
	}

	template <bool IsTailBlock>
	static void reconstruct(uint32_t *data, uint64_t startx, uint64_t endx,
			const RegressionInfo_t &regInfo);
	
	virtual void runConversion(uint32_t *data, uint64_t size) {
		convert<true>(data, size, regressionInfo[0], statInfo[0]);

		statInfo[0].dBitsPerInt = 0; // clear since we're gonna recompute it

		// get local minimums of uVDs and reduce each uVDs 
		// from corresponding local minimum
		uint32_t *uVDs = data;

		uint64_t kBlockNum = div_roundup(size, BlockSize);
		localMins.resize(kBlockNum);
	    localMins.shrink_to_fit();
		for (uint64_t i = 0; i < kBlockNum - 1; ++i) {
			getuVDs<false>(uVDs, BlockSize, localMins[i], uVDs, statInfo[0].dBitsPerInt);
			uVDs += BlockSize;
		}
		uint64_t tailBlockSize = size - (kBlockNum - 1) * BlockSize;
		getuVDs<true>(uVDs, tailBlockSize, localMins[kBlockNum - 1], uVDs, statInfo[0].dBitsPerInt);
	}

	virtual void runReconstruction(uint32_t *data, uint64_t size) {
		RegressionInfo_t regInfo(regressionInfo[0]);
		int iIntercept = regressionInfo[0].iIntercept;

		uint64_t kBlockNum = div_roundup(size, BlockSize);
		uint64_t thisBlockOffset = 0;
		for (uint64_t i = 0; i < kBlockNum - 1; ++i) {
			regInfo.iIntercept = iIntercept + localMins[i];
			reconstruct<false>(data, thisBlockOffset, thisBlockOffset + BlockSize, regInfo);
			thisBlockOffset += BlockSize;
		}
	
		regInfo.iIntercept = iIntercept + localMins[kBlockNum - 1]; 
		reconstruct<true>(data, thisBlockOffset, size, regInfo);
	}

	std::vector<uint32_t> localMins;
};


template <uint32_t BlockSize>
class SegLR : public LR {
public:
	virtual ~SegLR() = default;

	virtual std::string name() const {
        std::ostringstream lrname;
        std::string platform;
#ifdef __SSE2__
        platform = "SSE";
#else  /* !__SSE2__ */
#ifdef __AVX2__
        platform = "AVX";
#else  /* !__SSE2__ && !__AVX2__ */
        platform = "Scalar";
#endif  /* __AVX2__ */
#endif  /* __SSE2__ */
        lrname << "SegLR<" << BlockSize << ">_" << platform;
		return lrname.str();
	}

	virtual void runConversion(uint32_t *data, uint64_t size) {
		uint64_t kBlockNum = div_roundup(size, BlockSize);
		regressionInfo.resize(kBlockNum);
		statInfo.resize(kBlockNum);
		regressionInfo.shrink_to_fit();
		statInfo.shrink_to_fit();

		for (uint64_t i = 0; i < kBlockNum - 1; ++i) {
			convert<false>(data, BlockSize, regressionInfo[i], statInfo[i]);
			data += BlockSize;
		}

		uint64_t tailBlockSize = size - (kBlockNum - 1) * BlockSize;
		convert<true>(data, tailBlockSize, regressionInfo[kBlockNum - 1], statInfo[kBlockNum - 1]);
	}

	virtual void runReconstruction(uint32_t *data, uint64_t size) {
		uint64_t kBlockNum = div_roundup(size, BlockSize);
		for (uint64_t i = 0; i < kBlockNum - 1; ++i) {
			reconstruct<false>(data, BlockSize, regressionInfo[i]);
			data += BlockSize;
		}

		uint64_t tailBlockSize = size - (kBlockNum - 1) * BlockSize;
		reconstruct<true>(data, tailBlockSize, regressionInfo[kBlockNum - 1]);
	}
};

#include "LinearRegressionIMP.h"

#endif /* LINEARREGRESSION_H_ */
