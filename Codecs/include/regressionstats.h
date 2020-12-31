/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Mar 3, 2015
 */

#ifndef REGRESSIONSTATS_H_
#define REGRESSIONSTATS_H_

#include "common.h"
#include "displayHelper.h"
#include "IndexInfo.h"
#include "LinearRegression.h"
#include "util.h"

template <uint32_t IntervalSize>
class regressionstats {
public:
	regressionstats():
        kIntervalNum(MAXLEN / IntervalSize),
        numOfLists(kIntervalNum, 0), 
        totalListLen(kIntervalNum, 0),
        statInfo(kIntervalNum) { }

	void accumulate(uint64_t listLen, const std::vector<RegressionStatInfo_t> &segStatInfo) {
		uint64_t idx = listLen / IntervalSize;
		++numOfLists[idx];
		totalListLen[idx] += listLen;

		auto kSegNum = segStatInfo.size();
		double dSumOfRSquared = 0, dSumOfContractionRatio = 0, dSumOfBitsNeeded = 0;
		for (decltype(kSegNum) i = 0; i < kSegNum; ++i) {
			dSumOfRSquared += segStatInfo[i].dRSquared;
			dSumOfContractionRatio += segStatInfo[i].dContractionRatio;
			dSumOfBitsNeeded += segStatInfo[i].dBitsPerInt;
		}
		statInfo[idx].dRSquared += dSumOfRSquared / kSegNum;
		statInfo[idx].dContractionRatio += dSumOfContractionRatio / kSegNum;
		statInfo[idx].dBitsPerInt += dSumOfBitsNeeded;
	}

	std::ostream &display(std::ostream &os = std::cout) {
		for (uint64_t i = 0; i < kIntervalNum; ++i) {
			if (numOfLists[i]) {
				// Compute averages.
				statInfo[i].dRSquared /= numOfLists[i];
				statInfo[i].dContractionRatio /= numOfLists[i];
				statInfo[i].dBitsPerInt /= totalListLen[i];

				(stats_line (os))
						("[", i * IntervalSize)
						(")", (i + 1) * IntervalSize)
						("numOfLists", numOfLists[i])
						("totalListLen", totalListLen[i])
						("R^2", statInfo[i].dRSquared)
						("cr", statInfo[i].dContractionRatio)
						("bits/int", statInfo[i].dBitsPerInt);
			}
		}

		return os;
	}

private:
	uint64_t kIntervalNum;               // Number of intervals.
	std::vector<uint64_t> numOfLists;    // Number of lists falling into each interval.
	std::vector<uint64_t> totalListLen;
	std::vector<RegressionStatInfo_t> statInfo;
};

#endif /* REGRESSIONSTATS_H_ */
