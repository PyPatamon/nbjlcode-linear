/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Apr 09, 2016
 */
/**
 * Based on code by
 *     Daniel Lemire, https://github.com/lemire/FastPFor
 * which was available under the Apache License, Version 2.0.
 */

#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#include "Bits.h"
#include "common.h"
#include "displayHelper.h"

template <typename T>
class BitWidthHistoGram {
public:
    BitWidthHistoGram() : histo(33, 0) { }

	std::ostream &display(std::ostream &os = std::cout) {
        double sum = std::accumulate(histo.begin(), histo.end(), 0.0);
        if(sum == 0)
        	return os;

        stats_line statsline(os);
        for (uint64_t b = 0; b < histo.size(); ++b) {
        	if (histo[b])
        		statsline("b"+std::to_string(b), histo[b] / sum);
        }

        return os;
    }

    void eatIntegers(const T *data, uint64_t size) {
        for (uint32_t i = 0; i < size; ++i) {
			++histo[gccbits(data[i])];
        }
    }

    void eatDGaps(const T *data, uint64_t size) {
        if (size <= 1)
            return;

        for (uint64_t i = 0; i < size - 1; ++i) {
            assert(data[i + 1] > data[i]);
            T gap = data[i + 1] - data[i] - 1;
            ++histo[gccbits(gap)];
        }
    }

private:
    std::vector<uint64_t> histo;
};

#endif /* HISTOGRAM_H_ */
