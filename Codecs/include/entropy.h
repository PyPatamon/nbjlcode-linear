/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 8, 2015
 */
/**
 * Based on code by
 *     Daniel Lemire, https://github.com/lemire/FastPFor
 * which was available under the Apache License, Version 2.0.
 */

#ifndef ENTROPY_H_
#define ENTROPY_H_

#include "Bits.h"
#include "common.h"
#include "displayHelper.h"

template <typename T>
class EntropyRecorder {
public:
    EntropyRecorder() : counter(), totalLength(0) { }

    void clear() {
        counter.clear();
        totalLength = 0;
    }

	std::ostream &display(std::ostream &os = std::cout) const {
		(stats_line (os))
				("entropy", computeShannon());
		return os;
	}

    void eat(const T *in, uint64_t n) {
        if (n == 0)
            return;

        totalLength += n;
        for (uint64_t k = 0; k < n; ++k, ++in) {
            auto i = counter.find(*in);
            if (i != counter.end())
                i->second += 1;
            else
                counter[*in] = 1;
        }
    }

    double computeShannon() const {
        double total = 0;
        for (auto i = counter.cbegin(); i != counter.cend(); ++i) {
            double p = static_cast<double>(i->second) / totalLength;
            total += p * log(p) / log(2.0);
        }
        return -total;
    }

    double computeDataBits() const {
        double total = 0;
        for (auto i = counter.cbegin(); i != counter.cend(); ++i) {
            double p = static_cast<double>(i->second) / totalLength;
            total += p * gccbits(i->first);
        }
        return total;
    }

private:
    std::unordered_map<T, uint64_t> counter;
    uint64_t totalLength;
};

#endif /* ENTROPY_H_ */
