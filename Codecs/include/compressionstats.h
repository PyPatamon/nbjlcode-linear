/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 9, 2015
 */

#ifndef COMPRESSIONSTATS_H_
#define COMPRESSIONSTATS_H_

#include "common.h"
#include "displayHelper.h"
#include "Delta.h"
#include "entropy.h"
#include "Histogram.h"
#include "IntegerCodec.h"
#include "util.h"

class compressionstats {
public:
	compressionstats(const std::string &prep = "", uint32_t minlen = 1) :
		preprocessor(prep), minlength(minlen), er(), histo(), prepTime(0), prepSpeed(0), postpTime(0), postpSpeed(0),
		codecs(), valsProcessed(0), valsSkipped(0), totalcsize(), bitsPerInt(),
		encodingTime(), encodingSpeed(), decodingTime(), decodingSpeed(),
		compTime(), compSpeed(), decompTime(), decompSpeed(),
		unitOfEffectiveness("bits/int"), unitOfTime("us"), unitOfSpeed("million ints/s"),
		isSummarized(false), isNormalized(false) { }

	std::ostream &display(std::ostream &os = std::cout) {
		if (!isNormalized) {
			normalize();
		}

        (stats_line(os))
        		("preprocessor", preprocessor)
				("minlen", minlength)
        		("entropy", er.computeShannon())
				("prepTime", prepTime)
				("prepSpeed", prepSpeed)
				("postpTime", postpTime)
				("postpSpeed", postpSpeed)
				;
        histo.display(os);

		auto n = codecs.size();
		for (decltype(n) i = 0; i != n; ++i) {
                (stats_line(os))
                		("codec", codecs[i])
                		("bits/int", bitsPerInt[i])
						("encodingTime", encodingTime[i])
						("encodingSpeed", encodingSpeed[i])
						("decodingTime", decodingTime[i])
						("decodingSpeed", decodingSpeed[i])
						;
		}

		return os;
	}

	std::string preprocessor;        // Preprocessor (e.g. "RegularDeltaSSE", "SegLR<256>")
	uint32_t minlength;
	EntropyRecorder<uint32_t> er;
	BitWidthHistoGram<uint32_t> histo;
	double prepTime;                 // Time of converting docIDs to dgaps or VDs.
	double postpTime;                // Time of reconstructing docIDs from dgaps or VDs.

	std::vector<std::string> codecs;
	uint64_t valsProcessed;
	uint64_t valsSkipped;             // inputSize = valsProcessed + valsSkipped
	std::vector<uint64_t> totalcsize; // outputSize = totalcsize + valsSkipped
	std::vector<double> bitsPerInt;   // := (32.0 * outputSize) / inputSize
	std::vector<double> encodingTime;
	std::vector<double> decodingTime;
	std::vector<double> compTime;     // := prepTime + encodingTime
	std::vector<double> decompTime;   // := decodingTime + postpTime

private:
	void summarize() {
		if (isSummarized) {
			return;
		}

		prepSpeed = valsProcessed / prepTime;
		postpSpeed = valsProcessed / postpTime;

		auto n = codecs.size();
		for (decltype(n) i = 0; i != n; ++i) {
			double compressionRatio = static_cast<double>(valsProcessed + valsSkipped) / (totalcsize[i] + valsSkipped);
			bitsPerInt.push_back(32 / compressionRatio);

			compTime.push_back(prepTime + encodingTime[i]);
			decompTime.push_back(decodingTime[i] + postpTime);

			encodingSpeed.push_back(valsProcessed / encodingTime[i]);
			decodingSpeed.push_back(valsProcessed / decodingTime[i]);
			compSpeed.push_back(valsProcessed / compTime[i]);
			decompSpeed.push_back(valsProcessed / decompTime[i]);
		}

		isSummarized = true;
	}

	/**
	 * us -> ms
	 */
	static void timeNormalizer(double &time) {
		time /= 1000;
	}

	void normalize() {
		if (isNormalized) {
			return;
		}

		if (!isSummarized) {
			summarize();
		}

		timeNormalizer(prepTime);
		timeNormalizer(postpTime);
		auto n = codecs.size();
		for (decltype(n) i = 0; i != n; ++i) {
			timeNormalizer(encodingTime[i]);
			timeNormalizer(decodingTime[i]);
			timeNormalizer(compTime[i]);
			timeNormalizer(decompTime[i]);
		}

		unitOfTime = "ms";
		isNormalized = true;
	}

	double prepSpeed;
	double postpSpeed;
	std::vector<double> encodingSpeed;
	std::vector<double> decodingSpeed;
	std::vector<double> compSpeed;
	std::vector<double> decompSpeed;

	std::string unitOfEffectiveness;
	std::string unitOfTime;
	std::string unitOfSpeed;

	bool isSummarized;
	bool isNormalized;
};

#endif /* COMPRESSIONSTATS_H_ */
