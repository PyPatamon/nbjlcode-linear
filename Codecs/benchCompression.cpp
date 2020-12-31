/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 7, 2015
 */

#include "Array.h"
#include "common.h"
#include "compressionstats.h"
#include "CodecFactory.h"
#include "DeltaFactory.h"
#include "IndexInfo.h"
#include "IndexLoader.h"
#include "LinearRegressionFactory.h"
#include "regressionstats.h"
#include "util.h"
#include "ztimer.h"

using namespace Codecs;

template <typename container>
void verify(const container &a, const container &b) {
	auto mypair = std::mismatch(a.begin(), a.end(), b.begin());
	if (mypair.first != a.end()) {
		logger() << "First mismatching pair, "
			      << "expected: " << *mypair.first << ", "
				  << "miscalculated: " << *mypair.second << ", "
				  << "index of them: " << mypair.first - a.begin() << std::endl;
		exit(1);
	}
}

template <uint32_t runs>
void benchDeltaCompression(const std::vector<Array<uint32_t> > &docIDs, uint32_t minlen = 1) {
    WallClockTimer z;
	const uint64_t kListNum = docIDs.size();
	for (const auto &delta : DeltaFactory<uint32_t>::allSchemes()) {
		logger() << delta->name() << std::endl;
		compressionstats compstats(delta->name(), minlen);

		logger() << "Allocating dgaps array" << std::endl;
		std::vector<Array<uint32_t> > dgaps(docIDs);
		logger() << "Generating dgaps" << std::endl;
		for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
			// Progress
			if (uListIdx % (kListNum / 10) == 0)
				logger() << 100.0 * uListIdx / kListNum << "% done..." << std::endl;

			uint32_t *data = dgaps[uListIdx].begin();
			uint64_t nvalue = dgaps[uListIdx].size();

			// Check whether the list is sorted.
			if (!std::is_sorted(data, data + nvalue, std::less_equal<uint32_t>())) {
				logger() << "List #" << uListIdx << " is unsorted!" << std::endl;
				exit(1);
			}

			// Skip lists of length less than minlen.
			if (nvalue < minlen) {
				compstats.valsSkipped += nvalue;
				continue;
			}
			compstats.valsProcessed += nvalue;
			
			// Convert docIDs to dgaps.
			z.reset();
			delta->runDelta(data, nvalue);
			compstats.prepTime += static_cast<double>(z.split());

			compstats.er.eat(data, nvalue);
			compstats.histo.eatIntegers(data, nvalue);
		}

		for (const auto &codecName : CodecFactory::allNames()) { // codec->name() looks ugly.
            logger() << codecName << std::endl;

			compstats.codecs.push_back(codecName);
			std::shared_ptr<IntegerCodec> codec = CodecFactory::getFromName(codecName);

			uint64_t totalcsize = 0;
			double encodingTime = 0, decodingTime = 0;
			for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
				// Progress
				if (uListIdx % (kListNum / 10) == 0)
					logger() << 100.0 * uListIdx / kListNum << "% done..." << std::endl;

				uint64_t nvalue = dgaps[uListIdx].size();
				if (nvalue < minlen) // Skip lists of length less than minlen.
					continue;

				// Encoding...
                Array<uint32_t> compressedBuf(4 * nvalue);
				uint64_t csize = 0; // How many 32-bit words consumed.
                z.reset();
                codec->encodeArray(dgaps[uListIdx].begin(), nvalue, compressedBuf.begin(), csize);
                encodingTime += static_cast<double>(z.split());
				totalcsize += csize;

				// Decoding...
                Array<uint32_t> recoveryBuf(nvalue); // We forbid overshooting during decompression.
                double totalDecodingTime = 0;
                for (uint32_t run = 0; run <= runs; ++run) {
                    z.reset();
                    codec->decodeArray(compressedBuf.begin(), csize, recoveryBuf.begin(), nvalue); // We rely on nvalue to decode.
                    double elapsed = static_cast<double>(z.split());
                    if (run)
                        totalDecodingTime += elapsed;
                }
                decodingTime += totalDecodingTime / runs;

				// Verify encoding and decoding.
				verify(dgaps[uListIdx], recoveryBuf);
			}
			logger() << std::endl;

			compstats.encodingTime.push_back(encodingTime);
			compstats.decodingTime.push_back(decodingTime);
			compstats.totalcsize.push_back(totalcsize);
		}

		// Reconstruct docIDs from dgaps.
		logger() << "Verifying delta and prefixsum" << std::endl;
		for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
			// Progress
			if (uListIdx % (kListNum / 10) == 0)
				logger() << 100.0 * uListIdx / kListNum << "% done..." << std::endl;

			uint32_t *data = dgaps[uListIdx].begin();
			uint64_t nvalue = dgaps[uListIdx].size();

			if (nvalue < minlen)
				continue;

			z.reset();
			delta->runPrefixSum(data, nvalue);
			compstats.postpTime += static_cast<double>(z.split());

			verify(docIDs[uListIdx], dgaps[uListIdx]);
		}
		
		logger() << "Displaying compression statistics" << std::endl << std::endl;
		compstats.display();
	}
}

template <uint32_t runs>
void benchLRCompression(const std::vector<Array<uint32_t> > &docIDs, uint32_t minlen = 1) {
    WallClockTimer z;
	const uint64_t kListNum = docIDs.size();
	for (const auto &LR : LRFactory::allSchemes()) {
		logger() << LR[0]->name() << std::endl;
		compressionstats compstats(LR[0]->name());
		regressionstats<100 * 1000> regstats; // Default interval size is 100 * 1000

		logger() << "Allocating VDs array" << std::endl;
		std::vector<Array<uint32_t> > VDs(docIDs);
		logger() << "Generating VDs" << std::endl;
		uint64_t uLRProcessedListIdx = 0;
		for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
			// Progress.
			if (uListIdx % (kListNum / 10) == 0)
				logger() << 100.0 * uListIdx / kListNum << "% done..." << std::endl;

			uint32_t *data = VDs[uListIdx].begin();
			uint64_t nvalue = VDs[uListIdx].size();

			// Check whether the list is sorted.
			if (!std::is_sorted(data, data + nvalue, std::less_equal<uint32_t>())) {
				logger() << "List #" << uListIdx << " is unsorted!" << std::endl;
				exit(1);
			}

			// Skip lists of length less than minlen.
			if (nvalue < minlen) {
				compstats.valsSkipped += nvalue;
				continue;
			}
			compstats.valsProcessed += nvalue;

			// Convert docIDs to VDs.
			z.reset();
			LR[uLRProcessedListIdx]->runConversion(data, nvalue);
			compstats.prepTime += static_cast<double>(z.split());

			compstats.er.eat(data, nvalue);
			compstats.histo.eatIntegers(data, nvalue);
			regstats.accumulate(nvalue, LR[uLRProcessedListIdx]->statInfo);

			++uLRProcessedListIdx;
		}
		assert(uLRProcessedListIdx < MAXNUMLR);
		regstats.display();

		for (const auto &codecName : CodecFactory::allNames()) { // codec->name() looks ugly
			logger() << codecName << std::endl;

			std::shared_ptr<IntegerCodec> codec = CodecFactory::getFromName(codecName);
			compstats.codecs.push_back(codecName);

			uint64_t totalcsize = 0;
			double encodingTime = 0, decodingTime = 0;
			uLRProcessedListIdx = 0;
			for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
				// Progress.
				if (uListIdx % (kListNum / 10) == 0)
					logger() << 100.0 * uListIdx / kListNum << "% done..." << std::endl;

				uint64_t nvalue = VDs[uListIdx].size();
				if (nvalue < minlen) // Skip lists of length less than minlen.
					continue;

				// Encoding...
                Array<uint32_t> compressedBuf(4 * nvalue);
				uint64_t csize = 0; // How many 32-bit words consumed.
				z.reset();
				codec->encodeArray(VDs[uListIdx].begin(), nvalue, compressedBuf.begin(), csize);
                encodingTime += static_cast<double>(z.split());
				totalcsize += csize;
				totalcsize += LR[uLRProcessedListIdx++]->sizeOfAuxiliaryInfo();

				// Decoding...
                Array<uint32_t> recoveryBuf(nvalue);
                double totalDecodingTime = 0;
                for (uint32_t run = 0; run <= runs; ++run) {
                    z.reset();
                    codec->decodeArray(compressedBuf.begin(), csize, recoveryBuf.begin(), nvalue); // We rely on nvalue to decode.
                    double elapsed = static_cast<double>(z.split());
                    if (run)
                        totalDecodingTime += elapsed;
                }
                decodingTime += totalDecodingTime / runs;

				// Verify encoding and decoding.
				verify(VDs[uListIdx], recoveryBuf);
			}
			logger() << std::endl;

			compstats.encodingTime.push_back(encodingTime);
			compstats.decodingTime.push_back(decodingTime);
			compstats.totalcsize.push_back(totalcsize);
		}

		// Reconstruct docIDs from VDs.
		logger() << "Verifying converting and reconstructing" << std::endl;
		uLRProcessedListIdx = 0;
		for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
			// Progress.
			if (uListIdx % (kListNum / 10) == 0)
				logger() << 100.0 * uListIdx / kListNum << "% done..." << std::endl;

			uint32_t *data = VDs[uListIdx].begin();
			uint64_t nvalue = VDs[uListIdx].size();

			if (nvalue < minlen)
				continue;

			z.reset();
			LR[uLRProcessedListIdx]->runReconstruction(data, nvalue);
			compstats.postpTime += static_cast<double>(z.split());

			++uLRProcessedListIdx;

			verify(docIDs[uListIdx], VDs[uListIdx]);
		}

		logger() << "Displaying compression statistics" << std::endl << std::endl;
		compstats.display();
	}
}

int main(int argc, char **argv) {
	if (argc < 4) {
		logger() << "Run as './benchCompression index_dir dataset preprocessor" << std::endl;
		return -1;
	}

	// Load index.
	const std::string indexDir = argv[1];
	const std::string dataset = argv[2];
	logger() << "Processing " << dataset << std::endl;
	IndexLoader idxLoader(indexDir, dataset);
	idxLoader.loadIndex();

	std::string preprocessor = argv[3];
    uint32_t minlen = 1;
    if (argc > 4) {
        minlen = std::atoi(argv[4]);
    }

	if (preprocessor == "Delta") {
		logger() << "benchmarking Delta Compression" << std::endl;
		benchDeltaCompression<3>(idxLoader.postings, minlen);
	}
	else if (preprocessor == "LR") {
		logger() << "benchmarking LR Compression" << std::endl;
		benchLRCompression<3>(idxLoader.postings, minlen);
	}
	else { // run both
		logger() << "benchmarking Delta Compression" << std::endl;
		benchDeltaCompression<3>(idxLoader.postings, minlen);
		logger() << "benchmarking LR Compression" << std::endl;
		benchLRCompression<3>(idxLoader.postings, minlen);
	}

	return 0;
}
