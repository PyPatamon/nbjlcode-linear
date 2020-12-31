/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 8, 2015
 */

#ifndef INDEXLOADER_H_
#define INDEXLOADER_H_

#include "Array.h"
#include "common.h"
#include "displayHelper.h"
#include "IndexInfo.h"
#include "util.h"

class IndexLoader {
public:
	IndexLoader(const std::string &indexDir, const std::string &dataset) :
        dictPath(indexDir + dataset + ".ind1"),
		postingsPath(indexDir + dataset + ".ind2") {
			struct stat buf;
			stat(dictPath.c_str(), &buf);
			kListNum = buf.st_size / sizeof(dictionary_t);
			assert(kListNum < MAXNUM);
        }

    void loadDictionary() {
        FILE *fdict = fopen(dictPath.c_str(), "rb");
        if(!fdict) {
            logger() << "Dictionary doesn't exist!!!" << std::endl;
            exit(1);
        }

        dictionary.resize(kListNum);
        fread(&dictionary[0], sizeof(dictionary_t), kListNum, fdict);

        fclose(fdict);
    }

    void loadPostings() {
        if (dictionary.empty()) {
            logger() << "Please load dictionary first!!!" << std::endl;
            exit(1);
        }

        FILE *fpostings = fopen(postingsPath.c_str(), "rb");
        if(!fpostings) {
            logger() << "Postings doesn't exist!!!" << std::endl;
            exit(1);
        }

        postings.resize(kListNum);
        for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
            uint32_t length = dictionary[uListIdx].length;
            uint64_t offset = dictionary[uListIdx].offset;

            postings[uListIdx].reserve(length);
            fseek(fpostings, offset, SEEK_SET);
            fread(postings[uListIdx].begin(), sizeof(uint32_t), length, fpostings);
        }

        fclose(fpostings);
    }

	void loadIndex() {
		logger() << "Loading dictionary..." << std::endl;
		loadDictionary();
        logger() << "Loading postings lists..." << std::endl;
		loadPostings();
		logger() << "Loading finished" << std::endl;
	}

	std::string dictPath;
	std::string postingsPath;
	uint64_t kListNum = 0;
	std::vector<dictionary_t> dictionary;
	std::vector<Array<uint32_t> > postings;
};

#endif /* INDEXLOADER_H_ */
