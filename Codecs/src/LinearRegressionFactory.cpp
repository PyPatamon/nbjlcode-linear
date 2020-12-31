/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *	 Created on: Feb 17, 2015
 */

#include "LinearRegressionFactory.h"
#include "IndexInfo.h"

// Due to limited memory space, we need to benchmark our LR preprocessors one at a time.
static inline LRMap initializefactory() {
	LRMap lrmap;
	for (uint64_t i = 0; i < MAXNUMLR; ++i) {
        auto p = std::shared_ptr<LR>(new LR);
		lrmap[p->name()].push_back(p);

        p = std::shared_ptr<LR>(new LRSeg<256>);
		lrmap[p->name()].push_back(p);

        p = std::shared_ptr<LR>(new SegLR<256>);
		lrmap[p->name()].push_back(p);
	}
	
	return lrmap;
}

LRMap LRFactory::slrmap = initializefactory();
