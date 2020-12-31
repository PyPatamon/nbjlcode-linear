/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 17, 2015
 */

#ifndef INDEXINFO_H_
#define INDEXINFO_H_

#include <cstdint>

struct dictionary_t {
	uint32_t length;  // Length of postings list (how many 32-bit integers).
	uint64_t offset;  // Offset of postings list (how many bytes).
};

enum {
	MAXNUM = 19000000,  // Upper limit on number of postings lists.
	MAXNUMLR = 3000000, // upper limit on number of postings lists of length > MINLENLR.

	MINLENDELTA = 1,   // Skip lists of length < MINLENDELTA during d-gap based compression.
	MINLENLR = 5,      // Skip lists of length < MINLENLR during Linear Regression Compression.
	MAXLEN = 12000000  // Upper limit on length of postings lists.
};

#endif /* INDEXINFO_H_ */
