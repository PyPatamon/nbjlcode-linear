/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Feb 4, 2015
 */
/**  Based on code by
 *      Daniel Lemire, https://github.com/lemire/FastPFor
 *   which was available under the Apache License, Version 2.0.
 */

#ifndef COMMON_H_
#define COMMON_H_

// C headers (sorted)
#include <fcntl.h>
#ifndef _WIN32
# include <sys/mman.h>
# include <sys/resource.h>
#endif
#include <sys/stat.h>
#ifndef _WIN32
# include <sys/time.h>
#endif
#include <sys/types.h>
#include <time.h>

// C++ headers (sorted)
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _MSC_VER
# include <iso646.h>
# include <stdint.h>

# define __attribute__(n)
# define __restrict__
# define constexpr inline
#endif

#endif /* COMMON_H_ */
