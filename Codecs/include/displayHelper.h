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

#ifndef DISPLAYHELPER_H_
#define DISPLAYHELPER_H_

#include "common.h"

inline std::ostream& logger(std::ostream &os = std::cerr) {
	time_t t = std::time(nullptr);
	// XXX(ot): put_time unsupported in g++ 4.7
	// return std::cerr
	//     <<  std::put_time(std::localtime(&t), "%F %T")
	//     << ": ";
	std::locale loc;
	const std::time_put<char> &tp = std::use_facet<std::time_put<char>>(loc);
	const char *fmt = "%F %T";
	tp.put(os, os, ' ', std::localtime(&t), fmt, fmt + strlen(fmt));
	return os << ": ";
}

struct stats_line {
	stats_line(std::ostream &os = std::cout) : first(true), m_os(os) {
		m_os << "{";
	}

	~stats_line() {
		m_os << "}" << std::endl;
	}

	template <typename K, typename T>
	stats_line &operator()(K const &key, T const &value) {
		if (!first) {
			m_os << ", ";
		} else {
			first = false;
		}

		emit(key);
		m_os << ": ";
		emit(value);
		return *this;
	}

private:
	template <typename T>
	void emit(T const &v) const {
		m_os << v;
	}

	// XXX properly escape strings
	void emit(const char *s) const {
		m_os << '"' << s << '"';
	}

	void emit(std::string const &s) const {
		emit(s.c_str());
	}

	bool first;
	std::ostream &m_os;
};

#endif /* DISPLAYHELPER_H_ */
