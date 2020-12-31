/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Naiyong Ao, <aonaiyong@gmail.com>
 *     Created on: Dec 30, 2014
 */

#ifndef ARRAY_H_
#define ARRAY_H_

#include <cassert>
#include <cstdint>

template <typename T>
class Array {
public:
    Array() : data(nullptr), len(0) { }

    Array(uint64_t n) : data(new T[n]), len(n) { }

    Array(const Array &a) {
        reserve(a.size());
        for (uint64_t i = 0; i < len; ++i)
            data[i] = a[i];
    }

    ~Array() {
        if (data == nullptr) return;

        delete[] data;
        data = nullptr;
        len = 0;
    }

    void reserve(uint64_t n) {
        if (n <= len) return;

        T *a = new T[n];
        if (len > 0) {
            for (uint64_t i = 0; i < len; ++i) 
                a[i] = data[i];

            delete[] data;
            data = nullptr;
            len = 0;
        }

        data = a;
        len = n;
    }

    uint64_t size() const { return len; }

	T *begin() { return data; }
	const T *begin() const { return data; }

	T *end() { return data + len; }
	const T *end() const { return data + len; }

    T &operator[](uint64_t i) {
        assert(i < len);
        return data[i]; 
    }

    const T &operator[](uint64_t i) const {
        assert(i < len);
        return data[i]; 
    }

private:
    T *data = nullptr;
    uint64_t len = 0;
};

#endif /* ARRAY_H_ */
