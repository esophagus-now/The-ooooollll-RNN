#ifndef LAYER_TYPES_H
#define LAYER_TYPES_H 1

#include <vector>

struct floatvec : layer_data {
    std::vector<float> impl;

    floatvec() : layer_data("vector of floats", true) {}

    template <typename T>
    floatvec(T&& impl) : layer_data("vector of floats", true), impl(impl) {}

    floatvec* clone() const override {
        return new floatvec(*this);
    }
};

#endif