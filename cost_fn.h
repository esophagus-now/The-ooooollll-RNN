#ifndef COST_FN_H
#define COST_FN_H

#include <vector>
#include <stdexcept>
#include <memory>
#include <utility>
#include "base_types.h"
#include "layer_types.h"

struct sqerr : cost_fn {
    //feed-forward
    float cc(MSpan<float> const& x, MSpan<float> const& actual) override {
        float sum = 0.0;

        assert(x.dims[0] == actual.dims[0]);
        assert(x.dims[1] == actual.dims[1]);

        for (int i = 0; i < x.length(); i++) {
            for (int j = 0; j < x[i].length(); j++) {
                float diff = x[i][j] - actual[i][j];
                sum += diff*diff;
            }
        }

        return sum;
    }

    //get gradient
    Matrix<float> gg(MSpan<float> const& x, MSpan<float> const& actual) override {
        assert(x.dims[0] == actual.dims[0]);
        assert(x.dims[1] == actual.dims[1]);
        assert(x.length() > 0);

        Matrix<float> ret(x.dims);

        for (int i = 0; i < x.length(); i++) {
            for (int j = 0; j < x[i].length(); j++) {
                ret[i][j] = 2.0*(x[i][j] - actual[i][j]);
            }
        }

        return ret;
    }
};

/*
struct sqerr_generalized : cost_fn_generalized {
    sqerr impl;

    //calculate cost
    //Should this always return a float?
    float cc(std::shared_ptr<layer_data const> x, std::shared_ptr<layer_data const> actual) override {
        auto fvx = dynamic_cast<floatvec const*>(x.get());

        if (!fvx) {
            throw new std::runtime_error(
                std::string("Type mismatch: got ") 
                + x->to_string() 
                + " instead of floatvec");
        }

        auto fvactual = dynamic_cast<floatvec const*>(actual.get());

        if (!fvactual) {
            throw new std::runtime_error(
                std::string("Type mismatch: got ") 
                + x->to_string() 
                + " instead of floatvec");
        }

        return impl.cc(fvx->impl, fvactual->impl);
    }

    //get gradient
    std::shared_ptr<layer_data> gg(std::shared_ptr<layer_data const> x, std::shared_ptr<layer_data const> actual) override {
        auto fvx = dynamic_cast<floatvec const*>(x.get());

        if (!fvx) {
            throw new std::runtime_error(
                std::string("Type mismatch: got ") 
                + x->to_string() 
                + " instead of floatvec");
        }

        auto fvactual = dynamic_cast<floatvec const*>(actual.get());

        if (!fvactual) {
            throw new std::runtime_error(
                std::string("Type mismatch: got ") 
                + x->to_string() 
                + " instead of floatvec");
        }

        auto ret = impl.gg(fvx->impl, fvactual->impl);
        return std::make_shared<floatvec>(std::move(ret));
    }
};
*/

#endif