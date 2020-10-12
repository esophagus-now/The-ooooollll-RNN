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
    float cc(std::vector<float> const& x, std::vector<float> const& actual) override {
        float sum = 0.0;

        assert(x.size() == actual.size());

        for (unsigned i = 0; i < x.size(); i++) {
            float diff = x[i] - actual[i];
            sum += diff*diff;
        }

        return sum;
    }

    //get gradient
    std::vector<float> gg(std::vector<float> const& x, std::vector<float> const& actual) override {
        assert(x.size() == actual.size());
        assert(x.size() > 0);

        std::vector<float> ret(x.size(), 0.0);

        for (unsigned i = 0; i < x.size(); i++) {
            ret[i] = 2.0*(x[i] - actual[i]);
        }

        return ret;
    }
};


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

#endif