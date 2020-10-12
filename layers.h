#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <stdexcept>
#include <memory>
#include <utility>

#include "base_types.h"
#include "matrix.h"
#include "layer_types.h"

struct simple_linear : layer {
    float a;

    simple_linear() : a(1.0) {}
    simple_linear(float a) : a(a) {}

    //feed-forward
    std::vector<float> ff(std::vector<float> const& x) override {
        std::vector<float> ret(x);
        for (float &reti : ret) reti *= a;
        return ret;
    }

    //backprop
    virtual std::vector<float> bp(std::vector<float> const& x, std::vector<float> const& dy, float lr) override {
        std::vector<float> ret(dy);

        assert(x.size() == dy.size());

        float sum = 0.0;
        for (unsigned i = 0; i < dy.size(); i++) {
            sum += x[i]*dy[i];
            ret[i] *= a;
        }

        sum *= lr;
        a -= sum;

        #ifdef DEBUG
        std::cout << "\tAdjustment = " << -sum << std::endl;
        std::cout << "\t a is now " << a << std::endl;
        #endif

        return ret;
    }
};

//fully connected
struct fc : layer {
    unsigned r, c;
    Matrix<float> mat;
    std::vector<float> bias;
    activation_fn* act_fn;

    fc(unsigned n_out, unsigned n_in, activation_fn* act_fn) :  
            r(n_out), c(n_in), mat(n_out, n_in), act_fn(act_fn) {
        assert(n_in > 0);
        assert(n_out > 0);
        //mat = Matrix<float>(r,c);
        bias = std::vector<float>(r);

        auto gen = std::default_random_engine();
        auto dist = std::uniform_real_distribution<float>(-1.0, 1.0);

        //Start with el dumbo identity matrix
        for (unsigned i = 0; i < r; i++) {
            for (unsigned j = 0; j < c; j++) {
                mat[i][j] = dist(gen);
            }
            bias[i] = dist(gen);
        }
    }

    //feed-forward
    std::vector<float> ff(std::vector<float> const& x) override {
        assert(x.size() == c);
        std::vector<float> ret(bias);
        for (unsigned i = 0; i < r; i++) {

            for (unsigned j = 0; j < c; j++) {
                ret[i] += mat[i][j]*x[j];
            }

            ret[i] = (*act_fn)(ret[i]);
        }
        return ret;
    }

    //backprop
    virtual std::vector<float> bp(std::vector<float> const& x, std::vector<float> const& dy, float lr) override {
        assert(x.size() == c);
        assert(dy.size() == r);

        std::vector<float> ret(c, 0.0);

        std::vector<float> z = ff(x);
        std::vector<float> z2(z.size());
        float sum_sq = 0.0;
        for (unsigned i = 0; i < z.size(); i++) {
            z2[i] = dy[i] * (*act_fn)[z[i]];
            sum_sq += z2[i]*z2[i];
        }

        //z2 now contains elementwise_product(dy, sech^2(Ax + b))

        for (unsigned j = 0; j < c; j++) {
            for (unsigned i = 0; i < r; i++) {
                ret[j] += mat[i][j] * z2[i];
            }
        }

        //dErr/dAij = (dyi/dAij * dErr/dyi)
        for (unsigned j = 0; j < c; j++) {
            for (unsigned i = 0; i < r; i++) {
                mat[i][j] -= lr*z2[i]*x[j];
            }
            bias[j] -= lr * z2[j];
        }

        return ret;
    }
};

struct fc_generalized : layer_generalized {
    fc impl;

    fc_generalized(unsigned n_out, unsigned n_in, activation_fn* act_fn) :  
        impl(n_out, n_in, act_fn) 
    {}

    //feed-forward
    std::shared_ptr<layer_data> ff(std::shared_ptr<layer_data const> x) override {
        auto fv = dynamic_cast<floatvec const*>(x.get());
        if (!fv) {
            throw new std::runtime_error(
                std::string("Type mismatch: got ") 
                + x->to_string() 
                + " instead of floatvec");
        }

        auto ret = impl.ff(fv->impl);
        return std::make_shared<floatvec>(std::move(ret));
    }

    //backprop
    std::shared_ptr<layer_data> bp(std::shared_ptr<layer_data const> x, std::shared_ptr<layer_data const> dy, float lr) override {
        auto fvx = dynamic_cast<floatvec const*>(x.get());
        if (!fvx) {
            throw new std::runtime_error(
                std::string("Type mismatch: got ") 
                + x->to_string() 
                + " instead of floatvec");
        }
        
        auto fvdy = dynamic_cast<floatvec const*>(dy.get());
        if (!fvdy) {
            throw new std::runtime_error(
                std::string("Type mismatch: got ") 
                + dy->to_string() 
                + " instead of floatvec");
        }

        auto ret = impl.bp(fvx->impl, fvdy->impl, lr);
        return std::make_shared<floatvec>(std::move(ret));
    }
};

struct cnn : layer {
    //Given dimensions
    int Iw, Ih, Id; //Input dimensions
    int Kw, Kh; //Kernel dimensions (third and fourth dim are Id and Od)
    int s; //Stride
    int Od; //Output depth
    //The output width and height are forced based on these input sizes, 
    //and are calculated in the compute_sizes() function

    //Calculated dimensions
    int Ow, Oh;

    bool compute_sizes() {
        if (Iw*Ih <= 0) {
            throw std::runtime_error("Input must have positive number of elements");
        }

        if (s <= 0) {
            throw std::runtime_error("Stride must have positive size");
        }

        if (Kw*Kh <= 0) {
            throw std::runtime_error("Kernel must have positive number of elements");
        }

        if ((Iw - (Kw - 1)) % s != 0 || (Ih - (Kh - 1)) % s != 0) {
            throw std::runtime_error("Input sizes must be exact multiple of stride");
        }

        Ow = (Iw - (Kw - 1)) / s;
        Oh = (Ih - (Kh - 1)) / s;
    }

    cnn(
        int Iw, int Ih, int Id,
        int Kw, int Kh,
        int s
    ) :
        Iw(Iw), Ih(Ih), Id(Id),
        Kw(Kw), Kh(Kh),
        s(s)
    {
        compute_sizes();
    }


    //feed-forward
    std::vector<float> ff(std::vector<float> const& x) override {
        
    }

    //backprop
    std::vector<float> bp(std::vector<float> const& x, std::vector<float> const& dy, float lr) override {
        //dy is understood to be 1 x Ou, where Ou is the size of an
        //unwrapped output. The output has size Ow*Oh*Od, so Ou is
        //the product of those. 

        for (int i = 0; i < Od; i++) {

        }
    }
};

#endif