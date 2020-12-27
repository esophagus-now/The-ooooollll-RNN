#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include <utility>

#include "base_types.h"
#include "tensor.h"
#include "layer_types.h"
#include "debug.h"

struct simple_linear : layer {
    float a;

    simple_linear() : a(1.0) {}
    simple_linear(float a) : a(a) {}

    //feed-forward
    Matrix<float> ff(MSpan<float> const& x) override {
        Matrix<float> ret(x);
        for (int i = 0; i < ret.length(); i++) {
          for (int j = 0; j < ret[i].length(); j++) {
            ret[i][j] *= a;
          }
        }
        return ret;
    }

    //backprop
    virtual Matrix<float> bp(MSpan<float> const& x, MSpan<float> const& dy, float lr) override {
        Matrix<float> ret(dy);

        assert(std::equal(x.dims, x.dims+2, dy.dims));
        //assert(x.dims[0] == dy.dims[0])
        //assert(x.dims[1] == dy.dims[1]);

        float sum = 0.0;
        for (int i = 0; i < dy.length(); i++) {
            for (int j = 0; j < dy[i].length(); j++) {
                sum += x[i][j] * dy[i][j];
                ret[i][j] *= a;
            }
        }
        sum *= lr;

        a -= sum;

#ifdef DEBUG_PRINTS
        std::cout << "\tAdjustment = " << -sum << std::endl;
        std::cout << "\t a is now " << a << std::endl;
#endif

        return ret;
    }
};

//fully connected
struct fc : layer {
    // W: (num outputs) x (num inputs)
    // bias: (num outputs)
    Matrix<float> W;
    Vector<float> bias;
    activation_fn* act_fn;

    std::string name;
    static int num;

    fc(int n_out, int n_in, activation_fn* act_fn, std::string name) :  
            act_fn(act_fn), name(name) {
        //assert(n_in > 0);
        //assert(n_out > 0);
        //No longer needed? Tensor constructor will throw an
        //std::bad_Alloc if you give bad sizes
        
        //mat = Matrix<float>(r,c);
        int dims[] = {n_out, n_in};
        W = Matrix<float>(dims, uniform_randgen<float>(-1.0, 1.0));
        //std::cout << "Initial weights: " << W << std::endl;
        bias = Vector<float>(dims, uniform_randgen<float>(-1.0, 1.0));
        //std::cout << "Initial biases: " << bias << std::endl;

        // bias = Vector<float>();

        // auto gen = std::default_random_engine();
        // auto dist = std::uniform_real_distribution<float>(-1.0, 1.0);

        // //Start with el dumbo identity matrix
        // for (unsigned i = 0; i < r; i++) {
        //     for (unsigned j = 0; j < c; j++) {
        //         mat[i][j] = dist(gen);
        //     }
        //     bias[i] = dist(gen);
        // }
    }

    static std::string gen_name() {
        return "fc_" + std::to_string(num++);
    }

    fc(int n_out, int n_in, activation_fn* act_fn) :
        fc(n_out, n_in, act_fn, gen_name()) {}

    //feed-forward
    //Note to our future selves: this does x * transpose(W)
    // W: (num outputs) x (num inputs)
    // x: (batch size) x (num inputs)
    // ret: (batch size) x (num outputs)
    Matrix<float> ff(MSpan<float> const& x) override {
        assert(x.dims[1] == W.dims[1]);

        auto W_T = (&W).transpose();
        //std::cout << "W_T = " << W_T << std::endl;
        //auto W_T_T = W_T.transpose();
        //std::cout << "W_T_T = " << W_T_T << std::endl;

#ifdef DEBUG_PRINTS

        std::cout << "Calling tensormul on: x = ["
                  << x.dims[0] << "," << x.dims[1] << "]\n"
                  << x << "\n and W.transpose = [" 
                  << W_T.dims[0] << "," << W_T.dims[1] << "]\n"
                  << W_T
                  << "\n";

#endif

        auto ret = tensormul(x, W_T);
        //std::cout << "tensormul result: " << ret << std::endl;
        assert(ret.length() == x.length());
        assert(ret.dims[1] == bias.dims[0]);

        for (int i = 0; i < ret.length(); i++) { //batch size
            for (int j = 0; j < ret[i].length(); j++) { //number of outputs
                ret[i][j] = (*act_fn)(ret[i][j] + bias[j]);
            }
        }

        return ret;
    }

    //backprop
    // W: (num outputs) x (num inputs)
    // x: (batch size) x (num inputs)
    // dy: (batch size) x (num outputs)
    // ret: (batch size) x (num inputs)
    // In the notation of the scratchwork I sent earlier, 
    // in ff we compute y = x*W^T + [1,1,1,...,1]*bias
    // so A = x, B = W^T, and C = [1,1,1,...,1]*bias
    // That means dErr/dW = (dD/dB)^T = (dErr/dy)^T * x
    // and dErr/dx = (dD/dA) = (dErr/dy)*W
    // and dErr/dbias = (dD/dC) = (dErr/dy)
    virtual Matrix<float> bp(MSpan<float> const& x, MSpan<float> const& dy, float lr) override {
        assert(x.dims[1] == W.dims[1]);
        assert(dy.dims[0] == x.dims[0]);
        assert(dy.dims[1] == W.dims[0]);
        assert(bias.dims[0] == dy.dims[1]);

        Matrix<float> z = ff(x);
        assert(std::equal(z.dims, z.dims+2, dy.dims));
        Matrix<float> z2(dy);

        for (int i = 0; i < z2.dims[0]; i++) {
            for (int j = 0; j < z2.dims[1]; j++) {
                z2[i][j] *= (*act_fn)[z[i][j]]; //Silly operator[] for derivative
            }
        }

        auto z2_T = (&z2).transpose();
        
        Matrix<float> dErr_dW = tensormul(z2_T, x);
        Matrix<float> dErr_dx = tensormul(&z2, &W);
        assert(std::equal(dErr_dW.dims, dErr_dW.dims+2, W.dims));
        assert(std::equal(dErr_dx.dims, dErr_dx.dims+2, x.dims));
        //dErr_dbias = dy

        //Update weights
        for (int i = 0; i < W.dims[0]; i++) {
            for (int j = 0; j < W.dims[1]; j++) {
                W[i][j] -= lr * dErr_dW[i][j];
            }
        }

        //Update biases by summing over columns of dy (batch size)
        for (int i = 0; i < z2.dims[0]; i++) {
            for (int j = 0; j < z2.dims[1]; j++) {
                bias[j] -= lr * z2[i][j];
            }
        }

        //std::cout << "Weights are now " << W << "\n";
        //std::cout << "Biases are now" << bias << std::endl;

        return dErr_dx;
    }

    void dump(std::ostream& o) const override {
        o << "{\"" << name << "\": {\n";
        o << "\"W\": np.array(" << W << "),\n";
        o << "\"bias\": np.array(" << bias << ")\n";
        o << "}},";
    }
};

/*struct fc_generalized : layer_generalized {
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
};*/

#endif