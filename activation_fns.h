#ifndef ACTIVATION_FNS_H
#define ACTIVATION_FNS_H 1

#include "base_types.h"

struct identity : activation_fn {
    identity() {}

    float operator()(float x) override {
      return x;
    }
    float operator[](float x) override {
      return 1;
    }
};

struct hyptan : activation_fn {
    float operator()(float x) override {
      return tanh(x);
    }
    float operator[](float x) override {
      float tmp = cosh(x);
      return 1.0 / (tmp*tmp);
    }
};

struct sigmoid : activation_fn {
    float operator()(float x) override {
        return 1.0 / (1 + exp(-x));
    }
    
    float operator[](float x) override {
        float sig = operator()(x);
        return sig*(1.0 - sig);
    }
};

//https://www.desmos.com/calculator/3miy5kbrlj
struct modified_sigmoid : activation_fn {
    float x_t;
    float sig_x_t;
    float sig_minus_x_t;
    float d_sig_x_t;

    modified_sigmoid(float x_t) {
        this->x_t = x_t;
        auto s = sigmoid();

        sig_x_t = s(x_t);
        sig_minus_x_t = 1.0 - sig_x_t;
        d_sig_x_t = s[x_t];
    }

    modified_sigmoid() : modified_sigmoid(2.0) {}

    float operator()(float x) override {
        /*if (x < -x_t) {
            return d_sig_x_t*(x + x_t) + sig_minus_x_t;
        } else if (x > x_t) {
            return d_sig_x_t*(x - x_t) + sig_x_t;
        } else {
            return 1.0 / (1 + exp(-x));
        }*/
        return 1.0 / (1 + exp(-x));
    }
    
    float operator[](float x) override {
        if (abs(x) <= x_t) {
            //return d_sig_x_t;
            return 1.0;
        } else {
            float sig = operator()(x);
            return sig*(1.0 - sig);
        }
    }
};

struct hypsin : activation_fn {
    float operator()(float x) override {
        return sinh(x);
    }
    
    float operator[](float x) override {
        return cosh(x);
    }
};

//https://www.desmos.com/calculator/je2ixlcklj
struct oddln : activation_fn {
    float operator()(float x) override {
        if (x >= 0)
            return log(x + 1.0);
        else 
            return -log(1.0 - x);
    }
    
    float operator[](float x) override {
        if (x >= 0)
            return 1.0/(x + 1.0);
        else 
            return 1.0/(1.0 - x);
    }
};

struct relu : activation_fn {
  float operator()(float x) override {
    return x < 0 ? 0 : x;
  }

  float operator[](float x) override {
    //return x < 0 ? 0 : 1;
    return x >= 0;
  }
};

#endif