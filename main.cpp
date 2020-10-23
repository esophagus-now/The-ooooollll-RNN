#include <iostream>
#include <vector>
#include <assert.h>
#include <numeric>
#include <math.h>
#include <map>
#include <string>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <random>
#include <utility>

#include "base_types.h"
#include "activation_fns.h"
#include "layers.h"
#include "cost_fn.h"
#include "matrix.h"

using namespace std;

ostream& el(ostream &o) {return o << "\n";}

template <typename T>
ostream& operator<<(ostream &o, vector<T> const& v) {
    o << "[";
    auto delim = "";
    for (auto const& i : v) {
        o << delim << i;
        delim = ",";
    }

    return o << "]";
}

template <typename T>
ostream& operator<<(ostream &o, shared_ptr<T> const& p) {
    if (p) {
        return o << "shared_ptr to {" << *p << "}";
    } else {
        return o << "(null shared_ptr)" << el;
    }
}

/*
struct layer_data {
    string type;
    string name;

    layer_data(string const& type, string const& name) :
        type(type),
        name(name)
    {}

    virtual explicit operator string() const {
        return name + " of type " + type;
    }

};

struct my_custom_data : layer_data {
    vector<float> vf;
};

ostream& operator<<(ostream &o, layer_data const& ld) {
    return o << string(ld);
}
*/

class Model : layer {
    std::vector<shared_ptr<layer> > layers;

public:

    Model() {}

    //feed-forward
    std::vector<float> ff(std::vector<float> const& x) override {
        std::vector<float> cur(x);
        for (auto const& l : layers) {
            cur = l->ff(cur);
        }

        return cur;
    }

    //backprop
    virtual std::vector<float> bp(std::vector<float> const& x, std::vector<float> const& dy, float lr) override {
        std::vector<std::vector<float> > layer_inputs;
        layer_inputs.push_back(x);
        
        assert(layers.size() > 0);
        for (unsigned i = 0; i < layers.size() - 1; i++) {
            layer_inputs.push_back(
                layers[i]->ff(layer_inputs.back())
            );
        }

        std::vector<float> cur_dy(dy);
        for (int i = layers.size() - 1; i >= 0; i--){
            cur_dy = layers[i]->bp(layer_inputs[i], cur_dy, lr);
        }
        
        return cur_dy;
    }

    void append_fc_layer(int r, int c, activation_fn *act) {
        layers.push_back(make_shared<fc>(r, c, act));
    }

    void add_layer(std::shared_ptr<layer> pl) {
        layers.push_back(pl);
    }

    /*void add_layer(layer &&l) {
        layers.push_back(std::make_shared<layer>(l));
    }*/
};

class Model_generalized : layer_generalized {
    std::vector<shared_ptr<layer_generalized> > layers;

public:

    Model_generalized() {}

    //feed-forward
    std::shared_ptr<layer_data> ff(std::shared_ptr<layer_data const> x) override {
        auto cur = std::shared_ptr<layer_data>(x->clone());
        for (auto const& l : layers) {
            cur = l->ff(cur);
        }

        return cur;
    }

    //backprop
    virtual std::shared_ptr<layer_data> bp(std::shared_ptr<layer_data const> x, std::shared_ptr<layer_data const> dy, float lr) override {
        std::vector<std::shared_ptr<layer_data> > layer_inputs;
        layer_inputs.push_back(std::shared_ptr<layer_data>(x->clone()));
        
        assert(layers.size() > 0);
        for (unsigned i = 0; i < layers.size() - 1; i++) {
            layer_inputs.push_back(
                layers[i]->ff(layer_inputs.back())
            );
        }

        auto cur_dy = std::shared_ptr<layer_data>(dy->clone());
        for (int i = layers.size() - 1; i >= 0; i--){
            cur_dy = layers[i]->bp(layer_inputs[i], cur_dy, lr);
        }
        
        return cur_dy;
    }

    void append_fc_layer(int r, int c, activation_fn *act) {
        layers.push_back(make_shared<fc_generalized>(r, c, act));
    }

    void add_layer(std::shared_ptr<layer_generalized> pl) {
        layers.push_back(pl);
    }

    /*void add_layer(layer &&l) {
        layers.push_back(std::make_shared<layer>(l));
    }*/
};

/*
void train(Model const& m) {
    cost_fn *cfn = ...
}
*/

int main() {

    vector<float> f = {-0.1, -0.3, 0.4};
    cout << "Input: " << f << el;
    vector<float> actual = {-1, 0.2, 3.5};
    cout << "Actual: " << actual << el;

    cout << el;

    auto relu_fn = relu();
    auto oddln_fn = oddln();
    auto sigmoid_fn = sigmoid();
    Model model;
    model.add_layer(make_shared<fc>(5, 3, &oddln_fn));
    model.add_layer(make_shared<fc>(2, 5, &oddln_fn));
    model.add_layer(make_shared<fc>(5, 2, &relu_fn));
    model.add_layer(make_shared<fc>(3, 5, &oddln_fn));

    auto e = sqerr();

    float last_cost = -1.0; //Some impossible cost to make sure we don't
                            //terminate early
    float cost = -1.0;
    vector<float> output;
    #define max_iter 10000
    int num_iter = 0;
    do {
        last_cost = cost;
        
        output = model.ff(f);

        cost = e.cc(output, actual);

        model.bp(f, e.gg(output,actual), 0.015);
    } while (abs(cost - last_cost) > 1e-7 && ++num_iter < max_iter);

    if (num_iter >= max_iter) {
        cout << "Error, maximum number of iterations exceeded" << el;
    } else {
        cout << "Converged after " << num_iter << " iterations" << el;
    }
    
    cout << "Cost: " << cost << el;
    cout << "Output: " << output << el;


    //Above code was copy-pasted and then changed to use new
    //generalized versions


    cout << "-----------------------" << el << el;
    cout << "And now to try the generalized verions:" << el;

    Model_generalized gmodel;
    gmodel.add_layer(make_shared<fc_generalized>(5, 3, &oddln_fn));
    gmodel.add_layer(make_shared<fc_generalized>(2, 5, &oddln_fn));
    gmodel.add_layer(make_shared<fc_generalized>(5, 2, &relu_fn));
    gmodel.add_layer(make_shared<fc_generalized>(3, 5, &oddln_fn));

    auto ge = sqerr_generalized();

    last_cost = -1.0; //Some impossible cost to make sure we don't
                            //terminate early
    cost = -1.0;
    auto fv_input = make_shared<floatvec>(f);
    auto fv_actual = make_shared<floatvec>(actual);
    shared_ptr<layer_data> g_output;

    num_iter = 0;
    do {
        last_cost = cost;
        
        g_output = gmodel.ff(fv_input);

        cost = ge.cc(g_output, fv_actual); 

        gmodel.bp(fv_input, ge.gg(g_output, fv_actual), 0.015);
    } while (abs(cost - last_cost) > 1e-7 && ++num_iter < max_iter);

    if (num_iter >= max_iter) {
        cout << "Error, maximum number of iterations exceeded" << el;
    } else {
        cout << "Converged after " << num_iter << " iterations" << el;
    }
    
    cout << "Cost: " << cost << el;

    //How ugly... https://stackoverflow.com/questions/6795629/how-does-one-downcast-a-stdshared-ptr/6795680
    auto fv_output = static_pointer_cast<floatvec>(g_output);
    cout << "Output: " << fv_output->impl << el;


    std::vector<float> test_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int dims[] = {2, 1, 2, 3};
    int constexpr t_rank = sizeof(dims)/sizeof(*dims);
    auto t1 = Tensor<t_rank, float>(test_vec, dims);
    std::cout << "t1: " << t1 << std::endl;
    //TensorSpan<t_rank, float> mat(t);

    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            std::cout << t1[i][j] << el;
        }
        std::cout << std::endl;
    }

    std::vector<float> test_vec_2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};  
    int otherdims[] = {3, 1, 5};
    int constexpr other_rank = sizeof(otherdims)/sizeof(*otherdims);
    auto t2 = Tensor<other_rank, float>(std::move(test_vec_2), otherdims);
    //TensorSpan<other_rank, float> vecs(t2);
    cout << "t2: " << t2 << el;

    for (int i = 0; i < t2.dims[0]; i++) {
      cout << "vecs[" << i << "] = " << t2[i] << std::endl;
    }

    auto prod = tensormul(t1, t2);
    cout << prod << el;

    return 0;
}