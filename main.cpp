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
#include <chrono>
#include <fstream>

//#define DEBUG 1

#include "base_types.h"
#include "activation_fns.h"
#include "layers.h"
#include "cost_fn.h"
#include "tensor.h"
#include "debug.h"
#include "mnist/load_mnist.h"

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

class Model : layer {

public:
    std::vector<shared_ptr<layer> > layers;


    Model() {}

    //feed-forward
    Matrix<float> ff(MSpan<float> const& x) override {
        Matrix<float> cur(x);
        for (auto const& l : layers) {
            cur = l->ff(&cur);
        }

        return cur;
    }

    //backprop
    virtual Matrix<float> bp(MSpan<float> const& x, MSpan<float> const& dy, float lr) override {
        //Subtle bug: this used to be a vector of MSpans, but
        //then you get dangling pointers. This was happening 
        //because the actual result of running ff (see about 8-9
        //lines lower down) was being discarded but we were still
        //saving an MSpan into that matrix into the vector
        std::vector<Matrix<float> > layer_inputs;
        layer_inputs.emplace_back(x);
        
        assert(layers.size() > 0);
        for (unsigned i = 0; i < layers.size() - 1; i++) {
            layer_inputs.push_back(
                layers[i]->ff(&layer_inputs.back())
            );
        }

        Matrix<float> cur_dy(dy);
        for (int i = layers.size() - 1; i >= 0; i--){
            DEBUG("before_bp[" + to_string(i) + "]", ::dump(*(layers[i])));
            DEBUG("bp_inputs[" + to_string(i) + "]", ::dump(&layer_inputs[i]));
            DEBUG("bp_dy_in[" + to_string(i) + "]", ::dump(&cur_dy));
            cur_dy = layers[i]->bp(&layer_inputs[i], &cur_dy, lr);
            DEBUG("after_bp[" + to_string(i) + "]", ::dump(*(layers[i])));
            DEBUG("bp_outputs[" + to_string(i) + "]", ::dump(&cur_dy));
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

/*
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

    //void add_layer(layer &&l) {
    //    layers.push_back(std::make_shared<layer>(l));
    //}
};
*/

/*
void train(Model const& m) {
    cost_fn *cfn = ...
}
*/

template <typename fn, typename... T>
void time_fn(int iterations, fn F, T... args) {
    auto tic = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) F(args...);
    auto toc = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(toc-tic).count();

    cout << iterations << " iterations in " << duration << el;
}

ofstream debug_out;

Model train_mnist(std::vector<tpair> examples) {
    constexpr int input_dim = 784;
    constexpr int output_dim = 10;
    Model model;
    auto sigmoid_fn = sigmoid();
    model.add_layer(make_shared<fc>(input_dim, 512, &sigmoid_fn));
    // TODO: use softmax instead
    model.add_layer(make_shared<fc>(512, output_dim, &sigmoid_fn));
    //model.add_layer(make_shared<softmax>(512, output_dim));

    // TODO: use cross-entropy loss
    auto e = sqerr();
    float last_cost = -1.0; //Some impossible cost to make sure we don't
                            //terminate early
    float cost = -1.0;
    Matrix<float> output;
    #define max_epoch 1
    constexpr int batch_size = 32;
    int num_batches = examples.size() / batch_size;
    float lr = 0.015;
    int epoch = 0;

    do {
        // Epoch of training
        std::random_shuffle(examples.begin(), examples.end());

        // split into minibatches
        for (int batch_start_idx = 0, b = 0; b < num_batches; b++) {
            int this_batch_size = batch_size + (b < examples.size() % batch_size);

            int input_dims[2] = {this_batch_size, input_dim};
            int output_dims[2] = {this_batch_size, output_dim};
            std::vector<float> input_data(this_batch_size * input_dim);
            std::vector<float> expected_output_data(this_batch_size * output_dim);

            for (int i = 0; i < this_batch_size; i++) {
                assert(batch_start_idx + i < examples.size());
                assert(examples[batch_start_idx + i].first.size() == input_dim);
                assert(examples[batch_start_idx + i].second.size() == output_dim);
                assert(i*input_dim + input_dim < input_data.size());
                std::copy(examples[batch_start_idx + i].first.begin(), 
                          examples[batch_start_idx + i].first.end(), 
                          input_data.begin() + i * input_dim);
                std::copy(examples[batch_start_idx + i].second.begin(),
                          examples[batch_start_idx + i].second.end(),
                          expected_output_data.begin() + i * output_dim);
            }            

            Tensor<2,float> batch_inputs(std::move(input_data), input_dims);
            Tensor<2,float> expected_outputs(std::move(expected_output_data), output_dims);

            last_cost = cost;
            output = model.ff(&batch_inputs);

            cost = e.cc(&output, &expected_outputs);

            auto gradient = e.gg(&output, &expected_outputs);

            model.bp(&batch_inputs, &gradient, lr);

            batch_start_idx += this_batch_size;
        }
    } while (abs(cost - last_cost) > 1e-7 && ++epoch < max_epoch);

    if (epoch >= max_epoch) {
        cout << "Error, maximum number of epochs exceeded" << el;
    } else {
        cout << "Converged after " << epoch << " epochs" << el;
    }
    
    cout << "Cost: " << cost << el;
    cout << "Output: " << output << el;

    return model;
}

int main() {
    debug_out.open("debug.py");
    debug_out << "import numpy as np" << el;
    debug_out << "dbg_data = [" << el;

    //std::vector<float> f_vec = {-0.1, -0.3, 0.4};
    //Vector<float> f(f_vec);
    auto f = make_tensor<2,float>({1, 3}, {-0.1, -0.3, 0.4});
    cout << "Input: " << f << el;
    auto actual = make_tensor<2,float>({1, 3}, {-1, 0.2, 3.5});
    cout << "Actual: " << actual << el;

    cout << el;

    auto ident_fn = identity();
    auto relu_fn = relu();
    auto oddln_fn = oddln();
    auto sigmoid_fn = sigmoid();
    Model model;
    // model.add_layer(make_shared<fc>(5, 3, &oddln_fn));
    // model.add_layer(make_shared<fc>(2, 5, &oddln_fn));
    // model.add_layer(make_shared<fc>(5, 2, &relu_fn));
    model.add_layer(make_shared<fc>(3, 3, &ident_fn));
    //DEBUG("layer_0", model.layers[0])

    auto e = sqerr();

    float last_cost = -1.0; //Some impossible cost to make sure we don't
                            //terminate early
    float cost = -1.0;
    Matrix<float> output;
    #define max_iter 3
    int num_iter = 0;
    do {
        last_cost = cost;
        
        //DEBUG("model in", f);
        output = model.ff(&f);
        //DEBUG("model out", output);

        //cout << "Output " << output << endl;
        //debug_out << "iter_" << num_iter << " = {" << *model.layers[0] << "}" << std::endl;

        cost = e.cc(&output, &actual);

        auto gradient = e.gg(&output, &actual);

        model.bp(&f, &gradient, 0.015);
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


    /*cout << "-----------------------" << el << el;
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
    */




    /*

    std::vector<float> test_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int dims[] = {2, 2, 3};
    int constexpr t_rank = sizeof(dims)/sizeof(*dims);
    auto t1 = Tensor<t_rank, float>(test_vec, dims);
    std::cout << "t1: " << t1 << std::endl;

    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            std::cout << t1[i][j] << el;
        }
        std::cout << std::endl;
    }

    std::vector<float> test_vec_2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};  
    int otherdims[] = {3, 1, 1, 5, 1};
    int constexpr other_rank = sizeof(otherdims)/sizeof(*otherdims);
    auto t2 = Tensor<other_rank, float>(std::move(test_vec_2), otherdims);
    cout << "t2: " << t2 << el;

    for (int i = 0; i < t2.dims[0]; i++) {
      cout << "vecs[" << i << "] = " << t2[i] << std::endl;
    }

    auto prod = tensormul(t1, t2);
    cout << prod << el;


    srand(0);
    int dim1, dim2, dim3;
    cout << "Enter your dimensions" << el;
    cin >> dim1 >> dim2 >> dim3;
    int A_dims[] = {dim1, dim2};
    int constexpr A_rank = sizeof(A_dims) / sizeof(*A_dims);
    int B_dims[] = {dim2, dim3};
    int constexpr B_rank = sizeof(B_dims) / sizeof(*B_dims);

    std::vector<float> A(dim1 * dim2, 0);
    std::vector<float> B(dim2 * dim3, 0);
    
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        A[i*dim2 + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      }
    }
    for (int i = 0; i < dim2; i++) {
      for (int j = 0; j < dim3; j++) {
        B[i*dim3 + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      }
    }

    Tensor<A_rank,float> A_tensor(A, A_dims);
    Tensor<B_rank,float> B_tensor(B, B_dims);

    time_fn(
        500,
        [=](const std::vector<float>& A, const std::vector<float>& B) {
            std::vector<float> C(dim1 * dim3);
            for (int i = 0; i < dim1; i++) {
              //float const* A_row = A.data() + i*dim2;
              //float *C_row = C.data() + i*dim3;
              
              for (int j = 0; j < dim3; j++) {
                //C_row[j] = 0;
                C[i*dim3 + j] = 0;
                for (int k = 0; k < dim2; k++) {
                  //C_row[j] += A_row[k] * B[k*dim3 + j];
                  C[i*dim3 + j] += A[i*dim2 + k] * B[k*dim3 + j];
                }
              }
            }
        }, A, B
    );

    using fntype = Tensor<2,float>(*)(Tensor<2,float> const&, Tensor<2,float> const&);

    time_fn(
        500,
        static_cast<fntype>(tensormul), 
        //tensormul,
        A_tensor, 
        B_tensor
    );
    */
    debug_out << "]" << el;

    debug_out << R"###(
tags = set()
for x in dbg_data:
    tags.add(x["tag"])
tags = list(tags)

tag_groups = {}
for tag in tags:
    tag_groups[tag] = [x["data"] for x in dbg_data if x["tag"] == tag]
)###";
    debug_out.close();
    return 0;
}