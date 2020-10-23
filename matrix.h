#ifndef MATRIX_H
#define MATRIX_H


// #ifndef DEBUG
// #define DEBUG 1
// #endif

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <assert.h>

template <typename T>
class Matrix;

template <typename T>
class Span : public layer_data {
    friend class Matrix<T>;
    T *base;
    int stride;
    int len;

    //vector<T> mystuff;
    //bool do_i_own_my_stuff;
// protected:
public:
    Span(T *base, int stride, int len) :
        base(base), stride(stride), len(len),
        layer_data("Span", true)
    {}

    T& operator[](int ind) {
        #ifdef DEBUG
        assert(0 <= ind && ind < len);
        #endif
        return *(base + ind*stride);
    }

    Span* clone() const override {
        return new Span(*this);
    }
};

template <typename T>
class Matrix : public layer_data {
    std::vector<T> base;
    int r, c;

public:
    Matrix(int r, int c) : r(r), c(c), layer_data("Matrix", true) {
        #ifdef DEBUG
        assert(r > 0 && c > 0);
        #endif
        base = std::vector<T>(r*c, 0.0);
    }

    Span<T> operator[](int row) {
        #ifdef DEBUG
        assert(0 <= row && row < r);
        #endif
        return Span<T>(base.data() + row*c, 1, c);
    }

    Span<T> col(int column) {
        #ifdef DEBUG
        assert(0 <= column && column < c);
        #endif
        return Span<T>(base.data() + column, c, r);
    }

    Span<T> diag() {
        return Span<T>(base.data(), c + 1, std::min(r,c));
    }
    
    Matrix* clone() const override {
        return new Matrix(*this);
    }

    template <typename S>
    friend std::ostream& operator<<(std::ostream&, Matrix<S> &);
};

template <typename T>
std::ostream& operator<<(std::ostream& o, Matrix<T>& M) {
    o << "{";
    auto delim = "\n";
    for (int i; i < M.r; i++) {
        o << "\t";
        for (int j = 0; j < M.c; j++) {
            o << delim << M[i][j];
            delim = ", ";
        }
        delim = "\n";
    }

    return o << delim << "}";
}

//Template specialization below covers ndims = 1, we can assume
//this struct always has ndims > 1
template <int rank, typename T>
struct TensorSpan {
    static_assert(rank >= 1, "TensorSpan code only handles positive-rank tensors");

    //T *data;
    T const* data;
    int const* dims;
    int const* strides;

    TensorSpan(T const* data, int const* dims, int const* strides) :
        data(data), dims(dims), strides(strides)
    {
        #ifdef DEBUG
        std::cout << "Constructing a TensorSpan<" << rank << "> with dimensions [";
        auto delim = "";
        for (int i = 0; i < rank; i++) {
            std::cout << delim << dims[i];
            delim = ",";
        }
        std::cout << "]" << std::endl;
        #endif
    }

    TensorSpan<rank - 1, T> operator[] (int n) {
        //auto ptr = const_cast<T*>(data);
        //return TensorSpan<rank - 1, T>(ptr + n * strides[0], dims + 1, strides + 1);
        return TensorSpan<rank - 1, T>(data + n * strides[0], dims + 1, strides + 1);
    }

    TensorSpan<rank - 1, T> const operator[] (int n) const {
        return TensorSpan<rank - 1, T>(data + n * strides[0], dims + 1, strides + 1);
    }

    //Like doing my_tensor(:,:,:,n) in MATLAB syntax
    TensorSpan<rank - 1, T> back_index(int n) {
        return TensorSpan<rank - 1, T>(data + n*strides[rank-1], dims, strides);
    }

    //Like doing my_tensor(:,:,:,n) in MATLAB syntax
    TensorSpan<rank - 1, T> const back_index(int n) const {
        return TensorSpan<rank - 1, T>(data + n*strides[rank-1], dims, strides);
    }
};

template <typename T>
struct TensorSpan<1, T> {
    T const* data;

    int const* dims;
    int const* strides;

    TensorSpan(T const *data, int const* dims, int const* strides) :
        data(data), dims(dims), strides(strides)
    {}

    T& operator[] (int n) {
        auto ptr = const_cast<T*>(data);
        return ptr[n * strides[0]];
    }

    T const& operator[] (int n) const {
        return data[n * strides[0]];
    }
    
    //Like doing my_tensor(:,:,:,n) in MATLAB syntax
    T& back_index(int n) {
        return (*this)[n];
    }

    //Like doing my_tensor(:,:,:,n) in MATLAB syntax
    T const& back_index(int n) const {
        return (*this)[n];
    }
};

template<int ndims, typename T>
std::ostream& operator<<(std::ostream &o, TensorSpan<ndims, T> const& t) {
    o << "{";
    auto delim = "";

    for (int i = 0; i < t.dims[0]; i++) {
        auto thing = t[i];
        o << delim << thing; //Make it an l-value
        delim = ", ";
    }

    return o << "}";
}

template<typename T>
std::ostream& operator<<(std::ostream &o, TensorSpan<1, T> const& t) {
    o << "{";
    auto delim = "";

    for (int i = 0; i < t.dims[0]; i++) {
        o << delim << t[i];
        delim = ", ";
    }

    return o << "}";
}

template <int rank, typename T>
struct Tensor {
    std::vector<T> storage;
    int dims[rank];
    int strides[rank];

    int copy_dims_get_strides(int const *dims) {        
        int prod = 1;
        for (int i = rank - 1; i >= 0; i--) {
            this->dims[i] = dims[i];
            this->strides[i] = prod;
            prod *= dims[i]; //NOT a mistake: only update prod after setting strides[i]
        } 

        return prod;
    }

    //TODO: add constructors that might copy from a vector or something, 
    //rather than initialize an empty one
    Tensor(int const* dims) {
        int prod = copy_dims_get_strides(dims);
        storage = std::vector<T>(prod);
    }

    Tensor(std::vector<T> vec, int const* dims) : storage(std::move(vec)) {
        copy_dims_get_strides(dims);
    }

    TensorSpan<rank-1, T> operator[] (int n) {
        return TensorSpan<rank, T>(*this)[n];
    }

    TensorSpan<rank, T> operator& () {
        return TensorSpan<rank, T>(*this);
    }

    TensorSpan<rank, T> const operator& () const {
        return TensorSpan<rank, T>(*this);
    }

    operator TensorSpan<rank, T>() const {
        return TensorSpan<rank, T>(storage.data(), dims, strides);
    }
};

template <int rank, typename T>
std::ostream& operator<<(std::ostream &o, Tensor<rank,T> const& t) {
    return o << TensorSpan<rank,T>(t);
}

//Does NOT perform any safety checks. Will default-construct all
//elements in the output
//Base case: dot product of two 1-d arrays
template<int LHS_rank, int RHS_rank, typename T>
typename std::enable_if<(LHS_rank == 1) && (RHS_rank == 1),void>::type
tensormul(
    TensorSpan<LHS_rank, T> const& A, 
    TensorSpan<RHS_rank, T> const& B,
    T& dest
) {

#ifdef DEBUG

    std::cout   << "Called tensormul_wrapped<" 
                << LHS_rank 
                << "," 
                << RHS_rank
                << ">"
                << std::endl
    ;

    assert(A.dims[0] == B.dims[0]);
#endif

    dest = T();
    for (int i = 0; i < A.dims[0]; i++) {
      dest += A[i] * B[i];
    }
    return;
}

//Does NOT perform any safety checks. Will default-construct all
//elements in the output
template<int LHS_rank, int RHS_rank, typename T>
typename std::enable_if<(LHS_rank > 1),void>::type
tensormul(
    TensorSpan<LHS_rank, T> const& A, 
    TensorSpan<RHS_rank, T> const& B,
    TensorSpan<LHS_rank+RHS_rank-2, T> dest
) {
#ifdef DEBUG
    std::cout   << "Called tensormul<" 
                << LHS_rank 
                << "," 
                << RHS_rank
                << ">"
                << std::endl
    ;
#endif

    for (int i = 0; i < A.dims[0]; i++) {
        tensormul(A[i], B, dest[i]);
    }
}

//Does NOT perform any safety checks. Will default-construct all
//elements in the output
template<int LHS_rank, int RHS_rank, typename T>
typename std::enable_if<(LHS_rank == 1) && (RHS_rank > 1),void>::type
tensormul(
    TensorSpan<LHS_rank, T> const& A, 
    TensorSpan<RHS_rank, T> const& B,
    TensorSpan<LHS_rank+RHS_rank-2, T> dest
) {

#ifdef DEBUG
    std::cout   << "Called tensormul_wrapped<" 
                << LHS_rank 
                << "," 
                << RHS_rank
                << ">"
                << std::endl
    ;
#endif
    for (int i = 0; i < B.dims[RHS_rank-1]; i++) {
        tensormul(A, B.back_index(i), dest.back_index(i));
    }
}

template<int LHS_rank, int RHS_rank, typename T>
Tensor<LHS_rank+RHS_rank-2, T> tensormul(
    TensorSpan<LHS_rank, T> const& A, 
    TensorSpan<RHS_rank, T> const& B)
{
    static_assert(LHS_rank >= 2, "LHS must be matrix or array of matrices");
    static_assert(RHS_rank >= 1, "RHS must be vector or array of vectors");

#ifdef DEBUG
    std::cout   << "Called tensormul_wrapped<" 
                << LHS_rank 
                << "," 
                << RHS_rank
                << ">"
                << std::endl
    ;
#endif

    int constexpr ret_rank = LHS_rank + RHS_rank - 2;

    int ret_dims[ret_rank];
    int pos = 0;
    for (int i = 0; i < LHS_rank - 1; i++) {
        ret_dims[pos++] = A.dims[i];
    }
    for (int i = 1; i < RHS_rank; i++) {
        ret_dims[pos++] = B.dims[i];
    }

    auto delim = "";
    std::cout << "ret_dims = [";
    for (int i = 0; i < ret_rank; i++) {
        std::cout << delim << ret_dims[i];
        delim = ",";
    }
    std::cout << "]" << std::endl;

    Tensor<ret_rank, T> ret(ret_dims);

    if (A.dims[LHS_rank - 1] != B.dims[0]) {
        throw std::runtime_error("Inner dimensions must agree");
    }

    tensormul(A, B, &ret);

    return ret;
}

template<int LHS_rank, int RHS_rank, typename T>
Tensor<LHS_rank+RHS_rank-2, T> tensormul(
    Tensor<LHS_rank, T> const& A, 
    Tensor<RHS_rank, T> const& B)
{
    return tensormul(&A, &B);
}

#ifdef DEBUG
#undef DEBUG
#endif

#endif