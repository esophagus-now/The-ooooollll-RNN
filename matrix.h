#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
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

// int main() {
//     auto m = Matrix<float>(8,7);

//     for (int i = 0; i < 8; i++) {
//         for (int j = 0; j < 7; j++) {
//             m[i][j] = i*7 + j;
//         }
//     }

//     for (int i = 0; i < 8; i++) {
//         for (int j = 0; j < 7; j++) {
//             std::cout << m[i][j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     std::cout << m << std::endl;
//     return 0;
// }


//Template specialization below covers ndims = 1, we can assume
//this struct always has ndims > 1
template <int ndims, typename T>
class TensorSpan {
private:
    //T *data;
    T const* data;
    int const* dims;
    int stride;

    static int get_default_stride(int const* dims) {
        int prod = 1;
        for (int i = ndims - 1; i > 0; i--) prod *= dims[i];
        return prod;
    }

public:
    TensorSpan(T *data, int const* dims, int stride) :
        data(data), dims(dims), stride(stride)
    {}

    TensorSpan(T const* data, int const* dims, int stride) :
        data(data), dims(dims), stride(stride)
    {}
    
    TensorSpan(T *data, int const* dims) : TensorSpan(data, dims, get_default_stride(dims))
    {}

    TensorSpan<ndims - 1, T> operator[] (int n) {
        auto ptr = const_cast<T*>(data);
        return TensorSpan<ndims - 1, T>(ptr + n * stride, dims + 1, stride / dims[1]);
    }

    TensorSpan<ndims - 1, T> const operator[] (int n) const {
        return TensorSpan<ndims - 1, T>(data + n * stride, dims + 1, stride / dims[1]);
    }

    template <int N, typename S>
    friend std::ostream& operator<<(std::ostream&, TensorSpan<N,S> const&);

    //template <int N, typename S>
    //friend std::ostream& operator<<(std::ostream&, TensorSpan<N,S> const&&);
    
    //template<>
    //friend class TensorSpan<ndims + 1, T>;
};

template <typename T>
class TensorSpan<1, T> {
private:
    T const* data;

    int const* dims;
    int stride;

public:
    TensorSpan(T *data, int const* dims, int stride) :
        data(data), dims(dims), stride(stride)
    {}

    TensorSpan(T const *data, int const* dims, int stride) :
        data(data), dims(dims), stride(stride)
    {}

    T& operator[] (int n) {
        auto ptr = const_cast<T*>(data);
        return ptr[n * stride];
    }

    T const& operator[] (int n) const {
        return data[n * stride];
    }
    
    template <typename S>
    friend std::ostream& operator<<(std::ostream&, TensorSpan<1,S> const&);

    // template <typename S>
    // friend std::ostream& operator<<(std::ostream&, TensorSpan<1,S>&);
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

/*template<int ndims, typename T>
std::ostream& operator<<(std::ostream &o, TensorSpan<ndims, T> const&& t) {
    o << "{";
    auto delim = "";

    for (int i = 0; i < t.dims[0]; i++) {
        o << delim << t[i];
        delim = ", ";
    }

    return o << "}";
}*/

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

#endif