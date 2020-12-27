#ifndef TENSOR_H
#define TENSOR_H

// #ifndef DEBUG_PRINTS
// #define DEBUG_PRINTS 1
// #endif

#include <algorithm>
#include <array>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <iostream>
#include <assert.h>
#include <random>
#include <cmath>

template <typename T>
static bool compare(T const& a, T const& b) {
    return a == b;
}


#define FLOAT_TOL 1e-6
static bool compare(float const& a, float const& b) {
	return fabs(a - b) < FLOAT_TOL;
}

#define DOUBLE_TOL 1e-8
static bool compare(double const& a, double const& b) {
	return fabs(a - b) < DOUBLE_TOL;
}

//Template specialization below covers ndims = 1, we can assume
//this struct always has ndims > 1
template <int rank, typename T>
struct TSpan {
    static_assert(rank >= 2, "TSpan code only handles positive-rank tensors");

    T const* data;
    int const* dims;
    int const* strides;

    bool do_i_own_my_stuff;

    TSpan(T const* data, int const* dims, int const* strides, 
          bool you_should_own_your_stuff = false) :
        data(data), dims(dims), strides(strides), do_i_own_my_stuff(you_should_own_your_stuff)
    {
        #ifdef DEBUG_PRINTS
        std::cout << "Constructing a TSpan<" << rank << "> with dimensions [";
        auto delim = "";
        for (int i = 0; i < rank; i++) {
            std::cout << delim << dims[i];
            delim = ",";
        }
        std::cout << "]" << std::endl;
        #endif
        
        if (do_i_own_my_stuff) {
            int *dims_alloc = new int[rank];
            int *strides_alloc = new int[rank];

            for (int i = 0; i < rank; i++) {
                dims_alloc[i] = dims[i];
                strides_alloc[i] = strides[i];
            }

            this->dims = dims_alloc;
            this->strides = strides_alloc;
        }
    }

    ~TSpan() {
        if (do_i_own_my_stuff) {
            delete[] dims;
            delete[] strides;
        }
    }

    TSpan(TSpan const& other) :
        TSpan(other.data, other.dims, other.strides, true) 
    {}

    TSpan(TSpan&& other) {
        data = other.data;
        dims = other.dims;
        strides = other.strides;
        do_i_own_my_stuff = other.do_i_own_my_stuff;

        other.data = nullptr; //Technically unnecessary
        other.dims = nullptr; //Technically unnecessary
        other.strides = nullptr; //Technically unnecessary
        other.do_i_own_my_stuff = false; //Necessary
    }

    //TODO: this is more thorny
    TSpan& operator=(TSpan const&) = delete;

    TSpan& operator=(TSpan&& other) {
        if(this != &other) {
            //Get rid of what was here before
            if(do_i_own_my_stuff) {
                delete[] dims;
                delete[] strides;
            }

            //Copy from the other guy
            data = other.data;
            dims = other.dims;
            strides = other.strides;
            do_i_own_my_stuff = other.do_i_own_my_stuff;

            //Get rid of other guy's stuff
            other.data = nullptr; //Technically unnecessary
            other.dims = nullptr; //Technically unnecessary
            other.strides = nullptr; //Technically unnecessary
            other.do_i_own_my_stuff = false; //Necessary
        }
        return *this;
    }

    TSpan<rank - 1, T> operator[] (int n) {
        //auto ptr = const_cast<T*>(data);
        //return TSpan<rank - 1, T>(ptr + n * strides[0], dims + 1, strides + 1);
        return TSpan<rank - 1, T>(data + n * strides[0], dims + 1, strides + 1);
    }

    TSpan<rank - 1, T> const operator[] (int n) const {
        return TSpan<rank - 1, T>(data + n * strides[0], dims + 1, strides + 1);
    }

    //Like doing my_tensor(:,:,:,n) in MATLAB syntax
    TSpan<rank - 1, T> back_index(int n) {
        return TSpan<rank - 1, T>(data + n*strides[rank-1], dims, strides);
    }

    //Like doing my_tensor(:,:,:,n) in MATLAB syntax
    TSpan<rank - 1, T> const back_index(int n) const {
        return TSpan<rank - 1, T>(data + n*strides[rank-1], dims, strides);
    }

    TSpan<2, T> transpose() const {
        static_assert(rank == 2, "transpose only available for rank-2 tensors");
        int new_dims[2] = {dims[1], dims[0]};
        int new_strides[2] = {strides[1], strides[0]};

        TSpan<2, T> ret = TSpan(data, new_dims, new_strides, true);
        return ret;
    }

    int length() const {
      return dims[0];
    }
    
    void dump(std::ostream &o) const {
        o << "np.array(" << *this << ")";
    }

    bool operator==(const TSpan<rank, T>& other) const {
        if (dims[0] != other.dims[0])
            return false;
        for (int i = 0; i < dims[0]; i++) {
            if (!compare((*this)[i], other[i]))
                return false;
        }
        return true;
    }
};


template <typename T>
struct TSpan<1, T> {
    T const* data;

    int const* dims;
    int const* strides;
    
    bool do_i_own_my_stuff;

    TSpan(T const *data, int const* dims, int const* strides, 
          bool you_should_own_your_stuff = false) :
        data(data), dims(dims), strides(strides), do_i_own_my_stuff(you_should_own_your_stuff)
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

    int length() const {
      return dims[0];
    }

    void dump(std::ostream &o) const {
        o << "np.array(" << *this << ")";
    }

    bool operator==(const TSpan<1, T>& other) const {
        if (dims[0] != other.dims[0])
            return false;
        for (int i = 0; i < dims[0]; i++) {
            if (!compare((*this)[i], other[i]))
                return false;
        }
        return true;
    }
};

template<int ndims, typename T>
std::ostream& operator<<(std::ostream &o, TSpan<ndims, T> const& t) {
    o << "[";
    auto delim = "";

    for (int i = 0; i < t.dims[0]; i++) {
        auto thing = t[i];
        o << delim << thing; //Make it an l-value
        delim = ", ";
    }

    return o << "]";
}

template<typename T>
std::ostream& operator<<(std::ostream &o, TSpan<1, T> const& t) {
    o << "[";
    auto delim = "";

    for (int i = 0; i < t.dims[0]; i++) {
        o << delim << t[i];
        delim = ", ";
    }

    return o << "]";
}



template<typename realtype>
struct uniform_randgen {
    std::default_random_engine eng;
    std::uniform_real_distribution<realtype> dist;
    realtype next_out;

    uniform_randgen(realtype a, realtype b, uint32_t seed = 24) : dist(a,b) {
		eng.seed(seed);
        next_out = dist(eng);
    }

    realtype operator*() {
        return next_out;
    }

    uniform_randgen& operator++() {
        next_out = dist(eng);
        return *this;
    }
};

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

    //Makes things a bit easier when you have a default constructor
    Tensor() = default;

    //TODO: add constructors that might copy from a vector or something, 
    //rather than initialize an empty one
    Tensor(int const* dims) {
        int prod = copy_dims_get_strides(dims);
        storage = std::vector<T>(prod, T());
    }

    template <typename generator>
    Tensor(int const* dims, generator g) : Tensor(dims) {
        for(auto& item : storage) {
            item = *g;
            ++g;
        }
    }

    Tensor(TSpan<rank, T> const& t) : Tensor(t.dims, t.data) {}
    
    
    Tensor(std::vector<T> vec, int const* dims) : storage(std::move(vec)) {
        copy_dims_get_strides(dims);
    }

    typename std::conditional<(rank > 1), TSpan<rank-1, T>, T&>::type
    operator[] (int n) {
        return TSpan<rank, T>(*this)[n];
    }

    TSpan<rank, T> operator& () {
        return TSpan<rank, T>(*this);
    }

    TSpan<rank, T> const operator& () const {
        return TSpan<rank, T>(*this);
    }

    explicit operator TSpan<rank, T>() const {
        return TSpan<rank, T>(storage.data(), dims, strides);
    }

    int length() const {
      return dims[0];
    }

    void dump(std::ostream &o) const {
        o << "np.array(" << *this << ")";
    }

    bool operator==(Tensor<rank, T> const& other) const {
        return &(*this) == &other;
    }
};

//To random people who find this code: it would be nice
//if you made this work at compile time with nicer syntax
//thx
template<int rank, typename T>
Tensor<rank, T> make_tensor(std::initializer_list<int> const dims,
                            std::initializer_list<T> const vals
							) {
    std::vector<int> dims_vec(dims);
    std::vector<T> vals_vec(vals);

    int prod = 1;
    for (int d : dims) prod *= d;

    if (prod != static_cast<int>(vals.size())) {
        throw std::runtime_error("Given number of elements does not match dimensions");
    }

    return Tensor<rank,T>(dims_vec.data(), vals_vec.data());
}

template<int rank>
Tensor<rank, float> make_random_tensor(std::initializer_list<int> const dims,
                            uint32_t seed = 24
							) {
    std::vector<int> dims_vec(dims);

	uniform_randgen<float> gen(-1.0, 1.0);
    return Tensor<rank,float>(dims_vec.data(), gen);
}

template <int rank, typename T>
std::ostream& operator<<(std::ostream &o, Tensor<rank,T> const& t) {
    return o << TSpan<rank,T>(t);
}

//Does NOT perform any safety checks. Will default-construct all
//elements in the output
//Base case: dot product of two 1-d arrays
template<int LHS_rank, int RHS_rank, typename T>
typename std::enable_if<(LHS_rank == 1) && (RHS_rank == 1),void>::type
tensormul(
    TSpan<LHS_rank, T> const& A, 
    TSpan<RHS_rank, T> const& B,
    T& dest
) {

#ifdef DEBUG_PRINTS

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
    TSpan<LHS_rank, T> const& A, 
    TSpan<RHS_rank, T> const& B,
    TSpan<LHS_rank+RHS_rank-2, T> dest
) {
#ifdef DEBUG_PRINTS
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
    TSpan<LHS_rank, T> const& A, 
    TSpan<RHS_rank, T> const& B,
    TSpan<LHS_rank+RHS_rank-2, T> dest
) {

#ifdef DEBUG_PRINTS
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
    TSpan<LHS_rank, T> const& A, 
    TSpan<RHS_rank, T> const& B)
{
    static_assert(LHS_rank >= 1, "LHS must have positive rank");
    static_assert(RHS_rank >= 1, "RHS must have positive rank");

#ifdef DEBUG_PRINTS
    std::cout   << "Called tensormul<" 
                << LHS_rank 
                << "," 
                << RHS_rank
                << "> on \n"
    ;

    std::cout << A << "\n and \n" << B << std::endl;
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

#ifdef DEBUG_PRINTS
    auto delim = "";
    std::cout << "ret_dims = [";
    for (int i = 0; i < ret_rank; i++) {
        std::cout << delim << ret_dims[i];
        delim = ",";
    }
    std::cout << "]" << std::endl;
#endif

    Tensor<ret_rank, T> ret(ret_dims);

    if (A.dims[LHS_rank - 1] != B.dims[0]) {
        std::string msg = "Inner dimensions must agree. LHS dims = ";
        std::string delim = "[";
        for (int i = 0; i < LHS_rank; i++) {
            msg += delim + std::to_string(A.dims[i]);
            delim = ",";
        }

        delim = "], RHS dims = [";
        for (int i = 0; i < RHS_rank; i++) {
            msg += delim + std::to_string(B.dims[i]);
            delim = ",";
        }
        throw std::runtime_error(msg + "]");
    }

    tensormul(A, B, &ret);

    return ret;
}

template<int LHS_rank, int RHS_rank, typename T>
inline Tensor<LHS_rank+RHS_rank-2, T> tensormul(
    Tensor<LHS_rank, T> const& A, 
    Tensor<RHS_rank, T> const& B)
{
    return tensormul(&A, &B);
}

template <typename T>
using Matrix = Tensor<2,T>;

template <typename T>
using Vector = Tensor<1,T>;

template <typename T>
using MSpan = TSpan<2,T>;

template <typename T>
using VSpan = TSpan<1,T>;

//#ifdef DEBUG_PRINTS
//#undef DEBUG_PRINTS
//#endif

#endif