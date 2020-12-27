#ifndef BASE_TYPES_H
#define BASE_TYPES_H 1

#include <iostream>
#include <string>
#include <cstdint> //For uint64_t
#include <stdexcept> //For std::runtime_error
#include <utility> //For std::move
#include <memory> //For std::shared_ptr
#include <vector>
#include "tensor.h"


//Should layer_data be immutable?
//It would save a lot of headaches, and do we really
//need mutability?
class layer_data {
private:
    bool derived_object_is_mutable;
    uint64_t    tag; //Maybe we could use this for
                     //fast equality testing?
    static uint64_t next_tag;
    std::string type; //For pretty-printing. Do we need this?

public:
    bool operator==(layer_data const& other) {
        return tag == other.tag;
    }
    
    bool operator!=(layer_data const& other) {
        return tag != other.tag;
    }

    virtual std::string to_string() const {
        return std::string(derived_object_is_mutable ? "M" : "Imm") 
               + "utable object of type "
               + type
               + " with tag "
               + std::to_string(tag);
    }

    //See: https://isocpp.org/wiki/faq/virtual-functions#virtual-ctors
    virtual layer_data* clone() const = 0;

    virtual ~layer_data() {}

protected:
    //The ctors/operators are protected so that people can't use 
    //layer_datas directly. Instead, the derived classes need to
    //call them. This is especially important for copy constructing,
    //since we only want to do derived1 = derived2 (it makes no sense
    //to say base1 = base2).
    layer_data(std::string type, bool m = true) : 
        derived_object_is_mutable(m),
        tag(next_tag++),
        type(type)
    {}

    //When copy constructing, get a new tag if the derived object
    //is mutable. Otherwise, we can reuse the same tag
    layer_data(layer_data const& other) {
        derived_object_is_mutable = other.derived_object_is_mutable;
        type = other.type;

        if (derived_object_is_mutable) {
            tag = next_tag++;
        } else {
            tag = other.tag;
        }
        #ifdef DEBUG
        std::cout << "Called layer_data copy ctor" << std::endl;
        #endif
    }

    //When move-constructing, use same tag regardless of mutability
    layer_data(layer_data const&& other) {
        derived_object_is_mutable = other.derived_object_is_mutable;
        type = std::move(other.type);
        tag = other.tag;

        #ifdef DEBUG
        std::cout << "Called layer_data move ctor" << std::endl;
        #endif
    }

    //When assigning, need to check matching mutability
    layer_data& operator=(layer_data const& other) {
        if (derived_object_is_mutable != other.derived_object_is_mutable) {
            throw new std::runtime_error("Illegal assignment between layer_datas of different mutability");
        }
        if (type != other.type) {
            throw new std::runtime_error("Illegal assignment between layer_datas of different type");
        }

        if (derived_object_is_mutable) {
            tag = next_tag++;
        } else {
            tag = other.tag;
        }
        #ifdef DEBUG
        std::cout << "Called layer_data operator=" << std::endl;
        #endif
        return *this;
    }

    //When move-assigning, need to check matching mutability, 
    //but always use same tag regardless of mutability
    layer_data& operator=(layer_data const&& other) {
        if (derived_object_is_mutable != other.derived_object_is_mutable) {
            throw new std::runtime_error("Illegal assignment between layer_datas of different mutability");
        }
        if (type != other.type) {
            throw new std::runtime_error("Illegal assignment between layer_datas of different type");
        }

        tag = other.tag;
        
        #ifdef DEBUG
        std::cout << "Called layer_data move assignment operator" << std::endl;
        #endif
        return *this;
    }
};


struct layer {
    virtual ~layer() {}

    //feed-forward
    virtual Matrix<float> ff(MSpan<float> const& x) = 0;

    //backprop
    virtual Matrix<float> bp(MSpan<float> const& x, MSpan<float> const& dy, float lr) = 0;

    //For debugging
    virtual void dump(std::ostream &o) const { o << "(does not support dumping)"; }

};



std::ostream& operator<<(std::ostream &o, layer const& l);


struct layer_generalized {
    virtual ~layer_generalized() {}

    //feed-forward
    virtual std::shared_ptr<layer_data> ff(std::shared_ptr<layer_data const> x) = 0;

    //backprop
    virtual std::shared_ptr<layer_data> bp(std::shared_ptr<layer_data const> x, std::shared_ptr<layer_data const> dy, float lr) = 0;
};

struct cost_fn {
    virtual ~cost_fn() {}

    //calculate cost
    virtual float cc(MSpan<float> const& x, MSpan<float> const& actual) = 0;

    //get gradient
    virtual Matrix<float> gg(MSpan<float> const& x, MSpan<float> const& actual) = 0;
};

struct cost_fn_generalized {
    virtual ~cost_fn_generalized() {}

    //calculate cost
    //Should this always return a float?
    virtual float cc(std::shared_ptr<layer_data const> x, std::shared_ptr<layer_data const> actual) = 0;

    //get gradient
    virtual std::shared_ptr<layer_data> gg(std::shared_ptr<layer_data const> x, std::shared_ptr<layer_data const> actual) = 0;
};

struct activation_fn {
    virtual ~activation_fn() {}

    //Forward
    virtual float operator()(float x) = 0;
    
    //Derivative
    virtual float operator[](float x) = 0;
};


#endif