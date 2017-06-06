#pragma once
#include <array>
#include "TensorWrapper/Shape.hpp"
#include "TensorWrapper/Contraction.hpp"
namespace TWrapper {
namespace detail_ {

/** \brief A class that's always false
 *
 *  Apparently, do to some quirk of the C++ language you can't do
 *
 *  static_assert(false,message here);
 *
 *  because the assert is always false.  As soon as the compiler sees this it
 *  crashes.  It never goes on to find the specializations which will not
 *  contain said assertion  The work around is to make the following class,
 *  which we know is always false, but the compiler doesn't.  When the compiler
 *  sees it, it thinks that concievably it could be true for some value of T.
 *  Thus it does not immediatly crash and goes on to find the specialization.
 *
 *  The joys of template metaprogramming...
 *
 */
template <typename T> struct always_false : std::false_type {};

/** \brief This is the "base" class that wraps the calls to each Tensor
 *  library.
 *
 *  Base is in quotes because implementation is actually accomplished via
 *  template specialization.  To register a tensor library every member of this
 *  class must be specialized for your tensor library.
 */
template<size_t rank, typename T, typename Tensor_t>
struct TensorWrapperImpl {
    ///Returns the dimensions of a tenosr
    Shape<rank> dims(const Tensor_t&)const
    {
        static_assert(always_false<Tensor_t>::value,"Tensor defines no dims() API");
    }

    ///Returns a specific index
    T get_value(const std::array<size_t,rank>&)const
    {
        static_assert(always_false<Tensor_t>::value,"Tensor defines no get_value() API");
    }

    ///Returns true if two tensors are equal
    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t&, const RHS_t&)const
    {
        static_assert(always_false<Tensor_t>::value,"Tensor defines no are_equal() API");
        return false;//Silence compiler warning
    }

    ///Executes a contraction
    template<typename LHS_t, typename RHS_t>
    void contract(const Contraction<LHS_t,RHS_t>&)const
    {
        static_assert(always_false<Tensor_t>::value,"Tensor defines no contract() API");
    }

};

///A simple struct for converting between wrapped tensors
template<typename new_type>
struct TensorConverter: public std::false_type {
    template<typename Tensor_t>
    new_type operator()(const Tensor_t&)const{
        static_assert(always_false<Tensor_t>::value,"conversion not defined");
    }
};


}}//End namespaces
