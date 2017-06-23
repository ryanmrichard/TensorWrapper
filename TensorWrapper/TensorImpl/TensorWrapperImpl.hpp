#pragma once
#include <array>
#include "TensorWrapper/Shape.hpp"
#include "TensorWrapper/Contraction.hpp"
#include "TensorWrapper/MemoryBlock.hpp"

namespace TWrapper {
namespace detail_ {

/** \brief A class that's always false
 *
 *  Apparently, due to some quirk of the C++ language you can't do
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
 *  class must be specialized for your tensor library.  Regardless of how it is
 *  implemented, this class serves as the API all tensor libraries must satisfy
 *  in order to work with TensorWrapper.
 *
 *  \note Since specialization requires redefining all APIs we do not
 *        technically have to specify these functions here, we do so in the
 *        hope that the resulting compiler errors will be easier to comprehend
 *        than those resulting from such a function not existing.
 *
 *  \tparam rank The rank of the tensor the implementation is wrapping
 *  \tparam T The type of data in the tensor
 *  \tparam Tensor_t The enumeration for the type
 */
template<size_t rank, typename T, TensorTypes My_t>
struct TensorWrapperImpl {

    ///Must define a typedef type that is the basic tensor type
    ///Must define the type of a converter to this class
    using array_t=std::array<size_t,rank>;

    ///Returns the dimensions of a tensor
    template<typename Tensor_t>
    Shape<rank> dims(const Tensor_t&)const
    {
        static_assert(always_false<T>::value,"Tensor defines no dims() API");
    }

    template<typename Tensor_t>
    void get_memory(const Tensor_t&)const
    {
        static_assert(always_false<T>::value,"Tensor defines no const get_memory() API");
    }

    template<typename Tensor_t>
    void get_memory(Tensor_t&)const
    {
        static_assert(always_false<T>::value,"Tensor defines no get_memory() API");
    }

    template<typename Tensor_t>
    void slice(const Tensor_t&,const array_t&,const array_t&)const
    {
        static_assert(always_false<T>::value,"Tensor defines no slice() API");
    }

    ///Returns true if two tensors are equal
    template<typename LHS_t, typename RHS_t>
    void are_equal(const LHS_t&, const RHS_t&)const
    {
        static_assert(always_false<T>::value,"Tensor defines no are_equal() API");
    }

    ///Executes a contraction
    template<typename...Args>
    void contract(const Contraction<Args...>&)const
    {
        static_assert(always_false<T>::value,"Tensor defines no contract() API");
    }

    ///Scales the tensor
    template<typename RHS_t>
    void scale(const RHS_t&,double)const
    {
        static_assert(always_false<T>::value,"Tensor defines no scale() API");
    }

    ///Adds to the tensor
    template<typename LHS_t, typename RHS_t>
    void add(const LHS_t&,const RHS_t&)const
    {
        static_assert(always_false<T>::value,"Tensor defines no add() API");
    }

    ///Subtracts from the tensor
    template<typename LHS_t,typename RHS_t>
    void subtract(const LHS_t&,const RHS_t&,double)const
    {
        static_assert(always_false<T>::value,"Tensor defines no subtract() API");
    }

    ///Diagonalizes a tensor
    template<typename Tensor_t>
    void eigen_solver(const Tensor_t&)const
    {
        static_assert(always_false<T>::value,"Tensor defines no eigen_solver() API");
    }
};

template<typename Tensor_t>
struct TensorWrapperImplTraits;

template<size_t R, typename T, TensorTypes TensorType>
struct TensorWrapperImplTraits<TensorWrapperImpl<R,T,TensorType>>{
    using type=TensorWrapperImpl<R,T,TensorType>;
    using scalar_type=T;
    static const size_t rank=R;
    static const TensorTypes tensor_type=TensorType;
};

}}//End namespaces
