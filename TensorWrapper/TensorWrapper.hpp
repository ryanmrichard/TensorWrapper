#pragma once
#include <array>
#include "TensorWrapper/DisableUselessWarnings.hpp"
#include "TensorWrapper/TensorWrapperBase.hpp"

/** \file This is the main include file for the TensorWrapper library
 */
namespace TWrapper {

template<size_t Rank, typename T, detail_::TensorTypes T1>
class TensorWrapper: public TensorWrapperBase<Rank,T>{
private:
    detail_::TensorWrapperImpl<Rank,T,T1> impl_;
public:
    ///The type of an instance of this class
    using my_type=TensorWrapper<Rank,T,T1>;

    ///The type of the base class
    using base_type=TensorWrapperBase<Rank,T>;

    ///The type wrapped by this class
    using wrapped_t=typename detail_::TensorWrapperImpl<Rank,T,T1>::type;

    /** \brief The constructor used for wrapping the native classes of a tensor
     *         backend.
     *
     *  \note Whether this is a deep or shallow copy depends on the backend.
     *
     *  \param[in] other An instance from our backend that is implicitly
     *                   convertible to our wrapped type.
     */
    template<typename RHS_t>
    TensorWrapper(const RHS_t& other):
        base_type(TensorPtr(other),T1)
    {}


    /** \brief Constructor for allocating the memory of the tensor
     *
     *  After calling this constructor the memory is allocated, but in an
     *  undefined initialization state.
     */
    TensorWrapper(const std::array<size_t,Rank>& dims):
        base_type(T1)
    {
        base_type::tensor_=TensorPtr(impl_.allocate(dims));
    }

    TensorWrapper()=default;
    TensorWrapper(my_type&&)=default;
    my_type& operator=(const my_type&)=default;
    my_type& operator=(my_type&&)=default;

};

//template<typename T, detail_::TensorTypes T1>
//std::pair<TensorWrapper<1,T,T1>,TensorWrapper<2,T,T1>>
//self_adjoint_eigen_solver(const TensorWrapper<2,T,T1>& matrix){
//    detail_::TensorWrapperImpl<2,T,T1> impl;
//    auto rv=impl.self_adjoint_eigen_solver(matrix.tensor());
//    return {TensorWrapper<1,T,T1>(rv.first),
//            TensorWrapper<2,T,T1>(rv.second)};
//}

namespace detail_ {

//template<size_t rank,typename T,TensorTypes LHS_t,TensorTypes RHS_t>
//struct TensorConverter{
//    /** \brief Generic operation for converting two tensors.
//     *
//     *   For performance it is recommended that you specialize this class.  However,
//     *   as this will require N choose 2 specializations, where N is the number of
//     *   backends, this can quickly become too much, especially when most of them
//     *   are going to amount to deep copies anyways.  This implementation will do
//     *   the deep copy for you.  Because the tensor we are converting from may not
//     *   all be local we can't just call get_memory.  Instead, we need to get the
//     *   whole tensor as a slice (which guarantees that our tensor is local) and
//     *   then call get_memory.
//     *
//     */
//    template<typename Tensor_t>
//    static auto convert(const Tensor_t& rhs)
//    {
//        TensorWrapper<rank,T,RHS_t> wrapped_rhs(rhs);
//        std::array<size_t,rank> start{};
//        auto local=wrapped_rhs.get_slice(start,wrapped_rhs.dims());
//        TensorWrapper<rank,T,LHS_t> rv(wrapped_rhs.dims());
//        rv.set_slice(local.get_memory());
//        return rv;
//    }

//};

}

}
//All the operator overload definitions
#include "TensorWrapper/TWOperators.hpp"

#include "TensorWrapper/EnableUselessWarnings.hpp"
