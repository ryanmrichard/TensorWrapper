#pragma once
#include "TensorWrapper/Indices.hpp"
#include "TensorWrapper/Operations/OperationBase.hpp"
namespace TWrapper {
namespace detail_ {

/** \brief This is basically a slightly dressed Convert class.
 *
 */
template<typename Tensor_t,typename Index_t>
struct IndexedTensor: public OperationBase<IndexedTensor<Tensor_t,Index_t>>
{
    Tensor_t tensor_;
    using indices=Index_t;
    constexpr static size_t rank=Tensor_t::rank;
    using scalar_type=typename Tensor_t::scalar_type;

    constexpr IndexedTensor(const Tensor_t& tensor)
        :tensor_(tensor)
    {}

    template<TensorTypes TT>
    auto eval()const->decltype(tensor_.template eval<TT>())
    {
        return tensor_.template eval<TT>();
    }


};


}}//End namespaces
