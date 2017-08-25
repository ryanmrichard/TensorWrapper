#pragma once
#include "TensorWrapper/Indices.hpp"
#include "TensorWrapper/Operations/OperationBase.hpp"

/** \file This file contains a class for associating indices with a tensor
 *        regardless of what the wrapped types indices actually are.
 *
 */
namespace TWrapper {
namespace detail_ {

/** \brief This is basically a slightly dressed Convert class that can have the
 *         indices set manually.
 *
 *  Particularly when starting from a TensorWrapperBase we need some way to
 *  associate with a tensor its indices.  Unlike the Convert class which
 *  primarily exists to wrap an object in our common API, the IndexedTensor
 *  class primarily exists to associate indices with a tensor.
 *
 *  \tparam T The type of the tensor's elements
 *  \tparam Tensor_t The type of the tensor we are wrapping (is expected to be
 *                   derived from OperationBase)
 *  \tparam Index_t The index to associate with the tensor.
 *
 */
template<typename T, typename Tensor_t,typename Index_t>
struct IndexedTensor: public OperationBase<IndexedTensor<T,Tensor_t,Index_t>>
{
    ///The tensor we are wrapping
    Tensor_t tensor_;

    ///The indices of the tensor
    using indices=Index_t;

    ///The rank of the tensor
    constexpr static size_t rank=Index_t::size();

    ///The type of the elements in the tensor
    using scalar_type=T;

    /** \brief Makes a new IndexedTensor by copying a tensor.
     *
     * \param[in] tensor The tensor to wrap.
     * \throws ??? Throws if Tensor_t's constructor throws.  Strong throw
     *             guarantee.
     */
    constexpr IndexedTensor(const Tensor_t& tensor)
        :tensor_(tensor)
    {}

    std::array<size_t,rank> dimensions()const
    {
        return tensor_.dimensions();
    }

    /** \brief Returns the result of calling eval on the wrapped tensor.
     *
     *  \returns The result of calling eval on the wrapped tensor.
     *  \throws ??? Throws if Tensor_t::eval throws.  Strong throw guarantee.
     *  \tparam TT The enum of the backend to use.
     */
    template<TensorTypes TT>
    auto eval()const->decltype(tensor_.template eval<TT>())
    {
        return tensor_.template eval<TT>();
    }

};

}}//End namespaces
