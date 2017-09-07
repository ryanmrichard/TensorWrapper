#pragma once
#include "TensorWrapper/TMUtils/TypeComparisons.hpp"

/** \file Implements our lazy trace machinery.
 */

namespace TWrapper {
namespace detail_ {

/** \brief The class that implements our lazy trace.
 *
 *
 * \tparam Tensor_t The type of the expression to take the trace of
 */
template<typename Tensor_t>
struct Trace: public OperationBase<Trace<Tensor_t>> {
    ///The tensor
    Tensor_t lhs_;

    template<typename T>
    struct GetUniqueHelper;

    template<typename...Args>
    struct GetUniqueHelper<Indices<Args...>>
    {
        using type=typename IndicesFromTuple<
                                typename GetUnique<Args...>::type>::type;
    };

    ///Indices after trace
    using indices=typename GetUniqueHelper<typename Tensor_t::indices>::type;

    ///Rank of result
    constexpr static size_t rank=indices::size();

    ///Type of the tensor's elements
    using scalar_type=typename Tensor_t::scalar_type;

    /** \brief Makes a new Trace instance by copying the tensor
     *
     * \param[in] lhs The tensor
     * \throws ??? Throws if lhs's constructor throws. Strong throw guarantee.
     */
    constexpr Trace(const Tensor_t& lhs):
        lhs_(lhs)
    {}

    std::array<size_t,rank> dimensions()const
    {
        const auto l2r=indices().get_map(typename Tensor_t::indices());
        const auto ldims=lhs_.dimensions();
        std::array<size_t,rank> rv;
        size_t counter=0;
        for(size_t x: l2r)
            rv[counter++]=ldims[x];
        return rv;
    }

    /** \brief Actually runs the trace.
     *
     * \returns Whatever the backend returns
     * \tparam TT The enum of the backend to use for the contraction.
     * \throws ??? Throws if the backend throws.  Strong throw guarantee.
     */
    template<TensorTypes TT>
    auto eval()const
    {
        using Impl_t=TensorWrapperImpl<Tensor_t::rank,scalar_type,TT>;
        return Impl_t().template trace<typename Tensor_t::indices>(
                    lhs_.template eval<TT>());
    }
};

}}//End namespaces
