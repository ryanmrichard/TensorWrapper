#pragma once

/** \file Implements our lazy contraction machinery.
 */

namespace TWrapper {
namespace detail_ {

/** \brief The class that implements our lazy contraction.
 *
 *
 * \tparam LHS_t The type of the tensor on the left of the times sign.
 *               Expected to actually have indices.
 * \tparam RHS_t The type of the tensor on the right side of the times sign.
 *               Expected to actually have indices.
 */
template<typename LHS_t,typename RHS_t>
struct Contraction: public OperationBase<Contraction<LHS_t,RHS_t>> {
    ///The tensor on the left
    LHS_t lhs_;

    ///The tensor on the right
    RHS_t rhs_;

    ///Indices after contraction
    using indices=typename FreeIndices<typename LHS_t::indices,
                                       typename RHS_t::indices>::type;

    ///Rank of result
    constexpr static size_t rank=indices::size();

    ///Type of the tensor's elements
    using scalar_type=typename LHS_t::scalar_type;

    /** \brief Makes a new Contraction instance by copying the two tensors
     *
     * \param[in] lhs The tensor on the left of the * sign
     * \param[in] rhs The tensor on the right of the * sign
     * \throws ??? Throws if either LHS_t's or RHS_t's constructor throws.
     *             Strong throw guarantee.
     */
    constexpr Contraction(const LHS_t& lhs, const RHS_t& rhs):
        lhs_(lhs),rhs_(rhs)
    {}

    std::array<size_t,rank> dimensions()const
    {
        auto free=get_free(typename LHS_t::indices(),
                           typename RHS_t::indices());
        auto ldims=lhs_.dimensions();
        auto rdims=rhs_.dimensions();
        std::array<size_t,rank> rv;
        size_t counter=0;
        for(size_t i : free.first)
            rv[counter++]=ldims[i];
        for(size_t i : free.second)
            rv[counter++]=rdims[i];
        return rv;
    }

    /** \brief Actually runs the contraction.
     *
     * \returns Whatever the backend returns
     * \tparam TT The enum of the backend to use for the contraction.
     * \throws ??? Throws if the backend throws.  Strong throw guarantee.
     */
    template<TensorTypes TT>
    auto eval()const
    {
        using Impl_t=TensorWrapperImpl<LHS_t::rank,scalar_type,TT>;
        return Impl_t().contraction(lhs_.template eval<TT>(),
                                    rhs_.template eval<TT>(),
                                    typename LHS_t::indices(),
                                    typename RHS_t::indices());
    }
};

}}//End namespaces
