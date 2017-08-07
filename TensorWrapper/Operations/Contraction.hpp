#pragma once

namespace TWrapper {
namespace detail_ {

template<typename LHS_t,typename RHS_t>
struct Contraction: public OperationBase<Contraction<LHS_t,RHS_t>>
{
    LHS_t lhs_;
    RHS_t rhs_;

    ///Resulting indices
    using indices=typename FreeIndices<typename LHS_t::indices,
                                       typename RHS_t::indices>::type;

    ///Rank of result
    constexpr static size_t rank=indices::size();

    using scalar_type=typename LHS_t::scalar_type;



    constexpr Contraction(const LHS_t& lhs, const RHS_t& rhs):
        lhs_(lhs),rhs_(rhs)
    {}


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


template<typename T,typename LHS_t,typename lhs_S,
                     typename RHS_t, typename rhs_S>
auto operator*(const IndexedTensor<T,LHS_t,lhs_S>& lhs,
               const IndexedTensor<T,RHS_t,rhs_S>& rhs)
{

    return Contraction<IndexedTensor<T,LHS_t,lhs_S>,
                       IndexedTensor<T,RHS_t,rhs_S>>(lhs,rhs);
}

template<typename T,typename LHS_t,typename RHS_t1,
                    typename RHS_t, typename rhs_S>
auto operator*(const Contraction<LHS_t,RHS_t1>& lhs,
               const IndexedTensor<T,RHS_t,rhs_S>& rhs)
{

    return Contraction<Contraction<LHS_t,RHS_t1>,
                       IndexedTensor<T,RHS_t,rhs_S>>(lhs,rhs);
}

}}
