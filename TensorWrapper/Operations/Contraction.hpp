#pragma once

namespace TWrapper {
namespace detail_ {

template<typename LHS_t,typename RHS_t>
struct Contraction: public OperationBase<Contraction<LHS_t,RHS_t>>
{
    LHS_t lhs_;
    RHS_t rhs_;

    ///Rank of tensor product
    constexpr static size_t full_rank=LHS_t::size()+RHS_t::size();

    ///Rank of result
    constexpr static size_t rank=
            LHS_t::nunique(RHS_t())+RHS_t::nunique(LHS_t());

    using scalar_type=typename LHS_t::scalar_type;

    constexpr Contraction(const LHS_t& lhs,const RHS_t& rhs):
        lhs_(lhs),rhs_(rhs)
    {}


    template<TensorTypes TT>
    auto eval()const
    {
        using lhs_idx=typename LHS_t::indices;
        using rhs_idx=typename RHS_t::indices;
        using Impl_t=TensorWrapperImpl<LHS_t::rank,scalar_type,TT>;


        return Impl_t().contraction(lhs_.template eval<TT>(),
                                    rhs_.template eval<TT>(),
                                    lhs_idx(),rhs_idx());

    }
};

template<typename LHS_t,typename lhs_S, typename RHS_t, typename rhs_S>
auto operator*(const IndexedTensor<LHS_t,lhs_S>& lhs,
               const IndexedTensor<RHS_t,rhs_S>& rhs)
{

    return Contraction<IndexedTensor<LHS_t,lhs_S>,
                       IndexedTensor<RHS_t,rhs_S>>(lhs,rhs);
}



}}
