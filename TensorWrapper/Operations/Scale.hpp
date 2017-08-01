#pragma once
#include "TensorWrapper/Operations/OperationBase.hpp"
#include "TensorWrapper/TensorImpls.hpp"

namespace TWrapper {
namespace detail_ {

template<typename LHS_t, typename T>
struct ScaleOp: public OperationBase<ScaleOp<LHS_t,T>>{
    LHS_t lhs_;
    T scalar_;

    constexpr static size_t rank=LHS_t::rank;
    using scalar_type=typename LHS_t::scalar_type;

    ScaleOp(const LHS_t& lhs, T scale):
        lhs_(lhs),scalar_(scale)
    {}

    template<TensorTypes TT>
    auto eval()const->
        decltype(TensorWrapperImpl<rank,scalar_type,TT>().scale(
                     lhs_.template eval<TT>(),scalar_))
    {
        TensorWrapperImpl<rank,scalar_type,TT> impl;
        return impl.scale(lhs_.template eval<TT>(),scalar_);
    }
};

}}//End namespaces

template<typename LHS_t, typename T=typename LHS_t::scalar_type>
TWdet::ScaleOp<LHS_t,T>
operator*(const LHS_TWOp& lhs,T rhs)
{
    const LHS_t& lhs_up=static_cast<const LHS_t&>(lhs);
    return TWdet::ScaleOp<LHS_t,T>(lhs_up,rhs);
}
