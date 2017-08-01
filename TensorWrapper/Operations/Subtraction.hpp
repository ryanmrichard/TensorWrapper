#pragma once
#include "TensorWrapper/Operations/OperationBase.hpp"
#include "TensorWrapper/TensorImpls.hpp"

namespace TWrapper {
namespace detail_ {

template<typename LHS_t, typename RHS_t>
struct SubtractionOp : public OperationBase<SubtractionOp<LHS_t,RHS_t>> {
    constexpr static size_t rank=LHS_t::rank;
    using scalar_type=typename LHS_t::scalar_type;

    LHS_t lhs_;
    RHS_t rhs_;
    SubtractionOp(const LHS_t& lhs, const RHS_t& rhs):
        lhs_(lhs),rhs_(rhs)
    {}

    template<TensorTypes TT>
    auto eval()const
    {
        TensorWrapperImpl<rank,scalar_type,TT> impl;
        return impl.subtract(lhs_.template eval<TT>(),
                             rhs_.template eval<TT>());
    }
};

}}//End namespaces

BINARY_OP(SubtractionOp,-)
UNARY_OP(SubtractionOp,-)
