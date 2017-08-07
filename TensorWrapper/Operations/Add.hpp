#pragma once
#include "TensorWrapper/Operations/OperationBase.hpp"
#include "TensorWrapper/TensorImpls.hpp"

namespace TWrapper {
namespace detail_ {

template<typename LHS_t, typename RHS_t>
struct AddOp: public OperationBase<AddOp<LHS_t,RHS_t>>{
    LHS_t lhs_;
    RHS_t rhs_;
    AddOp(const LHS_t& lhs, const RHS_t& rhs):
        lhs_(lhs),rhs_(rhs)
    {}

    constexpr static size_t rank=LHS_t::rank;
    using scalar_type=typename LHS_t::scalar_type;

    template<TensorTypes TT>
    auto eval()const->
     decltype(TensorWrapperImpl<rank,scalar_type,TT>().add(lhs_.template eval<TT>(),
                                                           rhs_.template eval<TT>()))
    {
        TensorWrapperImpl<rank,scalar_type,TT> impl;
        return impl.add(lhs_.template eval<TT>(),
                        rhs_.template eval<TT>());
    }

};

}}//End namespaces

BINARY_OP(AddOp,+)
UNARY_OP(AddOp,+)
