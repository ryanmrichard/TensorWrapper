#pragma once
#include "TensorWrapper/Traits.hpp"

/* We will need to define some free functions to get the syntactic sugar right.
 * These definitions will be messy, but these macros should help aid the reader
 * (we avoid typedefs 'cause this is a header).
 */

//Detail namespace
#define TWdet TWrapper::detail_
//An operation on the left
#define LHS_TWOp TWdet::OperationBase<LHS_t>
//An operation on the right
#define RHS_TWOp TWdet::OperationBase<RHS_t>

//An operation between two operations
#define BINARY_OP(result,sym)\
template<typename LHS_t, typename RHS_t>\
TWdet::result<LHS_t,RHS_t> operator sym(const LHS_TWOp& lhs,\
                                        const RHS_TWOp& rhs)\
{\
    const LHS_t& lhs_up=static_cast<const LHS_t&>(lhs);\
    const RHS_t& rhs_up=static_cast<const RHS_t&>(rhs);\
    return TWdet::result<LHS_t,RHS_t>(lhs_up,rhs_up);\
}

//An operation where only the left is an op
#define UNARY_OP(result,sym)\
template<typename LHS_t, typename RHS_t,\
         typename=typename TWdet::EnableIfNotAnOperation<RHS_t>::type>\
TWdet::result<LHS_t,TWdet::Convert<RHS_t>> operator sym(const LHS_TWOp& lhs,\
                                                        const RHS_t& rhs)\
{\
    const LHS_t& lhs_up=static_cast<const LHS_t&>(lhs);\
    using new_rhs_t=TWdet::Convert<RHS_t>;\
    return TWdet::result<LHS_t,new_rhs_t>(lhs_up,new_rhs_t(rhs));\
}

#include "TensorWrapper/Operations/OperationBase.hpp"
#include "TensorWrapper/Operations/Convert.hpp"
#include "TensorWrapper/Operations/Add.hpp"
#include "TensorWrapper/Operations/Scale.hpp"
#include "TensorWrapper/Operations/Subtraction.hpp"
#include "TensorWrapper/Operations/IndexedTensor.hpp"
#include "TensorWrapper/Operations/Contraction.hpp"

#undef UNARY_OP
#undef BINARY_OP
#undef RHS_TWOp
#undef LHS_TWOp
#undef TWdet
