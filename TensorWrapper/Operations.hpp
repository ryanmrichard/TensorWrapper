#pragma once
#include "TensorWrapper/Traits.hpp"

/** \file Includes all of our lazy evaluation machinery.  This file is also what
 *        sets up the syntatic sugar over said lazy evalution.
 *
 * \note We use macros here because this is a header file and we
 *       don't want to pollute the global namespace with typedefs.
 */

//Detail namespace
#define TWdet TWrapper::detail_

#include "TensorWrapper/Operations/OperationBase.hpp"
#include "TensorWrapper/Operations/Convert.hpp"
#include "TensorWrapper/Operations/Permute.hpp"
#include "TensorWrapper/Operations/Add.hpp"
#include "TensorWrapper/Operations/Scale.hpp"
#include "TensorWrapper/Operations/Subtraction.hpp"
#include "TensorWrapper/Operations/IndexedTensor.hpp"
#include "TensorWrapper/Operations/Contraction.hpp"

/** \brief \relates AddOp
 *
 * Provides the syntactic sugar for adding non-indexed tensors together.
 *
 *
 */
template<typename LHS_t, typename RHS_t,
         typename TWdet::EnableIfNotAnOperation<RHS_t>::type=0>
auto operator+(const TWdet::OperationBase<LHS_t>& lhs,
               const RHS_t& rhs)
{
    using convert_t=TWdet::Convert<RHS_t>;
    using index_t=typename TWdet::GenericIndex<RHS_t::rank()>::type;
    using newr_t=TWdet::IndexedTensor<typename LHS_t::scalar_type,
                                      convert_t,index_t>;
    return TWdet::AddOp<LHS_t,newr_t>(lhs.cast(),
                              newr_t(TWdet::Convert<RHS_t>(rhs)));
}

/** \brief \relates AddOp
 *
 * Provides the syntactic sugar for adding tensors together.
 *
 */
template<typename LHS_t, typename RHS_t>
auto operator+(const TWdet::OperationBase<LHS_t>& lhs,
               const TWdet::OperationBase<RHS_t>& rhs)
{
    return TWdet::AddOp<LHS_t,RHS_t>(lhs.cast(),rhs.cast());
}


template<typename LHS_t, typename RHS_t,
         typename TWdet::EnableIfNotAnOperation<RHS_t>::type=0>
auto operator-(const TWdet::OperationBase<LHS_t>& lhs,
               const RHS_t& rhs)
{
    using convert_t=TWdet::Convert<RHS_t>;
    using index_t=typename TWdet::GenericIndex<RHS_t::rank()>::type;
    using newr_t=TWdet::IndexedTensor<typename LHS_t::scalar_type,convert_t,index_t>;
    return TWdet::SubtractionOp<LHS_t,newr_t>(lhs.cast(),
                              newr_t(TWdet::Convert<RHS_t>(rhs)));
}


template<typename LHS_t, typename RHS_t>
auto operator-(const TWdet::OperationBase<LHS_t>& lhs,
               const TWdet::OperationBase<RHS_t>& rhs)
{
    return TWdet::SubtractionOp<LHS_t,RHS_t>(lhs.cast(),rhs.cast());
}


template<typename LHS_t>
auto operator*(const TWdet::OperationBase<LHS_t>& lhs,
               typename LHS_t::scalar_type c)
{
    return TWdet::ScaleOp<LHS_t>(lhs.cast(),c);
}

template<typename RHS_t>
auto operator*(typename RHS_t::scalar_type c,
               const TWdet::OperationBase<RHS_t>& rhs)
{
    return rhs*c;
}

template<typename LHS_t, typename RHS_t>
auto operator*(const TWdet::OperationBase<LHS_t>& lhs,
               const TWdet::OperationBase<RHS_t>& rhs)
{
    return TWdet::Contraction<LHS_t,RHS_t>(lhs.cast(),rhs.cast());
}

#undef TWdet
