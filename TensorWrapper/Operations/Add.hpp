#pragma once
#include "TensorWrapper/Operations/OperationBase.hpp"
#include "TensorWrapper/TensorImpls.hpp"

/** \file This file contains the implementation of our lazy addition operator.
 */

namespace TWrapper {
namespace detail_ {

/** \brief The class responsible for implementing lazy addition.
 *
 *  Given an expression like: \f$A+B\f$ this class will store a copy of both
 *  \f$A\f$ and \f$B\f$.  Then when eval is called it will actually add \f$A\f$
 *  and \f$B\f$ together by calling the corresponding tensor backend.
 *
 *  Because of how C++ evalutes expressions it will be the case that \p LHS_t is
 *  always derived from OperationBase; however, \p RHS_t may not be.  Therefore
 *  we rely on convert's specializations and always wrap \p RHS_t in a convert
 *  call to guarantee the API we want.  Similarly, this is why we grab rank,
 *  indices, and scalar_type form the left type.
 *
 *  \tparam LHS_t The type of \f$A\f$.  Expected to derive from OperationBase
 *  \tparam RHS_t The type \f$B\f$.
 */
template<typename LHS_t, typename RHS_t>
struct AddOp: public OperationBase<AddOp<LHS_t,RHS_t>>{
    ///The tensor on the left side of the + sign
    LHS_t lhs_;

    ///The rank of the two tensors involved in the addition
    constexpr static size_t rank=LHS_t::rank;

    ///The type of the elements of the two tensors
    using scalar_type=typename LHS_t::scalar_type;

    ///The indices after adding
    using indices=typename LHS_t::indices;

    ///A permutation of the converted rhs
    using permute_t=Permutation<indices,typename RHS_t::indices,RHS_t>;

    ///The tensor on the right side of the + sign all wrapped up and ready to go
    permute_t rhs_;

    /** \brief Makes a new AddOp by copying the tensors on the two sides of the
     *  + sign.
     *
     *  \param[in] lhs The tensor on the left of the plus sign.
     *  \param[in] rhs The tensor or the right of the plus sign.
     *  \throws ??? Throws if LHS_t's or RHS_t's constructor throws.  Strong
     *          throw guarnatee.
     */
    AddOp(const LHS_t& lhs, const RHS_t& rhs):
        lhs_(lhs),rhs_(rhs)
    {}

    std::array<size_t,rank> dimensions()const
    {
        return lhs_.dimensions();
    }

    /** \brief Actually evaluates the addition
     *
     *   When called this function will call eval on both the left and right
     *   tensors forwarding the results to the requested backend to be added.
     *   Whatever the backend returns is what this function returns.
     *
     *  \tparam TT The enum of the backend to use for the addition.
     *  \throws ??? Throws if the backend throws.  Throw guarantee is the same
     *              as the backend.
     *  \returns Whatever the backend returns
     */
    template<TensorTypes TT>
    auto eval()const->decltype(TensorWrapperImpl<rank,scalar_type,TT>().
                               add(lhs_.template eval<TT>(),
                                   rhs_.template eval<TT>()))
    {
        TensorWrapperImpl<rank,scalar_type,TT> impl;
        return impl.add(lhs_.template eval<TT>(),
                        rhs_.template eval<TT>());
    }

};

}}//End namespaces
