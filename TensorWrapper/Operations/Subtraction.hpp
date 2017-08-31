#pragma once
#include "TensorWrapper/Operations/OperationBase.hpp"
#include "TensorWrapper/TensorImpls.hpp"

/** \file This file contains our operator for lazyily evaluating subtraction.
 */
namespace TWrapper {
namespace detail_ {


/** \brief The class responsible for implementing lazy subtraction.
 *
 *  Given an expression like: \f$A-B\f$ this class will store a copy of both
 *  \f$A\f$ and \f$B\f$.  Then when eval is called it will actually subtract
 *  \f$A\f$ and \f$B\f$ by calling the corresponding tensor backend.
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
struct SubtractionOp : public OperationBase<SubtractionOp<LHS_t,RHS_t>> {
    ///The tensor on the left side of the - sign
    LHS_t lhs_;

    ///The tensor on the right side of the - sign
    RHS_t rhs_;

    ///The rank of the two tensors involved in the subtraction
    constexpr static size_t rank=LHS_t::rank;

    ///The type of the two tensors' elements
    using scalar_type=typename LHS_t::scalar_type;

    ///The indices after subtracting
    using indices=typename LHS_t::indices;


    /** \brief Makes a new SubtractionOp by copying the tensors on the two sides
     *  of the - sign.
     *
     *  \param[in] lhs The tensor on the left of the minus sign.
     *  \param[in] rhs The tensor or the right of the mins sign.
     *  \throws ??? Throws if LHS_t's or RHS_t's constructor throws.  Strong
     *          throw guarnatee.
     */
    SubtractionOp(const LHS_t& lhs, const RHS_t& rhs):
        lhs_(lhs),rhs_(rhs)
    {}

    std::array<size_t,rank> dimensions()const
    {
        return lhs_.dimensions();
    }

    /** \brief Actually evaluates the subtraction.
     *
     * \returns Whatever the backend returns
     * \tparam TT The tensor backend to use
     * \throws??? Throws if the backend throws.  Strong throw guarantee.
     */
    template<TensorTypes TT>
    auto eval()const
    {
        TensorWrapperImpl<rank,scalar_type,TT> impl;
        return impl.template subtract<typename LHS_t::indices,
                                      typename RHS_t::indices>(
                    lhs_.template eval<TT>(),rhs_.template eval<TT>());
    }
};

}}//End namespaces
