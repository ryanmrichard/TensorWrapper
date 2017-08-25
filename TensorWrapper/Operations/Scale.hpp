#pragma once
#include "TensorWrapper/Operations/OperationBase.hpp"
#include "TensorWrapper/TensorImpls.hpp"

/** \file Contains our lazy evaluation implementation of scaling.
 */

namespace TWrapper {
namespace detail_ {

/** \brief Class for scaling a tensor by a scalar.
 *
 *
 *  \tparam LHS_t The type of the tensor to scale.  Expected to inherit from
 *                OperationBase.
 */
template<typename LHS_t>
struct ScaleOp: public OperationBase<ScaleOp<LHS_t>>{

    ///The tensor to scale
    LHS_t lhs_;

    ///The type of the elements in the tensor as well as the scalar
    using scalar_type=typename LHS_t::scalar_type;

    ///The rank of the tensor
    constexpr static size_t rank=LHS_t::rank;

    ///The indices of the tensor
    using indices=typename LHS_t::indices;

    ///The value to scale by
    scalar_type scalar_;

    /** \brief Makes a new lazy evalution operator that will scale a tensor.
     *
     *
     *  \param[in] lhs The type of the tensor to scale
     *  \param[in] scale The value to scale the tensor by.
     *  \throws ??? Throws if LHS_t's constructor throws.  Strong throw
     *              guarantee.
     */
    ScaleOp(const LHS_t& lhs, scalar_type scale):
        lhs_(lhs),scalar_(scale)
    {}

    /** \brief Actually scales the tensor.
     *
     * \tparam TT The enum for the backend we want to use for the scaling.
     * \returns Whatever the backend returns.
     * \throws ??? Throws if the backend throws.  Strong throw guarantee.
     */
    template<TensorTypes TT>
    auto eval()const
    {
        TensorWrapperImpl<rank,scalar_type,TT> impl;
        return impl.scale(lhs_.template eval<TT>(),scalar_);
    }
}; //End ScaleOp class

}}//End namespaces
