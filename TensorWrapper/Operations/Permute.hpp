#pragma once
#include "TensorWrapper/Operations/OperationBase.hpp"
#include "TensorWrapper/TensorImpls.hpp"

/** \file This file contains the implementation of our lazy permutation operator.
 *        It is implemented in terms of two classes a primary template and a
 *        specialization.  The specialization is essentially a null-op for when
 *        no permutation is required.
 */
namespace TWrapper {
namespace detail_{

/** \brief The class responsible for lazily permuting a tensor.
 *
 *  This is the primary template and handles the case when we actually need to
 *  do a permutation.
 *
 */
template<typename NewIdx, typename OldIdx, typename Tensor_t>
struct Permutation: public OperationBase<Permutation<NewIdx,OldIdx,Tensor_t>> {
    ///The tensor to permute
    Tensor_t t_;

    /** \brief Makes a new Permutation operation by copying \p t
     *
     *  \param[in] The tensor to permute.
     *  \throws ??? Throws if Tensor_t's constructor throws.  Strong throw
     *              guarantee.
     *
     */
    Permutation(const Tensor_t& t):t_(t){}

    ///The rank of the tensor
    constexpr static size_t rank=Tensor_t::rank;

    ///The type of the elements in the tensor
    using scalar_type=typename Tensor_t::scalar_type;

    ///The indices after permutation
    using indices=NewIdx;

    std::array<size_t,rank> dimensions()const
    {
        std::array<size_t,rank> rv,old2new=OldIdx::get_map(NewIdx());
        auto dims=t_.dimensions();
        for(size_t i=0;i<rank;++i)
        {
            if(old2new[i]==rank)
                throw std::runtime_error("Indices do not match");
            rv[old2new[i]]=dims[i];
        }
        return rv;
    }

    /** \brief Actually evaluates the permutation.
     *
     *
     */
    template<TensorTypes TT>
    auto eval()const->decltype(TensorWrapperImpl<rank,scalar_type,TT>().
                               permute(t_.template eval<TT>(),
                                       OldIdx::get_map(NewIdx())))
    {
        TensorWrapperImpl<rank,scalar_type,TT> impl;
        return impl.permute(t_.template eval<TT>(),
                            OldIdx::get_map(NewIdx()));
    }

};//End primary Permuation template

/** \brief Specialization of lazy permutation for when indices are the same.
 *
 *  This is specialization handles the case when the indices are already
 *  aligned.
 *
 */
template<typename NewIdx, typename Tensor_t>
struct Permutation<NewIdx,NewIdx,Tensor_t>:
        public OperationBase<Permutation<NewIdx,NewIdx,Tensor_t>> {
    ///The tensor to permute
    Tensor_t t_;

    /** \brief Makes a new Permutation operation by copying \p t
     *
     *  \param[in] The tensor to permute.
     *  \throws ??? Throws if Tensor_t's constructor throws.  Strong throw
     *              guarantee.
     *
     */
    Permutation(const Tensor_t& t):t_(t){}

    ///The rank of the tensor
    constexpr static size_t rank=Tensor_t::rank;

    ///The type of the elements in the tensor
    using scalar_type=typename Tensor_t::scalar_type;

    ///The indices after permutation
    using indices=NewIdx;

    /** \brief Actually evaluates the permutation.
     *
     *  This is a null op.
     */
    template<TensorTypes TT>
    auto eval()const->decltype(t_.template eval<TT>())
    {
        return t_.template eval<TT>();
    }

};//End Null Permuation specialization template


}}//End namespaces
