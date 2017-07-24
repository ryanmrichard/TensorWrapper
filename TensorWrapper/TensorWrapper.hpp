#pragma once
#include <array>
#include "TensorWrapper/DisableUselessWarnings.hpp"
#include "TensorWrapper/TensorWrapperBase.hpp"
#include "TensorWrapper/TensorImpls.hpp"
#include "TensorWrapper/Traits.hpp"

/** \file This is the main include file for the TensorWrapper library
 */
namespace TWrapper {

template<size_t Rank, typename T, detail_::TensorTypes T1>
class TensorWrapper: public TensorWrapperBase<Rank,T>{
private:
    using Impl_t=detail_::TensorWrapperImpl<Rank,T,T1>;
    Impl_t impl_;
    using tensor_ptr=detail_::TensorPtr<Rank,T>;

public:
    ///The type of an instance of this class
    using my_type=TensorWrapper<Rank,T,T1>;

    ///The type of the base class
    using base_type=TensorWrapperBase<Rank,T>;

    ///The type wrapped by this class
    using wrapped_t=typename detail_::TensorWrapperImpl<Rank,T,T1>::type;

    ///The type of an index
    using index_t=std::array<size_t,Rank>;

    ///\copydoc TensorWrapperBase()
    TensorWrapper():
        base_type(T1)
    {}
    template<typename RHS_t,
             typename detail_::EnableIfAnOperation<RHS_t>::type=0>
    TensorWrapper(RHS_t&& op):
        base_type(T1)
    {
        auto result=op.template eval<T1>();
        typename Impl_t::type temp_tensor=result.eval();
        base_type::tensor_=std::move(tensor_ptr(T1,std::move(temp_tensor)));
    }

    /** \brief The constructor used for wrapping the native classes of a tensor
     *         backend.
     *
     *  This is the main entry point for wrapping an already existing tensor.
     *  If the backend supports lazy evaluation, odds are this will trigger
     *  the evaluation (whether it triggers or not depends on whether said
     *  evaluation triggers when the temporary is assigned to an instance of
     *  the primary tensor type).
     *
     *  \note Whether this is a deep or shallow copy depends on the backend.
     *
     *  \param[in] other A universal reference to an instance from our backend
     *                   that is implicitly convertible to our wrapped type.
     *
     *  \throws std::bad_alloc if memory allocation fails.  May also throw if
     *          backend throws for move or copy (which ever applies).
     *          Same guarantee as the move/copy constructor of \p other.
     *
     * \tparam RHS_t The type of the tensor we are to wrap.
     */
    template<typename RHS_t,
             typename detail_::EnableIfNotAnOperation<RHS_t>::type=0>
    TensorWrapper(RHS_t&& other):
        base_type(std::move(tensor_ptr(T1,std::forward<RHS_t>(other))),T1)
    {}

    template<typename...Args>
    TensorWrapper& operator=(const detail_::Operation<Rank,Args...>& op)
    {
        auto result=op.template eval<T1>();
        base_type::tensor_=std::move(detail_::TensorPtr<Rank,T>(T1,result));
        return *this;
    }


    /** \brief Constructor for allocating the memory of the tensor
     *
     *  After calling this constructor the memory is allocated, but in an
     *  undefined initialization state.
     */
    TensorWrapper(const index_t& dims):
        base_type(T1)
    {
        base_type::tensor_=
                std::move(detail_::TensorPtr<Rank,T>(T1,
                            std::move(impl_.allocate(dims))
                ));
    }

    TensorWrapper(my_type&&)=default;
    my_type& operator=(const my_type&)=default;
    my_type& operator=(my_type&&)=default;

    Shape<Rank> shape()const override
    {
        auto op=detail_::make_op<Rank,T,detail_::DimsOp<Rank,T>>
                (this->de_ref());
        return op.template eval<T1>();
    }

    T operator()(const index_t& idx)const override
    {
        //Grab a 1 x 1 x 1.... slice of the tensor and return its only element
        index_t p1,zeros{};
        for(size_t i=0;i<Rank;++i)p1[i]=idx[i]+1;
        auto op=detail_::make_op<Rank,T,detail_::EraseType<Rank,T>>(
                    this->slice(idx,p1)
                    );
        my_type temp(std::move(op.template eval<T1>()));
        return temp.get_memory()(zeros);
    }

    template<typename...Args>
    T operator()(size_t elem1,Args...args)const
    {
        return (*this)(elem1,args...);
    }

    MemoryBlock<Rank,T> get_memory() override
    {
        auto op=detail_::make_op<Rank,T,detail_::GetMemoryOp<Rank,T>>
                (this->de_ref());
        return op.template eval<T1>();
    }

    void set_memory(const MemoryBlock<Rank,T>& other)override
    {
        auto op=detail_::make_op<Rank,T,detail_::SetMemoryOp<Rank,T>>(
                    this->de_ref(),other);
        op.template eval<T1>();
    }


    template<typename RHS_t>
    bool operator==(RHS_t&& rhs)const
    {
        auto op = (this->de_ref()==std::forward<RHS_t>(rhs));
        return op.template eval<T1>();
    }

    template<typename RHS_t>
    bool operator!=(RHS_t&& rhs)const
    {
        return !((*this)==std::forward<RHS_t>(rhs));
    }

    using base_type::operator+;
    using base_type::operator-;
    using base_type::operator*;
 };

//template<typename T, detail_::TensorTypes T1>
//std::pair<TensorWrapper<1,T,T1>,TensorWrapper<2,T,T1>>
//self_adjoint_eigen_solver(const TensorWrapper<2,T,T1>& matrix){
//    detail_::TensorWrapperImpl<2,T,T1> impl;
//    auto rv=impl.self_adjoint_eigen_solver(matrix.tensor());
//    return {TensorWrapper<1,T,T1>(rv.first),
//            TensorWrapper<2,T,T1>(rv.second)};
//}

template<typename T>
using EigenVector=TensorWrapper<1,T,detail_::TensorTypes::EigenMatrix>;
template<typename T>
using EigenMatrix=TensorWrapper<2,T,detail_::TensorTypes::EigenMatrix>;
template<size_t rank,typename T>
using EigenTensor=TensorWrapper<rank,T,detail_::TensorTypes::EigenTensor>;

}//End namespace TWrapper

template<typename LHS_t,typename T, TWrapper::detail_::TensorTypes T1,size_t R>
bool operator==(LHS_t&& lhs,const TWrapper::TensorWrapper<R,T,T1>& t)
{
    return t==std::forward<LHS_t>(lhs);
}

template<typename LHS_t,typename T, size_t R, TWrapper::detail_::TensorTypes T1>
bool operator!=(LHS_t&& lhs,const TWrapper::TensorWrapper<R,T,T1>& t)
{
    return t!=std::forward<LHS_t>(lhs);
}

#include "TensorWrapper/EnableUselessWarnings.hpp"
