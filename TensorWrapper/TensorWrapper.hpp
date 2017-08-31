#pragma once
#include <array>
#include "TensorWrapper/DisableUselessWarnings.hpp"
#include "TensorWrapper/TensorWrapperBase.hpp"
#include "TensorWrapper/RunTime.hpp"

/** \file This is the main include file for the TensorWrapper library it defines
 *  our public API.
 *
 * For a project utilizing TensorWrapper this should be the only include file
 * you need.
 */

///The main namespace for the TensorWrapper library
namespace TWrapper {

/** \brief The namesake class of the TensorWrapper library.
 *
 *  If you are making tensors you will be making instances of this class.  If
 *  your routine is taking tensors as input you will be working with the
 *  base class.
 *
 *  \tparam R The rank of the tensor
 *  \tparam T The type of the elements in the tensor
 *  \tparam TT The enum of the backend to use
 */
template<size_t R, typename T, detail_::TensorTypes TT>
class TensorWrapper: public TensorWrapperBase<R,T>{
private:
    ///The type of the API to the wrapper around the backend
    using Impl_t=detail_::TensorWrapperImpl<R,T,TT>;

    ///The type of the type-erased pointer stored in the base
    using pTensor=typename TensorWrapperBase<R,T>::pTensor;

    ///An instance of the API to the backend
    Impl_t impl_;    


    /** \brief Returns the pointer stored in the base.
     *
     *  This is basically a convenience function to account for the fact that
     *  C++ is annoying when it comes to templated class hierarchies and we
     *  would have to scope the base class member every time.
     *
     *  \returns The pointer stored in the base class
     *  \throws None No throw guarantee
     */
    pTensor& ptr_()noexcept
    {
        return this->tensor_;
    }


    ///\copydoc ptr_()
    const pTensor& ptr_()const noexcept
    {
        return this->tensor_;
    }

    ///Allows us to grab the tensor pointer from other TensorWrapper instances
    template<size_t R2, typename T2, detail_::TensorTypes TT2>
    friend class TensorWrapper;

public:
    ///The type of an instance of this class
    using my_type=TensorWrapper<R,T,TT>;

    ///The type of the base class
    using base_type=TensorWrapperBase<R,T>;

    ///The type of the backend's tensor class
    using wrapped_t=typename Impl_t::type;

    ///The type of an R element std::array of size_t s
    using index_t=typename base_type::index_t;

    ///\copydoc TensorWrapperBase()
    TensorWrapper()=default;

    /** \brief Returns a polymorphic deep copy of this instance allocated on the
     *  heap.
     *
     *  Code dealing with TensorWrapper instances will usually just use the
     *  copy/move constructors/assignment functions.  However, code dealing
     *  with the common base class to all TensorWrapper's, TensorWrapperBase,
     *  will invoke this function to perform a polymorphic deep copy and avoid
     *  slicing.
     *
     *  \returns A new deep copy of the current instance allocated on the heap.
     *  \throws std::bad_alloc if memory allocation fails.
     *
     */
    std::unique_ptr<base_type> clone()const override
    {
        return std::make_unique<my_type>(*this);
    }

    /** \brief The "copy" constructor for TensorWrapper instances that do not
     *   share the same backend.
     *
     *   This function ultimately will invoke the copy constructor of the
     *   backend.  Whether this is a deep copy depends on whether the backend's
     *   copy constructor deep copies.
     *
     *   \param[in] other The TensorWrapper instance to copy
     *
     *   \tparam TT2 The enum of the backend of \p other
     *
     */
    template<detail_::TensorTypes TT2,
             typename detail_::EnableIfNotSameBackend<TT,TT2>::type=0>
    TensorWrapper(const TensorWrapper<R,T,TT2>& other)
        :base_type(std::move(other.ptr_().template convert<TT>()))
    {}

    /** \brief The "copy" assignment operator for TensorWrapper instances that
     *  do not share the same backend.
     *
     *   This function ultimately will invoke the copy constructor of the
     *   backend.  Whether this is a deep copy depends on whether the backend's
     *   copy constructor deep copies.
     *
     *   \param[in] other The TensorWrapper instance to copy
     *   \returns The current instance as a copy of \p other.
     *   \tparam TT2 The enum of the backend of \p other
     *
     */
    template<detail_::TensorTypes TT2,
             typename detail_::EnableIfNotSameBackend<TT,TT2>::type=0>
    my_type& operator=(const TensorWrapper<R,T,TT2>& other)
    {
        base_type::tensor_=std::move(other.ptr_().template convert<TT>());
        return *this;
    }

    /** \brief The copy constructor for TensorWrapper instances that share the
     *   same backend.
     *
     *   This function ultimately will invoke the copy constructor of the
     *   backend.  Whether this is a deep copy depends on whether the backend's
     *   copy constructor deep copies.
     *
     *   \param[in] other The TensorWrapper instance to copy
     *
     */
    TensorWrapper(const my_type& /*other*/)=default;

    /** \brief The copy assignment operator for TensorWrapper instances that
     *  share the same backend.
     *
     *   This function ultimately will invoke the copy constructor of the
     *   backend.  Whether this is a deep copy depends on whether the backend's
     *   copy constructor deep copies.
     *
     *   \param[in] other The TensorWrapper instance to copy
     *   \returns The current instance containing a copy of \p other
     *
     */
    my_type& operator=(const my_type& /*other*/)=default;

    TensorWrapper(my_type&&)=default;
    my_type& operator=(my_type&&)=default;

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
             typename=typename detail_::EnableIfNotOpOrTW<RHS_t>::type>
    TensorWrapper(RHS_t&& other):
        base_type(std::move(pTensor(TT,std::forward<RHS_t>(other))))
    {}

    /** \brief Constructor for allocating the memory of the tensor
     *
     *  After calling this constructor the memory is allocated, but in an
     *  undefined initialization state.
     */
    TensorWrapper(index_t dims)
    {
        ptr_()=std::move(pTensor(TT,std::move(impl_.allocate(dims))));
    }

    template<typename RHS_t>
    TensorWrapper(const detail_::OperationBase<RHS_t>& op)
    {
        const RHS_t& up_op=static_cast<const RHS_t&>(op);
        wrapped_t temp_tensor=impl_.template eval<typename RHS_t::indices>(
                    up_op.template eval<TT>(),up_op.dimensions());
        ptr_()=std::move(pTensor(TT,std::move(temp_tensor)));
    }


    template<typename RHS_t>
    TensorWrapper& operator=(const detail_::OperationBase<RHS_t>& op)
    {
        TensorWrapper temp(op);
        *this=std::move(temp);
        return *this;
    }


    wrapped_t& data()
    {
        return ptr_().template cast<TT>();
    }

    const wrapped_t& data()const
    {
        return ptr_().template cast<TT>();
    }

    Shape<R> shape()const override
    {
        return impl_.dims(data());
    }

    T operator()(const index_t& idx)const override
    {
        //Grab a 1 x 1 x 1.... slice of the tensor and return its only element
        index_t p1;
        for(size_t i=0;i<R;++i)p1[i]=idx[i]+1;
        auto slice_of_t=impl_.slice(data(),idx,p1);
        my_type temp(std::move(slice_of_t));
        return temp.get_memory().block(0)[0];
    }

    wrapped_t slice(const index_t& start,const index_t& end)const
    {
        return impl_.slice(data(),start,end);
    }

    template<typename...Args>
    T operator()(size_t elem1,Args...args)const
    {
        return base_type::operator()(elem1,args...);
    }

    ///\copydoc TensorWrapperBase<R,T>::get_memory()
    MemoryBlock<R,T> get_memory() override
    {
        return impl_.get_memory(data());
    }

    ///\copydoc TensorWrapperBase<R,T>::set_memory()
    void set_memory(const MemoryBlock<R,T>& other)override
    {
        impl_.set_memory(data(),other);
    }

    template<typename RHS_t>
    bool operator==(detail_::OperationBase<RHS_t>& rhs)const
    {
        const RHS_t& up_cast=static_cast<RHS_t&>(rhs);
        return impl_.are_equal(data(),up_cast.template eval<TT>());
    }

    template<typename RHS_t,
             typename =typename detail_::EnableIfNotAnOperation<RHS_t>::type>
    bool operator==(RHS_t&& rhs)const
    {
        detail_::Convert<RHS_t> wrapped(rhs);
        return operator==(wrapped);
    }

    template<typename RHS_t>
    bool operator!=(RHS_t&& rhs)const
    {
        return !((*this)==std::forward<RHS_t>(rhs));
    }


    template<char...Chars,typename...Args>
    auto operator()(const detail_::C_String<Chars...>& str,Args...args)const
    {
        return base_type::operator()(str,args...);
    }
 };

template<typename T, detail_::TensorTypes TT>
std::pair<TensorWrapper<1,T,TT>,TensorWrapper<2,T,TT>>
self_adjoint_eigen_solver(const TensorWrapper<2,T,TT>& tensor_in)
{
    auto rv=detail_::TensorWrapperImpl<2,T,TT>().
               self_adjoint_eigen_solver(tensor_in.data());
    return std::make_pair(TensorWrapper<1,T,TT>(std::move(rv.first)),
                          TensorWrapper<2,T,TT>(std::move(rv.second)));
}



namespace detail_ {

template<size_t R, typename T, TensorTypes TT>
struct Convert<TensorWrapper<R,T,TT>> : public OperationBase<
        Convert<TensorWrapper<R,T,TT>>>
{
    using scalar_type=T;
    constexpr static size_t rank=R;
    using indices=IdxNotSet;

    const TensorWrapper<R,T,TT>& data_;
    Convert(const TensorWrapper<R,T,TT>& data):
        data_(data)
    {}

    template<TensorTypes TT2>
    auto eval()const->decltype(data_.data())
    {
        static_assert(TT2==TT,"Must convert tensor before operations");
        return data_.data();
    }
};

}//End namespace detail

template<typename T>
using EigenScalar=TensorWrapper<0,T,detail_::TensorTypes::EigenMatrix>;
template<typename T>
using EigenVector=TensorWrapper<1,T,detail_::TensorTypes::EigenMatrix>;
template<typename T>
using EigenMatrix=TensorWrapper<2,T,detail_::TensorTypes::EigenMatrix>;
template<size_t rank,typename T>
using EigenTensor=TensorWrapper<rank,T,detail_::TensorTypes::EigenTensor>;
template<size_t rank,typename T>
using GlobalArray=TensorWrapper<rank,T,detail_::TensorTypes::GlobalArrays>;

template class TensorWrapper<1,double,detail_::TensorTypes::EigenMatrix>;
template class TensorWrapper<2,double,detail_::TensorTypes::EigenMatrix>;

template class TensorWrapper<1,double,detail_::TensorTypes::EigenTensor>;
template class TensorWrapper<2,double,detail_::TensorTypes::EigenTensor>;
template class TensorWrapper<3,double,detail_::TensorTypes::EigenTensor>;
template class TensorWrapper<4,double,detail_::TensorTypes::EigenTensor>;

//template class TensorWrapper<1,double,detail_::TensorTypes::GlobalArrays>;
//template class TensorWrapper<2,double,detail_::TensorTypes::GlobalArrays>;


}//End namespace TWrapper

///Overload equality operator for our types on right
template<size_t R, typename T, TWrapper::detail_::TensorTypes TT,typename LHS,
         typename TWrapper::detail_::EnableIfNotATWrapper<LHS>::type=0>
bool operator==(LHS&& lhs,const TWrapper::TensorWrapper<R,T,TT>& rhs)
{
    return rhs==std::forward<LHS>(lhs);
}

///Overload not equal operator for our types on rigth
template<size_t R, typename T, TWrapper::detail_::TensorTypes TT, typename LHS>
bool operator!=(LHS&& lhs,const TWrapper::TensorWrapper<R,T,TT>& rhs)
{
    return rhs!=std::forward<LHS>(lhs);
}

#include "TensorWrapper/EnableUselessWarnings.hpp"
