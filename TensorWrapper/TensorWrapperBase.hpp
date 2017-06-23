#pragma once
#include "TensorWrapper/Contraction.hpp"
#include "TensorWrapper/Shape.hpp"
#include "TensorWrapper/MemoryBlock.hpp"
#include "TensorWrapper/TensorPtr.hpp"
#include "TensorWrapper/TensorImpl/TensorTypes.hpp"
#include "TensorWrapper/Operation.hpp"
#include "TensorWrapper/OperationImpls.hpp"

namespace TWrapper {
/** \brief  The class that actually does tensor-y stuff.
 *
 *  \tparam R The rank of the tensor
 *  \tparam T The type of the scalars in the tensor.
 */
template<size_t R,typename T>
class TensorWrapperBase{
protected:
    ///A type erased tensor
    detail_::TensorPtr<R,T> tensor_;

    ///That tensor's type
    detail_::TensorTypes ttype_;

    /** \brief Constructor used by derived classes after wrapping a tensor.
     *
     *
     *   \param[in] tensor The wrapped tensor we now own.  Memory allocation
     *                     etc. occurred in instantiating \p tensor.
     *   \param[in] ttype  The enumeration corresponding to the backend to
     *                     use for this tensor.
     *
     *   \throws No throw guarantee.
     */
    TensorWrapperBase(detail_::TensorPtr<R,T>&& tensor,
                      detail_::TensorTypes ttype)noexcept:
        tensor_(std::move(tensor)),
        ttype_(ttype)
    {}

    /** \brief The constructor for when the tensor isn't ready yet.
     *
     *  Unlike the other protected constructor this constructor assumes
     *  the derived class will be building
     *
     */
    TensorWrapperBase(detail_::TensorTypes ttype):
        ttype_(ttype)
    {}

public:

    detail_::TensorPtr<R,T>& tensor(){return tensor_;}
    const detail_::TensorPtr<R,T>& tensor()const{return tensor_;}
    detail_::TensorTypes type()const{return ttype_;}

    ///The type of a "rank"-dimensional vector of indices
    using index_t=std::array<size_t,R>;

    /** \brief Constructs a null tensor instance
     *
     *  The resulting instance is essentially a placeholder and can only be made
     *  usable by assigning or moving a legit instance into it.
     *
     *  \throws Never throws.
     */
    TensorWrapperBase()noexcept=default;

    /** \brief Constructs a new instance via a deep copy of \p other
     *
     *  \note This constructor relies on the derived classes having no state
     *        because it slices (in the C++ sense of the word) the class.
     *
     *  \param[in] other The tensor to deep copy.
     *
     *  \throws std::bad_alloc if there is insufficient memory for the copy.
     *          Strong throw guarantee.
     *
     */
    TensorWrapperBase(const TensorWrapperBase&)=default;

    /** \brief Assigns a deep copy of \p other to the current instance.
     *
     *  \param[in] other The tensor to deep copy.
     *  \return The current instance containing a deep copy of other.
     *  \throws std::bad_alloc if the allocation fails.  Strong throw guarantee.
     */
    TensorWrapperBase& operator=(const TensorWrapperBase&)=default;

    /** \brief Takes ownership of another TensorWrapper.
     *
     *  \param[in] other The instance we are taking over.
     *
     *  \throw No throw guarantee.
     */
    TensorWrapperBase(TensorWrapperBase&&)noexcept=default;

    /** \brief Takes ownership of another TensorWrapper freeing up current
     *         resources.
     *
     *  \param[in] other The tensor to take ownership of.
     *  \returns The current instance, now with the contents of other.
     *  \throws No throw guarantee.
     */
    TensorWrapperBase& operator=(TensorWrapperBase&&)noexcept=default;


    virtual ~TensorWrapperBase(){}

    ///@{
    ///Returns the rank of the wrapped tensor
    constexpr size_t rank()const{return R;}

    ///Returns the shape of the wrapped tensor
    Shape<R> shape()const
    {
        auto op= detail_::make_op<detail_::DimsOp<R,T>>(
                    detail_::make_op<detail_::DeRef<R,T>>(tensor_));

        return detail_::eval_op(ttype_,op);
    }
    ///@}


    /** \brief Returns an element of the tensor. This is a convenience API and
     *         is slow.
     *
     *  For production level code use get_memory or slice functions.
     *
     *  \return The requested element.
     */
    T operator()(const index_t& idx)const
    {
        index_t p1,zeros{};
        for(size_t i=0;i<R;++i)p1[i]=idx[i]+1;
        return slice(idx,p1).get_memory()(zeros);
    }

    /** \brief Syntactic sugar for accessing an element of the tensor using a
     *         comma separated list of indices rather than an object of type
     *         index_t.
     *
     *  This is a simple wrapper around the other operator() member function.
     *  The signature may look scary, but usage is simple:
     *
     *  \code{.cpp}
     *  TensorWrapper<3,T> tensor;//Assume this initialized already
     *  tensor(0,3,6);//<-Usage.  Returns the element at index {0,3,6}
     *  \endcode
     *
     *  \param[in] elem1 The index along the first dimension to retrieve
     *  \param[in] args The remaining "rank-1" indices to retrieve
     *
     *  \note The first element is split off to avoid this function's signature
     *  reducing to that of the other operator().
     *
     *  \return The value of the tensor at the input index
     */
    template<typename...Args>
    T operator()(size_t elem1,Args...args)const
    {
        static_assert(sizeof...(args)==R-1, "Number of indices != rank");
        return (*this)(index_t{elem1,args...});
    }

    TensorWrapperBase<R,T> slice(const index_t& start,
                                 const index_t& end)const
    {
//        return detail_::downcast(tensor_,ttype_,
//                   detail_::SliceOp<R,T>(),start,end);
    }
    MemoryBlock<R,const T> get_memory()const
    {
//        return detail_::downcast(tensor_,ttype_,
//                   detail_::GetMemoryOp<R,const T>());
    }

    MemoryBlock<R,T> get_memory()
    {
//        return detail_::downcast(tensor_,ttype_,
//                                       detail_::GetMemoryOp<R,T>());
    }

    void set_slice(const MemoryBlock<R,T>& other)
    {
//        detail_::downcast(tensor_,ttype_,
//                                detail_::SetMemoryOp<R,T>(),other);
    }

//    ///API for contraction
//    template<size_t N> constexpr
//    IndexedTensor<Rank,wrapped_t> operator()(const char(&idx)[N])const
//    {
//        return IndexedTensor<Rank,wrapped_t>(tensor_,idx);
//    }
};


}//End namespace
