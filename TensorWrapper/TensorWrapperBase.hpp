#pragma once
#include "TensorWrapper/Contraction.hpp"
#include "TensorWrapper/Shape.hpp"
#include "TensorWrapper/MemoryBlock.hpp"
#include "TensorWrapper/TensorPtr.hpp"
#include "TensorWrapper/TensorImpl/TensorTypes.hpp"
#include "TensorWrapper/OperationImpls.hpp"
#include "TensorWrapper/Downcast.hpp"

namespace TWrapper {

template<size_t R,typename T>
class TensorWrapperBase{
protected:
    ///A type erased tensor
    detail_::TensorPtr tensor_;

    ///That tensor's type
    detail_::TensorTypes ttype_;

    ///Constructor used by derived classes with a tensor
    TensorWrapperBase(detail_::TensorPtr& tensor,
                      detail_::TensorTypes ttype):
        tensor_(tensor),
        ttype_(ttype)
    {}

    ///Constructor used by derived class's default ctor
    TensorWrapperBase(detail_::TensorTypes ttype):
        ttype_(ttype)
    {}

public:
    using index_t=std::array<size_t,R>;

    virtual ~TensorWrapperBase(){}

    ///@{
    ///Returns the rank of the wrapped tensor
    constexpr size_t rank()const{return R;}

    ///Returns the shape of the wrapped tensor
    Shape<R> shape()const
    {
        return detail_::unary_downcast(tensor_,ttype_,detail_::DimsOp<R>());
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
        return detail_::unary_downcast(tensor_,ttype_,
                   detail_::SliceOp<R,T>(),start,end);
    }
    MemoryBlock<R,const T> get_memory()const
    {
        return detail_::unary_downcast(tensor_,ttype_,
                   detail_::GetMemoryOp<R,const T>());
    }

    MemoryBlock<R,T> get_memory()
    {
        return detail_::unary_downcast(tensor_,ttype_,
                                       detail_::GetMemoryOp<R,T>());
    }

    void set_slice(const MemoryBlock<R,T>& other)
    {
        detail_::unary_downcast(tensor_,ttype_,
                                detail_::SetMemoryOp<R,T>(),other);
    }

    template<typename RHS_t>
    auto operator+(const RHS_t& rhs)const
    {
        return detail_::unary_downcast(tensor_,ttype_,
                                       detail_::AddOp<R,T>(),rhs);
    }


//    ///API for contraction
//    template<size_t N> constexpr
//    IndexedTensor<Rank,wrapped_t> operator()(const char(&idx)[N])const
//    {
//        return IndexedTensor<Rank,wrapped_t>(tensor_,idx);
//    }
};


}//End namespace
