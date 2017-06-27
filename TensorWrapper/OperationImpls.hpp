#pragma once
#include "TensorWrapper/TensorImpl/TensorTypes.hpp"
#include "TensorWrapper/TensorImpl/TensorWrapperImpl.hpp"
#include "TensorWrapper/Shape.hpp"
#include "TensorWrapper/TensorPtr.hpp"

namespace TWrapper{
namespace detail_{

/** \brief An operation that dereferences a TensorPtr instance into the base
 *  form of a given backend.
 *
 *  There are three possibilities when dereferencing the TensorPtr:
 *  1. The tensor pointer is already of that form and this is a simple pointer
 *     dereference.
 *  2. We have a special way of making the input tensor into the output tensor
 *     and it's about the same cost as a dereference.
 *  3. We have to manually copy it into the new one.
 *
 *  All of the scenarios are handled by this class combined with the TensorPtr
 *  class.  In the event we run into 2 or 3 we need to store the temporary
 *  result and return a reference to it to ensure we keep the return type the
 *  same.
 *
 *  \tparam R The rank of the resulting tensor
 *  \tparam T The type of the scalar elements inside the tensor.
 *
 *
 *  \throws std::bad_cast if the tensor is not castable to the requested
 *           type and std::logic_error if it is not convertable.
 *
 */
template<size_t R, typename T>
struct DeRef{
    TensorPtr<R,T> buffer_;

    template<TensorTypes T1,typename Tensor_t>
    auto eval(Tensor_t&& tensor)->decltype(tensor.template cast<T1>())
    {
        //Catches the case where it's the same as requested type
        if(tensor.type()==T1)
            return tensor.template cast<T1>();
        //Need to convert
        buffer_=std::move(tensor.template convert<T1>());
        return buffer_.template cast<T1>();
    }
};

template<size_t R, typename T>
struct EraseType{
    template<TensorTypes T1, typename Tensor_t>
    TensorPtr<R,T> eval(Tensor_t&& tensor)const
    {
        return TensorPtr<R,T>(T1,std::forward<Tensor_t>(tensor));
    }
};


/** \brief Returns the shape of a tensor
 *
 *
 * \tparam R The rank of the tensor
 * \tparam T The scalar's type.
 */
template<size_t R,typename T>
struct DimsOp{
    template<TensorTypes T1, typename Tensor_t>
    Shape<R> eval(Tensor_t&& tensor){
        TensorWrapperImpl<R,T,T1> impl;
        return impl.dims(std::forward<Tensor_t>(tensor));
    }
};

template<size_t R,typename T>
struct AddOp{
    template <TensorTypes T1,typename LHS_t,typename RHS_t>
    auto eval(LHS_t&& lhs,RHS_t&& rhs)
        ->decltype(TensorWrapperImpl<R,T,T1>().add(std::forward<LHS_t>(lhs),
                                                   std::forward<RHS_t>(rhs)))
    {
        TensorWrapperImpl<R,T,T1> impl;
        return impl.add(std::forward<LHS_t>(lhs),
                        std::forward<RHS_t>(rhs));
    }
};

template<size_t R,typename T>
struct EqualOp{
    template <TensorTypes T1,typename LHS_t,typename RHS_t>
    bool eval(LHS_t&& lhs,RHS_t&& rhs)
    {
        TensorWrapperImpl<R,T,T1> impl;
        return impl.are_equal(std::forward<LHS_t>(lhs),
                              std::forward<RHS_t>(rhs));
    }
};

template<size_t R,typename T>
struct ScaleOp{
    template <TensorTypes T1, typename Tensor_t>
    auto eval(Tensor_t&& lhs,T rhs)->
        decltype(TensorWrapperImpl<R,T,T1>().scale(std::forward<Tensor_t>(lhs),
                                                   rhs))
    {
        TensorWrapperImpl<R,T,T1> impl;
        return impl.scale(std::forward<Tensor_t>(lhs),rhs);
    }
};

template<size_t R, typename T>
struct SubtractionOp{
    template<TensorTypes T1, typename LHS_t, typename RHS_t>
    auto eval(LHS_t&& lhs, RHS_t&& rhs)->
    decltype(AddOp<R,T>().template eval<T1>(std::forward<LHS_t>(lhs),
                 ScaleOp<R,T>().template eval<T1>(std::forward<RHS_t>(rhs),-1.0)
             ))
    {
        return AddOp<R,T>().template eval<T1>(
            std::forward<LHS_t>(lhs),
            ScaleOp<R,T>().template eval<T1>(std::forward<RHS_t>(rhs),-1.0)
        );
    }

};

template<size_t R, typename T>
struct GetMemoryOp{
    template <TensorTypes T1, typename Tensor_t>
    MemoryBlock<R,T> eval(Tensor_t&& tensor)const
    {
        TensorWrapperImpl<R,T,T1> impl;
        return impl.get_memory(std::forward<Tensor_t>(tensor));

    }
};

/** \brief Sets the memory of a tensor given a MemoryBlock instance
 *
 * \note At the moment, this function only returns to avoid having to specialize
 * eval_op for void functions.
 *
 *  \returns The tensor after setting its memory.
 *
 */
template<size_t R, typename T>
struct SetMemoryOp{
    template <TensorTypes T1, typename Tensor_t>
    TensorPtr<R,T> eval(Tensor_t&& tensor,const MemoryBlock<R,T>& mem)const
    {
        TensorWrapperImpl<R,T,T1> impl;
        impl.set_memory(std::forward<Tensor_t>(tensor),mem);
        return TensorPtr<R,T>(T1,std::forward<Tensor_t>(tensor));

    }
};

template<size_t R, typename T>
struct SliceOp{
    template<TensorTypes T1, typename Tensor_t>
    auto eval(Tensor_t&& tensor,
              const std::array<size_t,R>& start,
              const std::array<size_t,R>& end)->
    decltype(TensorWrapperImpl<R,T,T1>().slice(std::forward<Tensor_t>(tensor),
                                               start,end))
    {
        return TensorWrapperImpl<R,T,T1>().slice(
                    std::forward<Tensor_t>(tensor),
                    start,end);
    }
};

}}//End namespaces
