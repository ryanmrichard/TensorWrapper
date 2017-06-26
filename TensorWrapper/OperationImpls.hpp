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
 *  \tparam R The rank of the resulting tensor
 *  \tparam T The type of the scalar elements inside the tensor.
 *
 *
 *  \throws std::bad_cast if the tensor is not convertible to the requested
 *           type.
 */
template<size_t R, typename T>
struct DeRef{
    template<TensorTypes T1,typename Tensor_t>
    auto& eval(Tensor_t&& tensor)
    {
        return tensor.template cast<T1>();
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
    auto eval(Tensor_t&& lhs,T scale)
    {
        TensorWrapperImpl<R,T,T1> impl;
        return impl.scale(std::forward<Tensor_t>(lhs),scale);
    }
};

//template<size_t R,typename T>
//struct GetMemoryOp{
//    template<TensorTypes T1,typename TensorPtr_t>
//    static MemoryBlock<R,T> eval(TensorPtr_t&& tensor)
//    {
//        TensorWrapperImpl<R,T,T1> impl;
//        return impl.get_memory(std::forward<TensorPtr_t>(tensor));
//    }
//};

//template<size_t R,typename T>
//struct SliceOp{
//    template <typename Impl,typename Tensor_t>
//    static TensorWrapperBase<R,T> eval(const Impl& impl,
//                                       Tensor_t&& ptensor,
//                                       const std::array<size_t,R>& start,
//                                       const std::array<size_t,R>& end)
//    {
//        return impl.slice(std::forward<Tensor_t>(ptensor),start,end);
//    }
//};

//template<size_t R,typename T>
//struct SetMemoryOp{
//    template <typename Impl,typename TensorPtr_t>
//    void eval(const Impl& impl,
//              TensorPtr_t&& ptensor,
//              const MemoryBlock<R,T>& memory)
//    {
//        impl.set_memory(std::forward<TensorPtr_t>(ptensor),memory);
//    }
//};

//template<size_t R,typename T>
//struct EqualOp{
//    template <typename Impl,
//              typename LHS_t,
//              typename RHS_t>
//    auto eval(const Impl& impl,
//              LHS_t&& lhs,
//              RHS_t& rhs)
//    {
//        return impl.are_equal(std::forward<LHS_t>(lhs),
//                              std::forward<RHS_t>(rhs));
//    }
//};





//template<size_t R,typename T>
//struct SubtractOp{
//    template<typename Impl,typename LHS_t,typename RHS_t>
//    auto eval(const Impl& impl,
//              LHS_t& lhs,
//              RHS_t& rhs)
//    {
//        return impl.add(std::forward<LHS_t>(lhs),
//              ScaleOp<R,T>().eval(impl,std::forward<RHS_t>(rhs),-1.0));
//    }
//};

}}//End namespaces
