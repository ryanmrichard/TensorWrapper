#pragma once

namespace TWrapper{
template<size_t R,typename T>
class TensorWrapperBase;

namespace detail_{

template<size_t R, typename T>
struct DeRef{
    template<TensorTypes T1,typename Tensor_t>
    auto& eval(Tensor_t&& tensor)
    {
        return tensor.template cast<T1>();
    }
};

///Returns the shape of a tensor
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
        using return_t=decltype(impl.add(
            std::forward<LHS_t>(lhs),std::forward<RHS_t>(rhs)));
        return std::forward<return_t>(
                    impl.add(std::forward<LHS_t>(lhs),
                             std::forward<RHS_t>(rhs)));
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
//struct ScaleOp{
//    template <typename Impl,typename tensor_t>
//    auto eval(const Impl& impl,
//              tensor_t&& lhs,
//              T scale)
//    {
//        return impl.scale(std::forward<tensor_t>(lhs),scale);
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
