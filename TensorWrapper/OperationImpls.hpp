#pragma once

namespace TWrapper{
template<size_t R,typename T>
class TensorWrapperBase;

namespace detail_{

///Imparts an eval api to arbitrary data
template<typename...Args>
struct Data{

};

///Casts a tensor back to its native format
template<size_t R, typename T>
struct Caster{
    template<typename Impl>
    static auto eval(const Impl&,TensorPtr& ptensor){
        return ptensor.cast<Impl::type>();
    }
};

///Returns the shape of a tensor
template<size_t R>
struct DimsOp{
    template<typename Impl, typename Tensor_t>
    static Shape<R> eval(const Impl& impl,
                         const Tensor_t& tensor){
        return impl.dims(tensor);
    }
};

template<size_t R,typename T>
struct GetMemoryOp{
    template<typename Impl,typename TensorPtr_t>
    static MemoryBlock<R,T> eval(const Impl& impl,
                                 TensorPtr_t& ptensor)
    {
        return impl.get_memory(Caster<R,T>::eval(impl,ptensor));
    }
};

template<size_t R,typename T>
struct SliceOp{
    template <typename Impl>
    static TensorWrapperBase<R,T> eval(const Impl& impl,
                                       const TensorPtr& ptensor,
                                       const std::array<size_t,R>& start,
                                       const std::array<size_t,R>& end)
    {
        return impl.slice(Caster<R,T>::eval(impl,ptensor),start,end);
    }
};

template<size_t R,typename T>
struct SetMemoryOp{
    template <typename Impl,typename TensorPtr_t>
    void eval(const Impl& impl,
              TensorPtr_t& ptensor,
              const MemoryBlock<R,T>& memory)
    {
        impl.set_memory(Caster<R,T>::eval(impl,ptensor),memory);
    }
};

template<size_t R,typename T>
struct EqualOp{
    template <typename Impl,typename RHS_t>
    auto eval(const Impl& impl,
              const TensorPtr& lhs,
              const TensorPtr& rhs)
    {
        return impl.are_equal(Caster<R,T>::eval(impl,lhs),
                              Caster<R,T>::eval(impl,rhs));
    }
};

template<size_t R,typename T>
struct AddOp{
    template <typename Impl,typename RHS_t>
    auto eval(const Impl& impl,
              const TensorPtr& lhs,
              const RHS_t& rhs)
    {
        return impl.add(Caster<R,T>::eval(impl,lhs),
                        rhs);
    }
};

template<size_t R,typename T>
struct ScaleOp{
    template <typename Impl>
    auto eval(const Impl& impl,
              TensorPtr& lhs,
              T scale)
    {
        return impl.scale(Caster<R,T>::eval(impl,lhs),scale);
    }

    template <typename Impl>
    auto eval(const Impl& impl,
              T scale,
              TensorPtr& rhs)
    {
        return impl.scale(Caster<R,T>::eval(impl,rhs),scale);
    }
};

template<size_t R,typename T>
struct SubtractOp{
    template<typename Impl>
    auto eval(const Impl& impl,
              TensorPtr& lhs,
              TensorPtr& rhs)
    {
        return impl.add(Caster<R,T>::eval(impl,lhs),
                        ScaleOp<R,T>().eval(impl,rhs,-1.0));
    }
};

}}//End namespaces
