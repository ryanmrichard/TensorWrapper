#pragma once
#include "TensorWrapper/Operations/OperationBase.hpp"

namespace TWrapper {

template<size_t R, typename T, detail_::TensorTypes TT>
class TensorWrapper;

template<size_t R, typename T>
class TensorWrapperBase;

namespace detail_ {
template<size_t R, typename T>
class TensorPtr;


///Primary template for Convert, applies to scalars
template<typename data_t>
struct Convert: public OperationBase<Convert<data_t>>
{

    const data_t& data_;
    constexpr Convert(const data_t& data):
        data_(data)
    {}

    template<TensorTypes /*TT*/>
    constexpr const data_t& eval()const
    {
        return data_;
    }
};

template<size_t R, typename T, TensorTypes TT>
class Convert<TensorWrapper<R,T,TT>>;

template<size_t R, typename T>
class Convert<TensorWrapperBase<R,T>>;

template<size_t R, typename T>
class Convert<TensorPtr<R,T>>;

template class Convert<double>;

}}//End namespaces
