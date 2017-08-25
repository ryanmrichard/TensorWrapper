#pragma once

namespace TWrapper {
namespace detail_ {

///A type for when we are getting something that does not have indices
struct IdxNotSet{};


template<typename Derived_t>
struct OperationBase{

    using derived_type=Derived_t;


    const derived_type& cast()const
    {
        return static_cast<const Derived_t&>(*this);
    }

    template<TensorTypes TT>
    auto eval()const->decltype(cast().template eval<TT>())
    {
        return cast().template eval<TT>();
    }
};


}}//End namespaces
