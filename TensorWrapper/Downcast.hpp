#pragma once
#include "TensorWrapper/Operation.hpp"
#include "TensorWrapper/TensorImpl/TensorWrapperImpl.hpp"

namespace TWrapper{
namespace detail_{

#define UNARY_ENTRY(type)\
if(ttype==type){\
TensorWrapperImpl<R,T,type> impl();\
Operation<Caster<R,T>,decltype(impl),TensorPtr&>\
cast_op(Caster<R,T>(),impl,ptensor);\
return Operation<fxn_t,decltype(cast_op),Args...>(fxn,cast_op,\
                std::forward<Args...>(args...));\
}

template<size_t R, typename T, typename fxn_t, typename...Args>
auto unary_downcast(TensorPtr& ptensor, TensorTypes ttype,
                    fxn_t fxn, Args...args)
{
    UNARY_ENTRY(TensorTypes::EigenMatrix)
}

#undef UNARY_ENTRY

}}//End namespaces
