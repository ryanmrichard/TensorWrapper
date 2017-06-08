//This file not meant for inclusion outside TensorWrapper.hpp
namespace TWrapper{
template<size_t rank, typename T, typename Tensor_t>
template<typename RHS_t>
bool TensorWrapper<rank,T,Tensor_t>::operator==(const RHS_t& other)const
{
    return impl_.are_equal(tensor_,other);
}


template<size_t rank, typename T, typename Tensor_t>
template<typename RHS_t>
bool TensorWrapper<rank,T,Tensor_t>::operator!=(const RHS_t& other)const
{
    return ! impl_.are_equal(tensor_,other);
}



} //End namespace TWrapper

#define TWOPERATOR(sym,name)\
    template<typename LHS_t,size_t rank, typename T, typename Tensor_t>\
    auto operator sym(const LHS_t& lhs,\
    const TWrapper::TensorWrapper<rank,T,Tensor_t>& rhs)\
    {\
    TWrapper::detail_::TensorWrapperImpl<rank,T,Tensor_t> impl;\
    return impl.name(lhs,rhs.tensor());\
    }\
    template<typename LHS_t, size_t LHS_rank, typename T1,\
             typename RHS_t, size_t RHS_rank, typename T2>\
    auto operator sym(const TWrapper::TensorWrapper<LHS_rank,T1,LHS_t>& lhs,\
                      const TWrapper::TensorWrapper<RHS_rank,T2,RHS_t>& rhs)\
    {\
    return lhs sym rhs.tensor();\
    }

TWOPERATOR(+,add)
TWOPERATOR(-,subtract)
TWOPERATOR(==,are_equal)
#undef TWOPERATOR

///Left multipy by a double
template<size_t rank, typename T, typename Tensor_t>
decltype(auto) operator*(double lhs,
                         const TWrapper::TensorWrapper<rank,T,Tensor_t>& rhs)
{
    return rhs*lhs;
}
