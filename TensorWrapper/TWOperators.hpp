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
}

template<typename LHS_t,size_t rank, typename T, typename Tensor_t>
bool operator==(const LHS_t& lhs,const TWrapper::TensorWrapper<rank,T,Tensor_t>& rhs)
{
return rhs==lhs;
}


template<typename LHS_t,size_t rank, typename T, typename Tensor_t>
bool operator!=(const LHS_t& lhs,const TWrapper::TensorWrapper<rank,T,Tensor_t>& rhs)
{
return rhs!=lhs;
}

#define TWOPERATOR(sym)\
    template<size_t rank, typename T, typename LHS_t, typename RHS_t>\
    auto operator sym(const TWrapper::TensorWrapper<rank,T,LHS_t>& lhs,\
                   const TWrapper::TensorWrapper<rank,T,RHS_t>& rhs)\
        ->decltype(lhs.tensor() sym rhs.tensor() )\
    {\
        return lhs.tensor() sym rhs.tensor();\
    }\
    \
    template<size_t rank, typename T, typename LHS_t, typename RHS_t>\
    auto operator sym(const TWrapper::TensorWrapper<rank,T,LHS_t>& lhs,\
                   const RHS_t& rhs)\
        ->decltype(lhs.tensor() sym rhs)\
    {\
        return lhs.tensor() sym rhs;\
    }\
    \
    template<size_t rank, typename T, typename LHS_t, typename RHS_t>\
    auto operator sym(const LHS_t& lhs,\
                   const TWrapper::TensorWrapper<rank,T,RHS_t>& rhs)\
        ->decltype(lhs sym rhs.tensor() )\
    {\
        return lhs sym rhs.tensor();\
    }

TWOPERATOR(+)
TWOPERATOR(-)
TWOPERATOR(!=)
#undef TWOPERATOR

template<size_t rank, typename T, typename LHS_t>
auto operator*(const TWrapper::TensorWrapper<rank,T,LHS_t>& lhs, double rhs)
    ->decltype(lhs.tensor() * rhs )
{
    return lhs.tensor()*rhs;
}

template<size_t rank, typename T, typename RHS_t>
auto operator*(double lhs,const TWrapper::TensorWrapper<rank,T,RHS_t>& rhs)
    ->decltype(rhs.tensor()*lhs )
{
    return rhs.tensor()*lhs;
}
