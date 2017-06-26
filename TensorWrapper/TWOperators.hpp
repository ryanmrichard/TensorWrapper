////This file not meant for inclusion outside TensorWrapper.hpp

//Although these are all very similar, they seem to differ enough to not make
//a macro worth it
template<size_t R,typename T>
auto operator+(const TWrapper::TensorWrapperBase<R,T>& lhs,
               const TWrapper::TensorWrapperBase<R,T>& rhs)
{
    using namespace TWrapper::detail_;
    return make_op<AddOp<R,T>>(lhs.de_ref(),rhs.de_ref());
}

template<size_t R,typename T,typename LHS_t,
    typename TWrapper::detail_::EnableIfNotATWrapper<R,T,LHS_t>::type=0>
auto operator+(LHS_t&& lhs,const TWrapper::TensorWrapperBase<R,T>& rhs)
{
    using namespace TWrapper::detail_;
    return make_op<AddOp<R,T>>(std::forward<LHS_t>(lhs),rhs.de_ref());
}

template<size_t R, typename T, typename RHS_t,
    typename TWrapper::detail_::EnableIfNotATWrapper<R,T,RHS_t>::type=0>
auto operator+(const TWrapper::TensorWrapperBase<R,T>& lhs,RHS_t&& rhs)
{
    using namespace TWrapper::detail_;
    return make_op<AddOp<R,T>>(lhs.de_ref(),std::forward<RHS_t>(rhs));
}

template<size_t R,typename T>
bool operator==(const TWrapper::TensorWrapperBase<R,T>& lhs,
                const TWrapper::TensorWrapperBase<R,T>& rhs)
{
    using namespace TWrapper::detail_;
    auto op=make_op<EqualOp<R,T>>(lhs.de_ref(),rhs.de_ref());
    return eval_op(lhs.type(),op);
}

template<size_t R,typename T,typename LHS_t,
    typename TWrapper::detail_::EnableIfNotATWrapper<R,T,LHS_t>::type=0>
bool operator==(LHS_t&& lhs,const TWrapper::TensorWrapperBase<R,T>& rhs)
{
    using namespace TWrapper::detail_;
    auto op=make_op<EqualOp<R,T>>(std::forward<LHS_t>(lhs),rhs.de_ref());
    return eval_op(rhs.type(),op);
}

template<size_t R, typename T, typename RHS_t,
    typename TWrapper::detail_::EnableIfNotATWrapper<R,T,RHS_t>::type=0>
bool operator==(const TWrapper::TensorWrapperBase<R,T>& lhs,RHS_t&& rhs)
{
    using namespace TWrapper::detail_;
    auto op=make_op<EqualOp<R,T>>(lhs.de_ref(),std::forward<RHS_t>(rhs));
    return eval_op(lhs.type(),op);
}

template<size_t R, typename T>
auto operator*(const TWrapper::TensorWrapperBase<R,T>& lhs,T rhs)
{
    using namespace TWrapper::detail_;
    return make_op<ScaleOp<R,T>>(lhs.de_ref(),rhs);
}

template<size_t R, typename T>
auto operator*(T lhs, const TWrapper::TensorWrapperBase<R,T>& rhs)
{
    return rhs*lhs;
}

////Some operators we miss with the above macro

/////Left multipy by a T
//template<size_t rank, typename T, TWrapper::detail_::TensorTypes T1>
//decltype(auto)
//operator*(T lhs,const TWrapper::TensorWrapper<rank,T,T1>& rhs)
//{
//    return rhs*lhs;
//}

/////Right multipy by a T
////template<size_t rank, typename T, TWrapper::detail_::TensorTypes T1>
////decltype(auto)
////operator*(T lhs,const TWrapper::TensorWrapper<rank,T,T1>& rhs)
////{
////    return rhs*lhs;
////}
