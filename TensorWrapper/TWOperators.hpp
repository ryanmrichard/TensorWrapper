#pragma once
/** \file This file defines all of the operators needed to complete the
 *  syntactic sugar.  Their definitions are messy.
 */



namespace TWrapper{
namespace detail_{

template<typename fxn_t,typename LHS_t, typename RHS_t>
struct OperatorOverloader;

template<template<size_t,typename> fxn_t,size_t R, typename T>
struct OperatorOverloader<fxn_t<R,T>,TensorWrapperBase<R,T>,
                                TensorWrapperBase<R,T>>
{
    auto eval(const TensorWrapperBase<R,T>& lhs,
              const TensorWrapperBase<R,T>& rhs)->
    decltype(make_op<fxn_t<R,T>>(lhs.de_ref(),rhs.de_ref()))
    {
        return make_op<fxn_t<R,T>>(lhs.de_ref(),rhs.de_ref());
    }
};

template<typename fxn_t,size_t R, typename T, typename...RHS_t>
struct OperatorOverloader<fxn_t,TensorWrapperBase<R,T>,
                                   Operation<RHS_t...>>
{
    auto eval(const TensorWrapperBase<R,T>& lhs,
              const Operation<RHS_t...>& rhs)->
    decltype(make_op<fxn_t>(lhs.de_ref(),rhs))
    {
        return make_op<fxn_t>(lhs.de_ref(),rhs);
    }
};

template<typename fxn_t,size_t R, typename T, typename...LHS_t>
struct OperatorOverloader<fxn_t,Operation<LHS_t...>,
                                     TensorWrapperBase<R,T>>
{
    auto eval(const Operation<LHS_t...>& lhs,
              const TensorWrapperBase<R,T>& rhs)->
    decltype(make_op<fxn_t>(lhs,rhs.de_ref()))
    {
        return make_op<fxn_t>(lhs,rhs.de_ref());
    }
};

template<typename fxn_t,size_t R, typename T, typename RHS_t>
struct OperatorOverloader<fxn_t,TensorWrapperBase<R,T>,RHS_t>
{
    auto eval_(const TensorWrapperBase<R,T>& lhs, RHS_t&& rhs,long)->
    decltype(make_op<fxn_t>(rhs.de_ref(),
        make_op<DeRef<R,T>>(
            TensorPtr<R,T>(lhs.type(),std::forward<RHS_t>(rhs)))))
    {
        TensorPtr<R,T> ptemp(lhs.type(),std::forward<RHS_t>(rhs));
        return make_op<fxn_t>(make_op<DeRef<R,T>>(ptemp),lhs.de_ref());
    }

    auto eval_(const TensorWrapperBase<R,T>& lhs,  RHS_t&& rhs,int)->
    decltype(make_op<fxn_t>(lhs.de_ref(),std::forward<RHS_t>(rhs)))
    {
        return make_op<fxn_t>(lhs.de_ref(),std::forward<RHS_t>(rhs));
    }

    auto eval(const TensorWrapperBase<R,T>& lhs, RHS_t&& rhs)->
    decltype(eval_(lhs,std::forward<RHS_t>(rhs),0))
    {
        return eval_(lhs,std::forward<RHS_t>(rhs),0);
    }
};

template<typename fxn_t,size_t R, typename T, typename LHS_t>
struct OperatorOverloader<fxn_t,LHS_t,TensorWrapperBase<R,T>>
{
    auto eval_(LHS_t&& lhs,const TensorWrapperBase<R,T>& rhs,long)->
    decltype(make_op<fxn_t>(
        make_op<DeRef<R,T>>(
            TensorPtr<R,T>(rhs.type(),std::forward<LHS_t>(lhs))),
        rhs.de_ref()))
    {
        TensorPtr<R,T> ptemp(rhs.type(),std::forward<LHS_t>(lhs));
        return make_op<fxn_t>(make_op<DeRef<R,T>>(ptemp),rhs.de_ref());
    }

    auto eval_(LHS_t&& lhs,const TensorWrapperBase<R,T>& rhs,int)->
    decltype(make_op<fxn_t>(std::forward<LHS_t>(lhs),rhs.de_ref()))
    {
        return make_op<fxn_t>(std::forward<LHS_t>(lhs),rhs.de_ref());
    }

    auto eval(LHS_t&& lhs,const TensorWrapperBase<R,T>& rhs)->
    decltype(eval_(std::forward<LHS_t>(lhs),rhs,0))
    {
        return eval_(std::forward<LHS_t>(lhs),rhs,0);
    }
};

}}//End namespaces




///Signature of an operator between two tensorwrapper bases

///Signature of an operator between a tensorwrapper base and an operation
#define WOSig(sym,rexpr)\
template<size_t R,typename T,typename RHS_t,\
    typename EnableIfAnOperation<RHS_t>::type=0>\
auto operator sym(const TWrapper::TensorWrapperBase<R,T>& lhs,\
                  RHS_t&& rhs)->decltype(rexpr)

#define OWSig(sym,rexpr)\
template<size_t R,typename T,typename LHS_t,\
    typename EnableIfAnOperation<LHS_t>::type=0>\
auto operator sym(LHS_t&& rhs, const TWrapper::TensorWrapperBase<R,T>& lhs)->decltype(rexpr)

///Signature of an operator between anything that works and a wrapper base
#define BWSig(sym,rexpr)\
template<size_t R,typename T,typename LHS_t,\
    typename TWrapper::detail_::EnableIfNotAnOperationTWrapper<R,T,LHS_t>::type=0>\
auto operator sym(LHS_t&& lhs,const TWrapper::TensorWrapperBase<R,T>& rhs)\
    ->decltype(rexpr)

///Signature of an operator between a wrapper base and anything that works
#define WBSig(sym,rexpr)\
template<size_t R, typename T, typename RHS_t,\
    typename TWrapper::detail_::EnableIfNotATWrapper<R,T,RHS_t>::type=0>\
auto operator sym(const TWrapper::TensorWrapperBase<R,T>& lhs,RHS_t&& rhs)->\
    decltype(rexpr)

///Macro for making an operation
#define MAKE_OP(OpName,lhs,rhs)\
TWrapper::detail_::make_op<TWrapper::detail_::OpName<R,T>>(lhs,rhs)

///macro for making the op between two wrapper bases
#define WWOp(OpName)MAKE_OP(OpName,lhs.de_ref(),rhs.de_ref())
#define BWOp(OpName)MAKE_OP(OpName,std::forward<LHS_t>(lhs),rhs.de_ref())
#define WBOp(OpName)MAKE_OP(OpName,lhs.de_ref(),std::forward<RHS_t>(rhs))

///Macro for evaluating an operation
#define EvalOp(type,op)\
TWrapper::detail_::eval_op(type,op)

#define OPSet(OpName,sym)\
    WWSig(sym,WWOp(OpName)){return WWOp(OpName);}\
    BWSig(sym,BWOp(OpName)){return BWOp(OpName);}\
    WBSig(sym,WBOp(OpName)){return WBOp(OpName);}

#define EvalOpSet(OpName, sym)\
    WWSig(sym,EvalOp(lhs.type(),WWOp(OpName)))\
        {return EvalOp(lhs.type(),WWOp(OpName));}\
    BWSig(sym,EvalOp(rhs.type(),BWOp(OpName)))\
        {return EvalOp(rhs.type(),BWOp(OpName));}\
    WBSig(sym,EvalOp(lhs.type(),WBOp(OpName)))\
        {return EvalOp(lhs.type(),WBOp(OpName));}

//Our operations (finally)
OPSet(AddOp,+)
OPSet(SubtractionOp,-)
EvalOpSet(EqualOp,==)

//Scalar multiplication
template<size_t R, typename T>
auto operator*(const TWrapper::TensorWrapperBase<R,T>& lhs,T rhs)
{
    return MAKE_OP(ScaleOp,lhs.de_ref(),rhs);
}

template<size_t R, typename T>
auto operator*(T lhs, const TWrapper::TensorWrapperBase<R,T>& rhs)
{
    return rhs*lhs;
}

#undef EvalOpSet
#undef OpSet
