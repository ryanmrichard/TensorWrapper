#pragma once
#include<tuple>
#include "TensorWrapper/TensorImpl/TensorTypes.hpp"

namespace TWrapper {
namespace detail_ {


/** \brief This is the class that implements our lazy evaluation.
 *
 *  \warning The internals of this class are far more complex than they look.
 *     More specifically because we need to do perfect forwarding we have a lot
 *     of innocent looking wrapper calls that are actually ensuring we get
 *     const, lvalue/rvalue references, and values when we need them.
 *
 *  If you are going to try to figure out how this class works I recommend you
 *  make note of the following:
 *  0. You better understand SFINAE
 *  1. auto as a return type ends up resolving to rvalues for nested function
 *     calls, i.e.
 *     \code{.cpp}
 *     int& fxn1(int& a){return a;}
 *     int a=1;
 *     auto b=fxn1(a);//b's type is deduced to be int not int&
 *     decltype(fxn1(a)) c= fxn1(a);//c's type deduced to be int&
 *     \endcode
 *  2. We don't always return references, i.e. we can't declare our return
 *     types as `auto&`
 *  3. decltype's can't be used on recursive functions
 *  4. decltype's can be used on functions of the same name that stem from
 *     different classes
 *
 *  Creating an instance of this class is straightforward, we perfect forward
 *  the arguments to this class into a tuple and wait for someone to call the
 *  eval method.  When that function is called is when things go nuts.  First we
 *  defer to a static member function of a subclass (see points 3 and 4 above)
 *  to recursively decide whether each argument we were provided can be
 *  forwarded as is or if it needs to be evaluated (i.e. it's an Operation
 *  instance itself).  For each element this handled by one of the two
 *  `deref_` member functions.  These functions rely on SFINAE to determine
 *  whether we need to call eval or not, they are conveniently wrapped inside
 *  another member function `deref` to consolidate the return values.
 *  The getting and evaluating of an element is then consolidated into a
 *  function `process_elem`.
 *
 *
 *  \tparam fxn_t The type of the function/functor to apply to the arguments
 *  \tparam Args  The types of the arguments that will be passed to the
 *                function.
 */
template<typename fxn_t,typename...Args>
class Operation{
    ///The arguments we will evaluate/pass to the function
    std::tuple<Args...> args_;

    /** \brief The overload for arguments sporting an eval function
     *
     *  \param[in] impl The function/functor to pass to the eval function
     *  \param[in] ele  The object to call eval on
     *  \param[in] int  Causes the compiler to select this overload when both
     *                  are valid because 0 is of type int, but convertible to
     *                  long
     *
     *  \return The result of the eval function
     */
    template<TensorTypes T1,typename T>
    auto deref_(T&& ele,int)const ->decltype(ele.template eval<T1>())
    {
        return ele.template eval<T1>();
    }

    ///Overload for elements that do not have an eval function
    template<TensorTypes,typename T>
    T&& deref_(T&& ele,long)const
    {
        return ele;
    }

    ///Consolidates the two deref_ calls under one name
    template<TensorTypes T1,typename T>
    auto deref(T&& ele)const ->decltype(deref_<T1>(ele,0))
    {
        return deref_<T1>(std::forward<T>(ele),0);
    }

    ///Consolidates the getting and evaluating of the elements
    template<TensorTypes T1,size_t depth>
    auto process_elem()const->decltype(deref<T1>(std::get<depth>(args_)))
    {
        return deref<T1>(std::get<depth>(args_));
    }

    ///Struct for recursively calling process_elem and forwarding the results
    template<TensorTypes T1,size_t depth>
    struct Evaluator{
        template<typename...NewArgs>
        static auto eval(const Operation<fxn_t,Args...>& parent,
                         NewArgs&&...args)->decltype(
                Evaluator<T1,depth+1>::eval(parent,
                                    std::forward<NewArgs>(args)...,
                                    parent.process_elem<T1,depth>()
                ))
        {
            return Evaluator<T1,depth+1>::eval(parent,
                    std::forward<NewArgs>(args)...,
                    parent.process_elem<T1,depth>()
            );
        }
    };

    ///End of recursion over function's arguments, actually runs the function
    template<TensorTypes T1>
    struct Evaluator<T1,sizeof...(Args)>{
        template<typename...NewArgs>
        static auto eval(const Operation<fxn_t,Args...>&,NewArgs&&...args)
                ->decltype(
                    fxn_t().template eval<T1>(std::forward<NewArgs>(args)...))
        {
            return fxn_t().template eval<T1>(std::forward<NewArgs>(args)...);
        }
    };



public:

    Operation(Args&&...args):
        args_(std::forward<Args>(args)...)
    {}

    template<TensorTypes T1>
    auto eval()const->decltype(Evaluator<T1,0>::eval(*this))
    {
        return Evaluator<T1,0>::eval(*this);
    }

};

template<typename fxn_t,typename...Args>
Operation<fxn_t,Args...> make_op(Args&&...args)
{
    return Operation<fxn_t,Args...>(std::forward<Args>(args)...);
}

///Functor to be used with the next function, calls eval on the operation
struct EvalOpFunctor{
    template<TensorTypes T1,typename...Args>
    auto eval(const Operation<Args...>& arg)->decltype(arg.template eval<T1>())
    {
        return arg.template eval<T1>();
    }

};

/** \brief A convenience function for evaluating a nested operation
 *
 *  \return The result of the operation, the exact value of which depends on
 *  the operation being evaluated.
 */
template<typename...Args>
auto eval_op(TensorTypes type,const Operation<Args...>& arg)->
    decltype(apply_TensorTypes<EvalOpFunctor>(type,arg))
{
    return apply_TensorTypes<EvalOpFunctor>(type,arg);
}





}}//End namespaces
