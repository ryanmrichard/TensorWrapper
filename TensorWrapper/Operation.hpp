#pragma once
#include<tuple>
#include "TensorWrapper/TensorImpl/TensorTypes.hpp"

namespace TWrapper {
namespace detail_ {

/** \brief This is the class that implements our lazy evaluation.
 *
 *
 *  \tparam fxn_t The type of the function/functor to apply to the arguments
 *  \tparam Args  The types of the arguments that will be passed to the
 *                function.
 */
template<typename fxn_t,typename...Args>
class Operation{
    std::tuple<Args...> args_;

    using index_seq=std::make_index_sequence<sizeof...(Args)>;

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
    auto deref_(T&& ele,int)->decltype(ele.template eval<T1>())
    {
        return std::move(ele.template eval<T1>());
    }

    template<TensorTypes,typename T>
    T&& deref_(T&& ele,long)
    {
        return ele;
    }

public:
    Operation(Args&&...args):
        args_(std::forward<Args>(args)...)
    {}

    template<TensorTypes T1,size_t...Is>
    auto eval(std::index_sequence<Is...>)
    {
        auto unpacked=std::make_tuple(
                    (deref_<T1>(std::get<Is>(args_),0))...);
        fxn_t fxn_;
        return std::move(fxn_.eval<T1>(std::get<Is>(unpacked)...));
    }

    template<TensorTypes T1>
    auto eval()
    {
        return std::move(eval<T1>(index_seq{}));
    }

};

template<typename fxn_t,typename...Args>
auto make_op(Args&&...args)
{
    return Operation<fxn_t,Args...>(std::forward<Args>(args)...);
}


template<typename...Args>
auto eval_op(TensorTypes type,Operation<Args...>& arg)
{

    if(type==TensorTypes::EigenMatrix)
        return std::move(arg.template eval<TensorTypes::EigenMatrix>());
//    else if(type==TensorTypes::EigenTensor)
//        return arg.template eval<TensorTypes::EigenTensor>();
    //Shouldn't ever be able to get here
    throw std::out_of_range("Tensor type is not recognized");
}


}}//End namespaces
