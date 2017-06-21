#pragma once
#include<tuple>

namespace TWrapper {
namespace detail_ {

template <typename T,typename Impl_t>
struct has_eval_method
{
    template <typename, typename> class checker;

    template <typename T2>
    static std::true_type test(checker<T2, decltype(&T2:: template eval<Impl_t>)> *);

    template <typename T2>
    static std::false_type test(...);

    typedef decltype(test<T>(nullptr)) type;

    static const bool value =
        std::is_same<std::true_type,decltype(test<T>(nullptr))>::value;
};

template<typename T>
using EnableIf = typename std::enable_if<T::value, T>::type;

template<typename T>
using DisableIf = typename std::enable_if<!T::value, T>::type;

template<typename fxn_t,typename...Args>
class Operation{
    fxn_t fxn_;
    std::tuple<Args...> args_;

    template<typename impl_t, typename T>
    auto deref_(impl_t impl, T ele, EnableIf<has_eval_method<T,impl_t>>*)
    {
        return ele.eval(impl);
    }

    template<typename impl_t,typename T>
    auto deref_(impl_t, T ele, DisableIf<has_eval_method<T,impl_t>>*)
    {
        return ele;
    }


public:
    Operation(fxn_t fxn, Args...args):
        fxn_(fxn),args_(args...)
    {}

    template<typename impl_t,size_t...Is>
    auto eval(impl_t impl,std::index_sequence<Is...>)
    {
        auto unpacked =std::make_tuple((deref_(impl,std::get<Is>(args_),nullptr))...);
        return fxn_.eval(impl,std::get<Is>(unpacked)...);
    }

    template<typename impl_t>
    auto eval(impl_t impl)
    {
        return eval(impl,
                    std::make_index_sequence<sizeof...(Args)>{});
    }
};

}}//End namespaces
