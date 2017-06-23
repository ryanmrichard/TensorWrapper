#pragma once

namespace TWrapper {
namespace detail_ {

/** \brief A type for discerning if a type is convertable to a
 *         TensorWrapperBase instance
 *
 *   If \p Other_t is convertable to a TensorWrapperBase this type will
 *   contain a static boolean member \p value that will be set to true,
 *   otherwise, said member will be false.
 *
 *  \tparam R The rank the resulting TensorWrapper should have
 *  \tparam T The scalar type the resulting TensorWrapperBase should have
 *  \tparam Other_t The type to check
 */
template<size_t R,typename T, typename Other_t>
using IsATWrapper=
    std::is_convertible<
        typename std::remove_const<
            typename std::remove_reference<Other_t>::type
        >::type,
    TWrapper::TensorWrapperBase<R,T>
    >;

/** \brief A type that enables a function if \p Other_t is not convertible to
 *         a TensorWrapperBase.
 *
 * This type is intended to be used with SFINAE.  Specifically, put this type
 * in your list of template type parameters and it will enable a function if
 * \p Other_t is not convertible to a TensorWrapperBase.  We use this to comb
 *  our classes out of arbitrary template types.
 *
 * \tparam R the rank the resulting TensorWrapperBase should have
 * \tparam T the scalar type the resulting TensorWrapperBase instance should
 *           have.
 * \tparam Other_t The type to be checked
 */
template<size_t R,typename T, typename Other_t>
using EnableIfNotATWrapper=std::enable_if<!IsATWrapper<R,T,Other_t>::value,int>;

template<typename RHS_t>
struct OperationTraits
{
    static constexpr bool value=false;
};

template<typename...RHS_t>
struct OperationTraits<Operation<RHS_t...>>{
    using type=Operation<RHS_t...>;
    static constexpr bool value=true;
};

template<typename RHS_t>
using IsAnOperation=
    OperationTraits<
        typename std::remove_const<
            typename std::remove_reference<RHS_t>::type
        >::type>;

template<typename RHS_t>
using EnableIfAnOperation=std::enable_if<IsAnOperation<RHS_t>::value,int>;

template<typename RHS_t>
using EnableIfNotAnOperation=std::enable_if<!IsAnOperation<RHS_t>::value,int>;

}}//end namespaces
