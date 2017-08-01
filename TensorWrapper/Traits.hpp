#pragma once
#include <cstddef>
#include <type_traits>

namespace TWrapper {
namespace detail_ {
enum class TensorTypes;
}

template<size_t R, typename T, detail_::TensorTypes TT>
class TensorWrapper;

template<size_t R,typename T>
class TensorWrapperBase;

namespace detail_ {

template<typename T>
class OperationBase;

///A type for discerning the rank and scalar type of a tensor
template<typename tensor_t>
struct TensorTraits{
    constexpr static bool value=false;
};

template<size_t R, typename T>
struct TensorTraits<TWrapper::TensorWrapperBase<R,T>>
{
    constexpr static bool value=true;
    static const size_t rank=R;
    using scalar_type=T;
    using type=TWrapper::TensorWrapperBase<R,T>;
};

template<size_t R, typename T, TensorTypes TT>
struct TensorTraits<TWrapper::TensorWrapper<R,T,TT>>
{
    constexpr static bool value=true;
    static const size_t rank=R;
    using scalar_type=T;
    using type=TWrapper::TensorWrapper<R,T,TT>;
};

///Type for removing const/reference
template<typename T>
using CleanType=
    typename std::remove_const<typename std::remove_reference<T>::type>::type;

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
template<typename Other_t>
using IsATWrapper=TensorTraits<CleanType<Other_t>>;

/** \brief A type that enables a function if \p Other_t is not convertible to
 *         a TensorWrapperBase.
 *
 * This type is intended to be used with SFINAE.  Specifically, put this type
 * in your list of template type parameters and it will enable a function if
 * \p Other_t is not convertible to a TensorWrapperBase.  We use this to comb
 *  our classes out of arbitrary template types.
 *
 * \tparam Other_t The type to be checked
 */
template<typename Other_t>
using EnableIfNotATWrapper=std::enable_if<!IsATWrapper<Other_t>::value,int>;

///Type for discerning if another type is an Operation
template<typename RHS_t>
struct OperationTraits
{
    static constexpr bool value=
            std::is_convertible<RHS_t,OperationBase<RHS_t>>::value;
};

template<typename RHS_t>
struct OperationTraits<OperationBase<RHS_t>>{
    using type=OperationBase<RHS_t>;
    static constexpr bool value=true;
};

template<typename RHS_t>
using IsAnOperation=OperationTraits<CleanType<RHS_t>>;

template<typename RHS_t>
struct IsOpOrTW
{
    static constexpr bool is_op=IsAnOperation<RHS_t>::value;
    static constexpr bool is_tw=IsATWrapper<RHS_t>::value;
    static constexpr bool value=is_op || is_tw;
};

template<typename RHS_t>
using EnableIfAnOperation=std::enable_if<IsAnOperation<RHS_t>::value,int>;

template<typename RHS_t>
using EnableIfNotAnOperation=std::enable_if<!IsAnOperation<RHS_t>::value,int>;

template<typename RHS_t>
using EnableIfOpOrTW=std::enable_if<IsOpOrTW<RHS_t>::value,int>;

template<typename RHS_t>
using EnableIfNotOpOrTW=std::enable_if<!IsOpOrTW<RHS_t>::value,int>;


}}//end namespaces
