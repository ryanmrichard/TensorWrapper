/** \file This file contains structs to be used for comparing a series of types.
 *
 */

#pragma once
#include <type_traits>
#include <tuple>

namespace TWrapper {
namespace detail_ {

/** \brief Determines the number of times a type appears in a list.
 *
 *  The resulting structure will contain a static member "value" which will be
 *  equal to the number of times the type appeared.
 *
 *  \tparam T The type for which the count will be found.
 *  \tparam List The list of types to search.
 */
template<typename T, typename...List>
struct TypeCount;

template<typename T>
struct TypeCount<T>:public std::false_type {};

///End recursion specialization for TypeCount
template<typename T, typename List>
struct TypeCount<T,List>: public std::is_same<T,List> {};

///Primary template of TypeCount
template<typename T, typename T2, typename...List>
struct TypeCount<T,T2,List...> {
    ///The number of times T appears in the list [T2,Args...]
    constexpr static size_t value=
            TypeCount<T,List...>::value+std::is_same<T,T2>::value;
};

/** \brief Makes a type containing all of the unique types in a list
 *
 *  Algorithmically let L be an N element long list.  Let L_I be the I-th
 *  permuation of the list (ordering of permutations is irrelevant) and let
 *  L_I[x] be the x-th element of L_I.  Then, at a depth of I we are going to
 *  compare L_I[0] with the list of elements starting at L_I[1] and ending at
 *  L_I[N-1] via TypeCount.  If TypeCount's value is zero, we know L_I[0] is
 *  unique and we add it to the tuple.
 *
 *  \note To understand why it's permuations and not just depth first, consider
 *  a list of types [int,double,int].  At a depth of 0 we correctly determine
 *  that int is non-unique; however, at a depth of 1 we incorrectly determine
 *  both double and int are unique if we do not consider the first int.
 *
 *  \tparam depth Which permutation are we?  Recursion stops when depth equals
 *                the size of \p List.
 *  \tparam T The first type in the list
 *  \tparam List The list for which we want the unique indices
 */
template<size_t depth, size_t done,typename Fxn_t,typename...List>
struct GetUniqueImpl;

template<typename Fxn_t>
struct GetUniqueImpl<0,0,Fxn_t>{
    using type=std::tuple<>;
};

///Recursion end point GetUniqueImpl
template<size_t done, typename Fxn_t, typename T, typename...List>
struct GetUniqueImpl<done,done,Fxn_t,T,List...> {
    using type=std::tuple<>;
};

///Primary template for GetUniqueImpl
template<size_t depth, size_t done, typename Fxn_t, typename T, typename...List>
struct GetUniqueImpl<depth,done,Fxn_t,T,List...> {
    constexpr static bool is_unique=Fxn_t()(TypeCount<T,List...>::value);
    using my_type=typename std::conditional<is_unique,
                                std::tuple<T>,std::tuple<>>::type;
    using base_type=typename GetUniqueImpl<depth+1,done,Fxn_t,List...,T>::type;
    using type=decltype(std::tuple_cat(my_type(),base_type()));
};

///Functor that returns true if its argument is true
struct IsTrue{constexpr bool operator()(bool val)const{return val;}};

///Functor that returns true if its argument is false
struct IsFalse{constexpr bool operator()(bool val)const{return !val;}};

///Specialization of GetUniqueImpl to actually get unique types
template<typename...Args>
using GetUnique=GetUniqueImpl<0,sizeof...(Args),IsFalse,Args...>;

///Specialization of GetUniqueImpl to get common types
template<typename...Args>
using GetCommon=GetUniqueImpl<0,sizeof...(Args),IsTrue,Args...>;

///Type whose member value will tell you whether on not all types are unique
template<typename...Args>
using AllUnique=std::is_same<std::tuple<Args...>,
                              typename GetUnique<Args...>::type>;

}}
