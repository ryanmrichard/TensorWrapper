#pragma once
#include <utility>
#include <tuple>
#include <cstdlib>

/** \file Contains the classes and free functions for performing compile-time
 *  parsing of string indices.
 *
 *  Throughout TensorWrapper we assume that every index to be summed over
 *  occurs at most twice in a term.  This ensures all products can be performed
 *  pairwise (I think that this must be the case in general, and that it follows
 *  from the associative property of the tensor product, but I'm admittedly not
 *  sure if that is a  proof...).  This still leaves two situations: the two
 *  indices occur on the smae tensor (like the trace of a matrix) or they occur
 *  on two different tensors in the term.  We leave it for the IndexedTensor
 *  class to take care of the dispatching of these two cases.
 * 
 *  It is perhaps also worth clarifying what assuming an index only occurs twice
 *  actually entails for this machinery.  Basically it boils down to the
 *  get_free/get_dummy functions which assume that if an index is unique/common
 *  it is  free/dummy.  No check is done to ensure that for example a unique
 *  index occurs only one within the tensor it is unique to (unique-ness is
 *  only checked against the other tensor).  Similarly no check is done to
 *  see if a common index only appears twice (that is it may additionally appear
 *  two or more times within one or both tensors).  The rest of the machinery
 *  here is quite general and should work even if an index does not appear only
 *  twice.
 *
 *  This whole file can really use some clean-up.  It should be possible to
 *  rewrite most (all?) of the routines in terms of get_map
 *
 */

namespace TWrapper {
namespace detail_ {

///Class for compile-time hashing of a compile time string
template<char...Chars>
struct HashedCString{};

template<char c>
struct HashedCString<c>
{
    constexpr static size_t value=static_cast<size_t>(c) + 0x9e3779b9;
};

template <char c,char...Chars>
struct HashedCString<c,Chars...>{
    constexpr static size_t old_hash=HashedCString<Chars...>::value;
    constexpr static size_t value=
         old_hash^static_cast<size_t>(c)+0x9e3779b9+(old_hash<<6)+(old_hash>>2);
};

/** \brief Compile-time class to hold an index.
 *
 *  This class serves to wrap a compile-time string literal in a manner
 *  suitable for template metaprogramming.
 *
 *  \tparam Chars The literal characters in the index the user provided us.
 */
template <char...Chars>
struct C_String {

    ///Returns true if this instance is the same index as the other instance
    template<char...Chars2>
    constexpr bool operator==(const C_String<Chars2...>&)
    {
        return std::is_same<C_String<Chars...>,C_String<Chars2...>>::value;
    }

    ///Returns true if this instance is not the same index as the other instance
    template<char...Chars2>
    constexpr bool operator!=(const C_String<Chars2...>& other)
    {
        return !((*this)==other);
    }

    constexpr static size_t hash_value=HashedCString<Chars...>::value;
};


/** \brief Function for turning our input string into the appropriate C_String.
 *
 *  \tparam Str A class that returns the string to type-ize
 *  \tparam indices the elements of the std::index_sequence
 */
template <typename Str, size_t...I>
auto build_string(std::index_sequence<I...>) {
        return C_String<Str().chars[I]...>();
}

///Macro for making it easy for the user to declare indices.
#define make_index(str) [&]{\
        struct Str { const char * chars = str; };\
        return TWrapper::detail_::build_string<Str>(\
                std::make_index_sequence<sizeof(str)>());\
}()




/** \brief A class to hold the various indices associated with a tensor.
 *
 *  This class is designed to be used at compile-time.  Ultimately it is pure
 *  template meta-programming with a pretty face thanks to C++11's constexpr.
 *
 *  If this class gets too slow try writing in terms of hashes.  I started
 *  making a slimmed down version of this file in Indices2.hpp that uses
 *  hashes.
 *
 *  \TODO: there's alot of private recursion going on that is pretty similar.
 *  It should be possible to consolidate into the get_common/get_unique
 *  recursion trees, but at the moment those routines are written in terms of
 *  the other ones.
 *
 * \tparam Args The indices wrapped in this instance.
 */
template<typename...Args>
class Indices{
private:

    ///Helper type to get the type of the I-th index
    template<size_t I>
    using TypeI=typename std::tuple_element<I,std::tuple<Args...>>::type;

    ///The return type when recursion ends
    template<size_t I,typename return_t>
    using EndRecursion=
        typename std::enable_if<I==sizeof...(Args),return_t>::type;

    ///The return type while recursion is occurring
    template<size_t I,typename return_t>
    using Recursion=
        typename std::enable_if<I!=sizeof...(Args),return_t>::type;

    ///End point for counting an index
    template<size_t I, char...Chars>
    constexpr
    static EndRecursion<I,size_t> count_impl(const C_String<Chars...>&)
    {
        return 0;
    }

    ///Recursion body for counting
    template<size_t I, char...Chars>
    constexpr
    static Recursion<I,size_t> count_impl(const C_String<Chars...>& other)
    {
        return (get<I>()==other) + count_impl<I+1>(other);
    }

    template<size_t...I>
    constexpr
    static std::array<size_t,sizeof...(I)>
    get_counts_impl(std::index_sequence<I...>)
    {
        return {count(get<I>())...};
    }


    template<size_t I,typename...Args2>
    constexpr static bool is_common(const Indices<Args2...>& other)
    {
        return other.count(get<I>())!=0;
    }


    template<size_t I,typename...Args2>
    constexpr
    static EndRecursion<I,size_t> ncommon_impl(const Indices<Args2...>&)
    {
        return 0;
    }

    template<size_t I,typename...Args2>
    constexpr
    static Recursion<I,size_t> ncommon_impl(const Indices<Args2...>& other)
    {
        return is_common<I>(other) + ncommon_impl<I+1>(other);
    }


    template<size_t, size_t depth,typename...Args2>
    constexpr
    static EndRecursion<depth,size_t>
    ith_common_impl(size_t,const Indices<Args2...>&)
    {
        return depth;
    }

    template<size_t counter, size_t depth,typename...Args2>
    constexpr static Recursion<depth,size_t>
    ith_common_impl(size_t I,const Indices<Args2...>& other)
    {
        constexpr bool is_good = is_common<depth>(other);
        return (is_good && counter==I)? depth :
                      ith_common_impl<counter+is_good,depth+1>(I,other);
    }

    template<typename...Args2,size_t...I>
    constexpr static std::array<size_t,sizeof...(I)>
    get_common_impl(const Indices<Args2...>& other,
                    std::index_sequence<I...>)
    {
        return {ith_common(I,other)...};
    }

    template<typename...Args2,size_t...I>
    constexpr static std::array<size_t,sizeof...(I)>
    get_unique_impl(const Indices<Args2...>& other,
                    std::index_sequence<I...>)
    {
        return {ith_unique(I,other)...};
    }

    template<size_t I, size_t, char...Chars>
    constexpr static EndRecursion<I,size_t>
    position_impl(size_t, const C_String<Chars...>&)
    {
        return I;
    }

    template<size_t I, size_t counter, char...Chars>
    constexpr static Recursion<I,size_t>
    position_impl(size_t cnt, const C_String<Chars...>& other)
    {
        constexpr bool is_good=(get<I>()==other);
        return (is_good && counter==cnt) ? I :
                              position_impl<I+1,counter+is_good>(cnt,other);
    }

    template<typename RHS_t,size_t...I>
    constexpr static std::array<size_t,sizeof...(I)>
    get_map_impl(const RHS_t& rhs,std::index_sequence<I...>)noexcept
    {
        return {rhs.position(0,get<I>())...};
    }

public:
    /** \brief Returns the number of indices contained within this index set.
     *
     *
     * \returns The number of indices contained within this index set.
     * \throw None. No throw guarantee
     */
    constexpr static size_t size() noexcept
    {
        return sizeof...(Args);
    }

    /** \brief Returns the \p I -th index in the set.
     *
     * \note The index to return must be passed as a template non-type to avoid
     * this function having multiple return types.
     *
     * \TODO If I is not in the range [0,size()) you will get some cryptic
     * compiler error along the lines of "invalid use of incomplete type
     * struct std::tuple_element<0,std::tuple<>>".  We should probably catch
     * the cases where I>=size() and print a more readable error.  This
     * requires us to not rely on std::tuple.
     *
     * \tparam I The index to return.  Must be in the range [0,size()).
     *
     * \returns The requested index.
     * \throws None No throw guarantee.
     */
    template<size_t I>
    constexpr static TypeI<I> get() noexcept
    {
        return std::get<I>(std::tuple<Args...>());
    }


    /** \brief Returns the number of times an index appears in the current set.
     *
     * \param[in] idx The index we want the count of.
     * \returns A value in the range [0,size()) corresponding to the number of
     *          times \p idx appears in the current set.
     * \tparam Chars The characters in the idx we are looking for.
     * \throws None.  No throw guarantee.
     */
    template<char...Chars>
    constexpr static size_t count(const C_String<Chars...>& idx)noexcept
    {
        return count_impl<0>(idx);
    }

    /** \brief Returns an array containing the number of times each index
     *  appears in this set.
     *
     *  This is basically a wrapper around the count() method that calls it
     *  for each index.
     *
     *  \returns A size() element array where element i is the number of times
     *  the i-th index appears in this index set.
     *  \throws None. No throw guarantee.
     *
     */
    constexpr static std::array<size_t,sizeof...(Args)>
    get_counts()noexcept
    {
        return get_counts_impl(std::make_index_sequence<sizeof...(Args)>());
    }


    /** \brief Returns the number of indices common to this set of indices and
     *  the set given by \p other.
     *
     *  For each of the indices in this set this function will determine if that
     *  index appears in \p other.  This function will then return the number of
     *  times the above procedure returned true.  Although this sounds simple
     *  there's a couple of gotchas the table should explain better than words.
     *
     *
     *  <table>
     *  <tr><th>This set                      <th>Other set        <th>Result
     *  <tr><td>i,j,k                         <td>l,m,n            <td>0
     *  <tr><td>i,j,k                         <td>i,j,k            <td>3
     *  <tr><td>i,j,k                         <td>j,k,l            <td>2
     *  <tr><td>i,j,k                         <td>i,i,k            <td>2
     *  <tr><td>i,i,k                         <td>i,j,k            <td>3
     *  </table>
     *
     *  The first three rows are straightforward.  Pay particular attention to
     *  the last two rows where an index is repeated within a set.  If the
     *  set with the repeated index is "this set" each of the indices will be
     *  checked.  It is not this function's job to see if an index is repeated
     *  within a set, but rather to see if it is repeated between sets.  The
     *  former task falls to the count() function.
     *
     *  \note This function is not necessarilly commutative, i.e. in general
     *  this->ncommon(other)!=other.ncommon(*this)
     *
     *  \param[in] other The set to compare this instance with.
     *  \returns The number of indices common among this set and the other set.
     *           The result will be in the range [0,size()).
     *  \tparam Args2 The types of the indices stored in the other set.
     *  \throw None. No throw guarantee.
     */
    template<typename...Args2>
    constexpr static size_t ncommon(const Indices<Args2...>& other)noexcept
    {
        return ncommon_impl<0>(other);

    }

    /** \brief Returns the position of the \p I -th index in this set that is
     *  common to both this set and \p other.
     *
     *  Like ncommon() usage of this function is straightforward although the
     *  answers may come as gotchas if one is not careful.  Again, the trouble
     *  likely will be for cases where there are repeated indices within one
     *  of the two sets.  For example, the position of the 0-th and first
     *  common index to the sets i,i,k and k,j,i is 0 and 1 not 0 and 2.
     *
     *  \param[in] i Which unique index do you want the position of? i must be
     *             in the range [0,ncommon(other)).
     *  \param[in] other The set to compare against.
     *
     *  \returns An integer in the range [0,size()) corresponding to the
     *  position of the \p i -th common index.
     *
     *  \tparam Args2 The indices in the other set.
     *
     *  \throw None.  No throw guarantee.
     */
    template<typename...Args2>
    constexpr static size_t
    ith_common(size_t i,const Indices<Args2...>& other)noexcept
    {
        return ith_common_impl<0,0>(i,other);
    }


    /** \brief Returns all the positions of the indices in this set that are
     *  common to other.
     *
     *  This function is a convenience function built around ith_common that
     *  retrieves all ncommon(other) of the common indices for you.  Thus the
     *  same caveats apply here.
     *
     *  \param[in] other The set to compare against
     *  \returns An array, \p array, containing ncommon(other) size_t's where
     *  the i-th element of \p array is the position of the i-th common index
     *  in this.
     *  \tparam Args2 The indices in the other set.
     *  \throws None. No throw guarantee.
     */
    template<typename...Args2>
    constexpr static auto get_common(const Indices<Args2...>& other)noexcept
    {
        constexpr size_t ncomm=ncommon(other);
        return get_common_impl(other,std::make_index_sequence<ncomm>());
    }


    /** \brief Returns the number of indices that are in this set, but not the
     *  other set.
     *
     *  In terms of set theory this is the set difference of this set's indices
     *  with \p other 's indices.
     *
     * \param[in] other The set to compare with.
     * \returns A value in the range [0,size()) corresponding to the number of
     *          indices that only appear in this set.
     * \tparam Args2 The indices in \p other
     * \throw None. No throw guarantee.
     */
    template<typename...Args2>
    constexpr static size_t nunique(const Indices<Args2...>& other)noexcept
    {
        return size()-ncommon(other);
    }

    /** \brief Returns the \p i -th index that is in this set, but not the other
     *  set.
     *
     *
     *  \param[in] i Which of the nunique(other) indices should we return? i
     *             should be in the range [0,nunique(other))
     *  \param[in] other The set of indices to compare against.
     *  \returns An integer in the range [0,size()) corresponding to position in
     *  this set of the \p i-th unique index, or if \p i >= size() then the
     *  number of unique indices size().
     *  \tparam Args2 The indices in the other set.
     *  \throws None. No throw guarantee.
     */
    template<typename...Args2>
    constexpr static size_t
    ith_unique(size_t i, const Indices<Args2...>& other)noexcept
    {

        constexpr size_t com=ncommon(other);
        constexpr std::array<size_t,com> common=get_common(other);
        constexpr size_t N=size();
        size_t counter=0;
        for(size_t k=0;k<N;++k)
        {
            bool is_good=true;
            for(size_t j=0;j<com && is_good; ++j)
                is_good=(common[j]!=k);

            if(is_good && counter==i)
                return k;
            counter+=is_good;
        }
        return N;
    }


    /** \brief Returns the position of the indices that are in this index, but
     *  not in other.
     *
     *
     *  \param[in] other The set to compare with.
     *  \return An nunique(other) element array, where element i is the position
     *  of the i-th unique index in this.
     *  \tparam Args2 The indices in the other set
     *  \throw None. No throw guarantee
     */
    template<typename...Args2>
    constexpr static auto get_unique(const Indices<Args2...>& other)noexcept
    {
        constexpr size_t unq=nunique(other);
        return get_unique_impl(other,std::make_index_sequence<unq>());
    }

    template<size_t i,typename...Args2>
    constexpr static bool
    is_unique(const Indices<Args2...>& other)noexcept
    {
        constexpr auto unique=get_unique(Indices<Args2...>{});
        constexpr size_t nunq=nunique(Indices<Args2...>{});
        for(size_t j=0;j<nunq;++j)
            if(unique[j]==i)
                return true;
        return false;
    }


    template<typename RHS_t>
    constexpr auto static get_map(const RHS_t& rhs)noexcept
    {
        constexpr size_t rank=size();
        return get_map_impl(rhs,std::make_index_sequence<rank>());
    }

    /** \brief Returns the position of the \p cnt -th occurence of index
     *  \p other in this set.
     *
     * \param[in] cnt Which occurance of \p other do you want the position of?
     *            cnt should be in the range [0,count(other)).
     * \param[in] other The index we are looking for.
     * \returns An integer in the range [0,size()) corresponding to the position
     * of the \p cnt -th occurance of the \p other. Or, if \p other does not
     * occur at least \p cnt times in this index, size().
     * \tparam Chars The letters in the index we are looking for.
     * \throw None.  No thorw guarantee.
     */
    template<char...Chars>
    constexpr static size_t position(size_t cnt, const C_String<Chars...>& other)
    {
        return position_impl<0,0>(cnt,other);
    }

};

template<typename...OtherIndxs>
using make_indices=Indices<OtherIndxs...>;

template<typename LHS_t, typename RHS_t>
struct MakeUnion{};

template<typename LHS_t, typename...Args>
struct MakeUnion<LHS_t,Indices<Args...>>
{
    using type=Indices<LHS_t,Args...>;
};


template<size_t Idx, bool Lgood, bool Rgood, size_t LMax, size_t RMax,
         typename LHS_t, typename RHS_t>
struct FreeIndicesImpl;

template<size_t LMax, size_t RMax,typename LHS_t, typename RHS_t>
struct FreeIndicesImpl<RMax,false,false,LMax,RMax,LHS_t,RHS_t>
{
    using type=Indices<>;
};

template<size_t Idx, size_t LMax, size_t RMax, typename LHS_t, typename RHS_t>
struct FreeIndicesImpl<Idx,false,true,LMax,RMax,LHS_t,RHS_t> : public
        FreeIndicesImpl<Idx+1,false,Idx+1!=RMax,LMax,RMax,LHS_t,RHS_t>
{
    constexpr static bool is_unique=RHS_t::template is_unique<Idx>(LHS_t{});
    constexpr static bool rgood=Idx+1!=RMax;
    using base_type=typename
        FreeIndicesImpl<Idx+1,false,Idx+1!=RMax,LMax,RMax,LHS_t,RHS_t>::type;
    using Idx_t=decltype(RHS_t::template get<Idx>());
    using unique_type=typename MakeUnion<Idx_t,base_type>::type;
    using type=typename std::conditional<is_unique,unique_type,base_type>::type;
};

template<size_t Idx, size_t LMax, size_t RMax,typename LHS_t, typename RHS_t>
struct FreeIndicesImpl<Idx,true,true,LMax, RMax,LHS_t,RHS_t> : public
     FreeIndicesImpl<Idx+1!=LMax?Idx+1:0,Idx+1!=LMax,true,LMax,RMax,LHS_t,RHS_t>
{
    constexpr static bool is_unique=LHS_t::template is_unique<Idx>(RHS_t{});
    constexpr static bool lgood=Idx+1!=LMax;
    constexpr static bool rgood=0!=RMax;
    using base_type=typename
        FreeIndicesImpl<lgood?Idx+1:0,lgood,true,LMax,RMax,LHS_t,RHS_t>::type;
    using Idx_t=decltype(LHS_t::template get<Idx>());
    using unique_type=typename MakeUnion<Idx_t,base_type>::type;
    using type=typename std::conditional<is_unique,unique_type,base_type>::type;
};

template<typename LHS_t, typename RHS_t>
struct FreeIndices
{
    constexpr static size_t LMax=LHS_t::size();
    constexpr static size_t RMax=RHS_t::size();
    constexpr static bool lgood=0!=LMax;
    constexpr static bool rgood=0!=RMax;
    using type=
        typename FreeIndicesImpl<0,lgood,rgood,LMax,RMax,LHS_t,RHS_t>::type;
};

template<typename LHS_t, typename RHS_t, size_t...NLHS,size_t...NRHS>
constexpr std::pair<std::array<size_t,sizeof...(NLHS)>,
                    std::array<size_t,sizeof...(NRHS)>>
get_free_impl(const LHS_t& lhs,
              const RHS_t& rhs,
              std::index_sequence<NLHS...>,
              std::index_sequence<NRHS...>)
{
    return {{LHS_t::ith_unique(NLHS,rhs)...},
            {RHS_t::ith_unique(NRHS,lhs)...}};
}


/** \brief Returns the free indices in a pairwise contraction.
 *
 *  Given the product of two tensors, the left having rank L, and the right
 *  having rank R our goal is to determine the set of indices that are free
 *  (i.e. appear only once in either tensor).
 *
 *
 */
template<typename LHS_t, typename RHS_t>
constexpr auto get_free(const LHS_t& lhs, const RHS_t& rhs)noexcept
{
   constexpr size_t lnfree=lhs.nunique(rhs);
   constexpr size_t rnfree=rhs.nunique(lhs);
   return get_free_impl(lhs,rhs,
                        std::make_index_sequence<lnfree>(),
                        std::make_index_sequence<rnfree>());
}

template<typename LHS_t, typename RHS_t, size_t...NLHS>
constexpr std::pair<std::array<size_t,sizeof...(NLHS)>,
                    std::array<size_t,sizeof...(NLHS)>>
get_dummy_impl(const LHS_t& lhs,
               const RHS_t& rhs,
               std::index_sequence<NLHS...>)
{
    constexpr auto lcommon=lhs.get_common(rhs);
    if(lcommon.size()==0)//Technically not needed, but shuts compiler up
        return {{},{}};
    return {{lcommon[NLHS]...},
            {RHS_t::position(0,lhs.template get<lcommon[NLHS]>())...}};
}


/** \brief Returns the dummy indices in a pairwise contraction.
 *
 *  \note This function will fail to compile if
 *     lhs.ncommon(rhs)!=rhs.ncommon(lhs).  This can only occur if
 *     an index common to the two tensors occurs L times in the left one and
 *     R times in the right one such that L!=R.  Hence this will catch a lot
 *     of use cases where an index does not appear at most twice, but will not
 *     catch them all.
 *
 *
 */
template<typename LHS_t, typename RHS_t>
constexpr auto get_dummy(const LHS_t& lhs, const RHS_t& rhs)noexcept
{
   constexpr size_t lcommon=lhs.ncommon(rhs);
   constexpr size_t rcommon=rhs.ncommon(lhs);
   static_assert(lcommon==rcommon,"Error an index appears more than twice");
   return get_dummy_impl(lhs,rhs,std::make_index_sequence<lcommon>());
}

template<char...Cs>
std::string stringify_impl(C_String<Cs...>)
{
    char str[]={Cs...};
    return std::string(str);
}

template<typename...Args>
std::string stringify(const Indices<Args...>&)
{
    std::array<std::string,sizeof...(Args)> buffer{stringify_impl(Args())...};
    std::string rv;
    for(size_t i=0;i<buffer.size()-1;++i)
        rv+=buffer[i]+",";
    rv+=buffer[buffer.size()-1];
    return rv;
}

template<size_t R>
struct GenericIndexBase;

template<size_t R>
struct GenericIndexBase:public GenericIndexBase<R-1> {
    using base_t=GenericIndexBase<R-1>;
    static constexpr char value=static_cast<char>(R);
    using my_idx=C_String<value,'\0'>;
    using type=typename MakeUnion<my_idx,typename base_t::type>::type;
};

template<>
struct GenericIndexBase<0>{
    static constexpr char value=static_cast<char>(0);
    using type=make_indices<C_String<value,'\0'>>;
};

template<size_t R>
struct GenericIndex{
    using type=typename GenericIndexBase<R-1>::type;
};

template<>
struct GenericIndex<0>{
    using type=Indices<>;
};

}}//End namespaces
