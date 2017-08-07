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


template<size_t...hvs>
struct Indices
{
    constexpr static size_t size=sizeof...(hvs);

    template<size_t I>
    constexpr static size_t get()
    {
        return std::get<I>(std::make_tuple(hvs...));
    }

};

template<typename LHS_t, typename RHS_t>
struct IndexUnion;

template<typename LHS_t, typename...Args>
struct IndexUnion<LHS_t,Indices<Args...>>
{
    using type=Indices<LHS_t,Args...>;
};

template<size_t LDepth, size_t RDepth, typename LHS_t, typename RHS_t>
struct IndexPairTraits;

//Recursion end point
template<size_t...LArgs, size_t...RArgs>
struct IndexPair<sizeof...(LArgs),0,
                 Indices<LArgs...>,
                 Indices<RArgs...>> {
  using free_indices=Indices<>;
  constexpr static size_t nfree=0;
  constexpr static size_t ndummy=0;

};

//End inner loop
template<size_t L, size_t...LArgs, size_t...RArgs>
struct IndexPair<L,sizeof...(RArgs),Indices<LArgs...>,Indices<RArgs...>>: public
         IndexPair<L+1,0,Indices<LArgs...>,Indices<RArgs...>>
{
    using base=IndexPair<L+1,0,Indices<LArgs...>,Indices<RArgs...>>;
    using free_indices=typename base::free_indices;
    constexpr static size_t nfree=base::nfree;
    constexpr static size_t ndummy=base::ndummy;
};


}}
