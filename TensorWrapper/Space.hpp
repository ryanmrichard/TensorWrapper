#pragma once
#include<vector>
#include<array>

namespace TensorWrapper {

/**
 * @brief Describes a tensor space.
 *
 * Let @f$\mathbb{T}@f$ be a tensor-product space given by the tensor product of
 * @f$o@f$ vector spaces, the @f$i@f$-th of which is @f$\mathbb{S_i}@f$.  The
 * elements of @f$\mathbb{T}@f$ are order @f$o@f$ tensors containing
 * @f$\prod_{i=0}^{o}l_i@f$ elements, where @f$l_i@f$ is the number of vectors
 * required to span @f$\mathbb{S_i}@f$.  The purpose of this class is to provide
 * information related to the structure of @f$\mathbb{T}@f$ and of the
 * @f$\mathbb{S_i}@f$s comprising it (rigorously speaking each
 * @f$\mathbb{S_i}@f$ is also a tensor-product space which is why we use the
 * same class to model two seemingly unrelated concepts).
 *
 * In designing this class its main purposes are envisioned as:
 * 1. Storing the length of each mode (@f$\mathbb{S_i}@f$ being the @f$i@f$-th
 *    mode).
 * 2. Storing symmetry relations among the modes (namely does interchange of
 *    two modes change the value of the element or not)
 * 3. Storing subspaces (more rigorously subsets as we do not actually
 *    require the resulting subsets to form a space) of our current space
 *    a. One important set of "subspaces" are the partitions arising from
 *       cyclically choosing vectors
 * 4. Providing user-defined aliases of the various subspaces (think using
 *    the indices "i,j,k" to indicate that we want the first subspace and
 *    "a,b,c" for the second
 */
class Space {
public:
    ///The type of object used for indexing into the space
    using size_type = std::size_t;
    ///The type of a range (low index, high index, stride)
    using range_type = std::array<size_type, 3>;

    /**
     * @brief
     * @throw None. No throw guarantee.
     */
    Space()noexcept = default;

    /**
     * @brief Makes a deep copy of another instance.
     * @param[in] rhs The instance to copy.
     * @throw std::bad_alloc if there is insufficient memory for the copy. Strong
     *        throw guarantee.
     */
    Space(const Space& /*rhs*/) = default;

    /**
     * @brief Takes ownership of another instance.
     * @param[in] rhs The instance to take ownership of.  After this call @p rhs
     *            is in a valid, but otherwise undefined state.
     * @throw None. No throw guarantee.
     */
    Space(Space&& /*rhs*/)noexcept = default;

    /**
     * @brief Cleans up any memory held by the current instance.
     * @throw None. No throw guarantee.
     *
     */
    ~Space()noexcept = default;

    /**
     * @brief Assigns a deep copy of another instance's state to the current
     *        instance.
     * @param[in] rhs The instance to copy.
     * @return The current instance containing a deep-copy of @p rhs's state.
     * @throw std::bad_alloc if there is insufficient memory for the copy. Strong
     *        throw guarantee.
     */
    Space& operator=(const Space& /*rhs*/)= default;

    /**
     * @brief Takes ownership of another instance.
     * @param[in] rhs The instance to take ownership of.  After this call @p rhs
     *            is in a valid, but otherwise undefined state.
     * @return The current instance containing @p rhs's state.
     * @throw None. No throw guarantee.
     */
    Space& operator=(Space&& /*rhs*/)noexcept= default;

    /**
     * @brief Creates a "full" space (rigorously speaking we only know we have
     * the first element).
     *
     * This constructor is used to form a full space.  More specifically it
     * assumes that this space contains all vectors @f$[0, l_i)@f$ where
     * @f$l_i@f$ is the number of vectors spanning the @f$i@f$-th mode.
     * Consequentially the order @f$o@f$ tensors in the resulting space will
     * contain @f$\prod_{i=1}^ol_i@f$ elements.
     *
     * @tparam container_type The type of the input set of dimensions.  It
     * should be a random-access container.
     * @param lengths A container such that the @f$i@f$-th element is the
     * dimensionality of the @f$i@f$-th space in the tensor product.  The
     * length of @p lengths should be equal to the order of the tensors living
     * in the resulting space.
     * @throw std::bad_alloc if there is insufficient memory to create the
     * vectors.  Strong throw guarantee.
     */
    template <typename container_type>
    Space(const container_type& lengths):
        Space(std::vector<size_type>(lengths.size()),lengths)
    {}

    /**
     * @brief Creates a space containing only a subset of a tensor-product
     * space.
     *
     * Compared to the single container constructor, this constructor allows one
     * to form Space instances containing only a subset of all the possible
     * vectors from each mode.  Specifically, for the @f$i@f$-th mode the
     * resulting Space will be populated only by the vectors @f$[n_0^i,n_i)@f$ where
     * @f$n_0^i@f$ is the index of the first vector of the @f$i@f$-th mode to
     * include and @f$n_i@f$ is the first vector to exclude.
     *
     * @tparam low_container_type The type of the container storing the first
     * vector of each mode.  Must be random-access.
     * @tparam high_container_type The type of the container storing the first
     * vector to exclude for each mode.  Must be random-access.
     * @param lows A list where the @f$i@f$-th element is the first vector to
     *        include for the @f$i@f$-th mode, @f$i\in[0,l_i)@f$.
     * @param highs A list where the @f$i@f$-th element is the first vector to
     *        exclude for the @f$i@f$-th mode, @f$i\in[0,l_i]@f$.
     */
    template<typename low_container_type, typename high_container_type>
    Space(const low_container_type& lows, const high_container_type& highs):
            Space(lows, highs, std::vector<size_type>(lows.size(),1))
    {}

    /**
     * @brief Returns the number of modes in the tensor.
     *
     * @return The number of spaces that were combined to form the
     * tensor-product space.
     * @throw None. No throw guarantee.
     */
    size_type order()const noexcept{return lengths_.size();}

    /**
     * @brief Returns the number of vectors required to span the space.
     *
     * The number of basis vectors required to span the space is given by:
     * @f[
     *  \frac{n_i-n_i^0}{p_i}
     * @f]
     * where @f$p_i@f$ is the stride between elements in the range
     * @f$[n_i^0, n_i)@f$.
     *
     * @return The number of elements in a tensor from this space.
     * @throw No throw guarantee.
     */
    size_type size()const noexcept ;

    /**
     * @brief Returns the number of subspaces contained within this space.
     * @return The number of subspaces within this space
     * @throw None. No throw guarantee.
     */
    size_type nsubspaces()const noexcept {return subspaces_.size();}

    /**
     * @brief Returns true if the requested index is part of this space.
     *
     * Given an index @f$I@f$, of length @f$n@f$ whose @f$i@f$-th value is
     * @f$I_i@f$, this function will check if
     * @f$n_i^0\le I_i \lt n_i \forall i \in [0,n)@f$.  For the case when
     * @f$n@f$ is equal to @f$o@f$ this is self-explanatory.  For cases when
     * @f$n\ne o@f$ this function will return false.
     *
     * @tparam container_type The type of @p idx.  Must satisfy the concept
     *         of random-access container.
     * @param idx The index to look for.
     * @return True if the index is located within the current space and false
     * otherwise.
     * @throw None. No throw guarantee.
     */
    template<typename container_type>
    bool count(const container_type& idx)const noexcept;

    /**
     * @brief Returns true if this is the empty space.
     * @return True if this is the empty vector space and false otherwise.
     * @throw None. No throw guarantee.
     */
    bool empty()const noexcept{return !subspaces_.size();}

    bool operator==(const Space& rhs)const noexcept;

    bool operator!=(const Space& rhs)const noexcept{return !(*this==rhs);}

    bool operator<(const Space& rhs)const noexcept;

    bool operator<=(const Space& rhs)const noexcept{
        return *this<rhs || *this==rhs;
    }

    bool operator>(const Space& rhs)const noexcept{return rhs<*this;}

    bool operator>=(const Space& rhs)const noexcept{return rhs<=*this;}

private:
    ///The spaces contained within this space
    std::vector <Space> subspaces_;

    ///The shape of this instance
    std::vector <range_type> lengths_;

    /**
     * @brief Creates a space containing only a subset of a tensor-product
     * space and the spacing between those elements is parameterized
     *
     * Compared to the two container constructor this constructor also allows
     * you to specify the stride between elements.  For the moment, it's not
     * publicaly exposed as it's unclear that an average user would ever want to
     * set this.
     *
     * @tparam low_container_type The type of the container storing the first
     * vector of each mode.  Must be random-access.
     * @tparam high_container_type The type of the container storing the first
     * vector to exclude for each mode.  Must be random-access.
     * @tparam stride_container_type The type of the container storing the
     * stride between consecutive elements of each mode.  Must be random-access.
     * @param lows A list where the @f$i@f$-th element is the first vector to
     *        include for the @f$i@f$-th mode, @f$i\in[0,l_i)@f$.
     * @param highs A list where the @f$i@f$-th element is the first vector to
     *        exclude for the @f$i@f$-th mode, @f$i\in[0,l_i]@f$.
     * @param strides A list where the @f$i@f$-th element is the the stride
     *        between elements of the @f$i@f$-th mode.
     */
    template<typename low_container_type, typename high_container_type,
             typename stride_container_type>
    Space(const low_container_type& lows, const high_container_type& highs,
          const stride_container_type& strides);
};

//Implementations for templated functions

template<typename container_type>
bool Space::count(const container_type& idx)const noexcept{
    if(idx.size() != order()) return false;
    for(size_type i=0; i < order(); ++i)
        if(lengths_[i][0] > idx[i] || lengths_[i][1] <= idx[i] )
            return false;
    return true;
}

template<typename low_container_type, typename high_container_type,
         typename stride_container_type>
Space::Space(const low_container_type& lows, const high_container_type& highs,
      const stride_container_type& strides)
{
    for(size_type i = 0; i < lows.size(); ++i)
        lengths_.push_back(range_type{lows[i], highs[i], strides[i]});
    subspaces_.push_back(*this);
}


} // End namespace


