#pragma once

#include<vector>
#include<array>
#include<stdexcept>

namespace TensorWrapper {

/**
 * @brief Describes a tensor space.
 *
 * Perhaps the best definition of a tensor is an element of a tensor-product
 * space (which itself is just the tensor product of many vector spaces).  This
 * class describes the space that the tensor is an element of.  In particular it
 * is concerned with:
 *
 * - The number of basis vectors spanning each mode.
 * - Symmetry relations among the modes
 *
 * It is decoupled from other considerations such as:
 * - Subspaces
 *   - Handled by derived classes
 * - How and where the elements of the tensor representation are stored
 *   - Handled by the Layout and Allocator classes
 * - Sparsity
 *   - Handled by SparseSubspace
 *
 */
class Space {
    public:
    ///The type of object used for indexing into the space
    using size_type = std::size_t;
    ///The type of a range (low index, high index) array
    using range_type = std::array<size_type, 2>;

    /**
     * @brief Makes the space spanned by the zero vector.
     *
     * Strictly speaking a vector space can not be empty as all vector spaces
     * must include the zero vector.  This constructor builds the zero vector
     * space.
     *
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
     * @brief Creates a tensor space given the lengths of each mode.
     *
     * @tparam container_type The type of the input set of dimensions.  It
     * must have cbegin() and cend().
     * @param lengths An @f$o@f$ element container where the @f$i@f$-th element
     * is the length of the @f$i@f$-th mode of the space.
     * @throw std::bad_alloc if there is insufficient memory to create the
     * vector.  Strong throw guarantee.
     */
    template<typename container_type>
    Space(const container_type& lengths):
      lengths_(lengths.cbegin(), lengths.cend()) {}

    /**
     * @brief Cleans up any memory held by the current instance.
     * @throw None. No throw guarantee.
     *
     */
    virtual ~Space()noexcept = default;

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
     * @brief Returns the number of modes in an element of this space.
     *
     * Per the usual statements an order of 0 indicates that elements of the
     * space are scalars, an order of 1 means they are vectors, 2 matrices, etc.
     *
     * @return The number of spaces that were combined to form the
     * tensor-product space.
     * @throw None. No throw guarantee.
     */
    size_type order() const noexcept { return lengths_.size(); }

    /**
     * @brief Returns the number of components an element of this space has.
     *
     * Generally speaking the number of components in a tensor is the product of
     * the lengths of each mode.  The gotcha on this is for spaces containing
     * scalars in which there is still one element.
     *
     * @return The number of elements in a tensor from this space.
     * @throw No throw guarantee.
     */
    size_type size() const noexcept;

    /**
     * @brief Returns the length of a particular mode of the tensor.
     * @param i The mode whose length is desired.  Should be in the range [0,
     * order) although no check is performed to ensure that it is and access
     * outside this range is undefined behavior.
     * @return The length of the requested mode.
     * @throw None. No throw guarantee.
     */
    size_type length(size_type i) const noexcept { return lengths_[i]; }

    /**
     * @brief Returns true if the requested index is part of this space.
     *
     * Given an index @f$I@f$, of length @f$n@f$ whose @f$i@f$-th value is
     * @f$I_i@f$, this function will check if
     * @f$n_i^0\le I_i \lt n_i \forall i \in [0,n)@f$.  For the case when
     * @f$n@f$ is equal to @f$o@f$ this is self-explanatory.  For cases when
     * @f$n>o@f$ this function will return false and for cases where @f$n<$@f$
     * this function will compare the first against the first @f$n@f$ indices
     * of the space (equivalent to asking if a subspace is present).
     *
     * @tparam container_type The type of @p idx.  Must satisfy the concept
     *         of random-access container.
     * @param idx The index to look for.
     * @return True if the index is located within the current space and false
     * otherwise.
     * @throw None. No throw guarantee.
     */
    template<typename container_type>
    bool count(const container_type& idx) const noexcept {
        return count_(std::vector<size_type>(idx.cbegin(), idx.cend()));
    }

    /**
     * @brief Swaps the modes of a tensor around.
     *
     * Usage of this function is best explained by an example.  Assume you
     * have an order 3 tensor called A, and you provided the input:
     *
     * from: {1, 2, 0}
     * to: {0, 1, 2}
     *
     * This function will create a new tensor B whose first mode is A's second
     * mode, B's second mode will be A's third mode, and B's third mode will be
     * A's first mode.
     *
     * @tparam from_type The type of the container listing the original indices.
     *         Must satisfy the concept sequence container.
     * @tparam to_type The type of the container holding the new indices. Must
     *         satisfy the concept sequence container.
     * @param from Up to order() indices in the range [0,order()) such that the
     *        @f$i@f$-th value is the number of the original mode that will
     *        be the to[@f$i@f$]-th mode of the new tensor.
     * @param to Up to order() indices in the range [0, order()) such that the
     *        @f$i@f$-th value is the new index of the original from[@f$i@f$]-th
     *        mode.
     * @return The current Space with the modes in the new order.
     * @throw std::bad_alloc if there is insufficient memory to copy the
     *        lengths.  Strong throw guarantee.
     *
     */
    template<typename from_type, typename to_type>
    Space& shuffle(const from_type& from, const to_type& to)  {
        return shuffle_(std::vector<size_type>(from.cbegin(), from.cend()),
                        std::vector<size_type>(to.cbegin(), to.cend()));
    }

    /**
     * @brief Compares two spaces for exact equality.
     *
     * Two spaces are equal if they are spanned by the same set of vectors.
     * Since we don't actually know the vectors, we assume that the user isn't
     * comparing apples and oranges and instead define equality as all of the
     * indices in the current instance are also found in @p rhs.
     *
     * @param rhs The space to compare against.
     * @return true if this space is exactly equal to @p rhs and false
     * otherwise.
     * @throw None. No throw guarantee.
     */
    bool operator==(const Space& rhs) const noexcept;

    /**
     * @brief Determines if two spaces are different.
     *
     * See operator== for the definition of equality.  This function simply
     * negates it.
     *
     * @param rhs The space to compare against.
     * @return True if there is at least one index found in the current instance
     * that is not found in @p rhs or vice versa.
     * @throw None. No throw guarantee.
     */
    bool operator!=(const Space& rhs) const noexcept { return !(*this == rhs); }

    /**
     * @brief Determines if the current space is a proper subspace of another.
     *
     * Strictly speaking this function only checks if the basis vectors spanning
     * this space are a proper subset of the basis vectors spanning @p rhs.
     * Like equality, we do not compare the actual basis vectors, but rather
     * compare the indices.  With such a comparison the current instance is a
     * proper subspace of @p rhs if and only if every index found in the current
     * index is also found in @p rhs and there exists at least one index found
     * within @p rhs that is not within this instance.
     *
     * It may not be obvious, but a subspace need not have the same number of
     * modes as its superspace.  An obvious example is the empty space, which is
     * a subspace of all spaces aside from itself.  More generally say our
     * current instance has @f$o@f$ modes and @p rhs has @f$n@f$ modes, we
     * need to know the mapping between our modes and those of @p rhs and
     * within a mode we need to know the mapping from our basis vectors to
     * those of @p rhs.  Presently we assume both us and @p rhs use the same
     * ordering of basis vectors for all common modes and furthermore we
     * assume that the @f$o@f$ modes to compare are the first @f$o@f$ modes
     * of @p rhs and appear in the same order.  What this all boils down to
     * is that every index in the current space must be valid as the first
     * @f$o@f$ digits of an index in @p rhs.
     *
     *
     * @param rhs The space to compare against.
     * @return true if this is a proper subspace of @p rhs and false otherwise.
     * @throw None. No throw guarantee.
     */
    bool operator<(const Space& rhs) const noexcept;

    /**
     * @brief Determines if the current space is a subspace.
     *
     * This is a convenience function to check for proper subspace or equality.
     * See operator< and operator== for the respective definitions.
     *
     * @param rhs The space to compare against.
     * @return True if every index in the current subspace is contained
     * within @p rhs and false otherwise.
     * @throw None. No throw guarantee.
     */
    bool operator<=(const Space& rhs) const noexcept {
        return *this < rhs || *this == rhs;
    }

    /**
     * @brief Determines if the current space is a proper superspace.
     *
     * The current instance is a proper superspace of @p rhs if and only if
     * @p rhs is a proper subspace of the current instance.  Hence this function
     * is simply a wrapper around rhs<*this.
     *
     * @param rhs The space to compare against.
     * @return True if every index in @p rhs is contained within the current
     * instance and there exists at least one additional index within the
     * current instance.  False otherwise.
     * @throw None. No throw guarantee.
     */
    bool operator>(const Space& rhs) const noexcept { return rhs < *this; }

    /**
     * @brief Determines if the current space is a superspace.
     *
     * This is a convenience function wrapped around rhs<=*this.
     *
     * @param rhs The space to compare against.
     * @return True if @p rhs is a subspace of the current instance and false
     * otherwise.
     * @throw None. No throw guarantee.
     */
    bool operator>=(const Space& rhs) const noexcept { return rhs <= *this; }

    protected:

    ///The method actually implementing the look-up of an element
    virtual bool count_(const std::vector <size_type>& idx) const noexcept;

    ///The method actually responsible for shuffling the modes
    virtual Space& shuffle_(const std::vector<size_type>& from,
                            const std::vector<size_type>& to);
    private:
    ///The shape of this instance
    std::vector <size_type> lengths_;
};

//Implementations for templated functions



//template<typename from_type, typename to_type>
//Space& Space::swap_modes(const from_type& from, const to_type& to) {
//    if(from.size() != to.size())
//        throw std::invalid_arguement("parameter from and parameter to must "
//                                     "have the same size.");
//    for(size_type i=0; i<from.size(); ++i)
//        lengths_[from[i]].swap(lengths_[to[i]]);
//
//};

} // End namespace


