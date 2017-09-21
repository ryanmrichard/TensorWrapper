#pragma once
#include <array>
#include <algorithm> //For copy
#include <functional> //For function

namespace TWrapper {
namespace detail_ {

/** \brief Implements the default mechanism for obtaining the next index.
 *
 *
 */
template<size_t rank>
void next(std::array<size_t,rank>& idx,
          const std::array<size_t,rank>& first,
          const std::array<size_t,rank>& last,
          bool row_major) noexcept
{
    for(size_t i=0;i<rank;++i)
    {
        const size_t ii=(row_major?rank-1-i:i);
        if(++idx[ii] < last[ii])
        {
            const size_t begin_offset=(row_major ? ii+1 : 0);
            const size_t end_offset=(row_major ? rank : ii);
            std::copy(first.data()+begin_offset,
                      first.data()+end_offset,
                      idx.data()+begin_offset);
            break;
        }
    }
}

}// End namespace detail_

/** \brief A class that generates the indices in the order they are laid out in
 *   memory.
 *
 *
 *  \tparam rank The rank of the tensor we are iterating over
 *
 */
template<size_t rank>
class IndexItr{
public:
    using value_type=std::array<size_t,rank>;

    using reference=value_type&;

    using const_reference=const value_type&;

    using pointer=value_type*;

    using const_pointer=const value_type*;

    using incrementor_type=
        std::function<void(reference,const_reference,const_reference,bool)>;

    /** \brief Makes a new iterator
     *
     *  This constructor is capable of making an iterator that points to
     *  either the first element or just past the last element.
     *
     *  \param[in] last The last element in the range
     *  \param[in] begin Is this iterator pointing to the first or just past
     *                   the last element? Default is the first
     *  \param[in] row_major True if the data is row major. Default is true.
     *  \param[in] first The first element in the range.  Default is the zero
     *                   vector.
     *  \param[in] fxn The function to call to increment the index
     *
     *  \throws None  No throw guarantee.
     */
    IndexItr(const_reference last,
             bool begin=true,
             bool row_major=true,
             const_reference first=value_type({}),
             incrementor_type fxn=detail_::next<rank>)noexcept:
        first_(first),last_(last),idx_(begin?first:last),row_major_(row_major),
        next_(fxn)
    {}

    /** \brief Checks if two IndexItrs point to the same index
     *
     *   This operator compares only the index of the two iterators.  It
     *   does not ensure they stem from the same shape instance, nor does it
     *   ensure they iterate in the same order (row major/column major), nor
     *   if the iteration ranges are equivalent.
     *
     *   \param[in] other The IndexItr instance to compare to.
     *
     *   \return True if the iterators point to the same index
     *
     *   \throw None.  No throw guarantee.
     */
    bool operator==(const IndexItr& other)const noexcept
    {
        return idx_==other.idx_;
    }

    ///Same as operator==, except returns true if iterators are not equal
    bool operator!=(const IndexItr& other)const noexcept
    {
        return !((*this)==other);
    }

    /** \brief Compares the indices contained within two IndexItr instances
     *         lexographically.
     *
     * \param[in] other The IndexItr instance to compare to.
     *
     * \return True if the index in this instance is less than that in
     *              \p other in a lexographic sense.
     *
     * \throws No throw guarantee.
     */
    bool operator<(const IndexItr& other)const noexcept
    {
        return idx_<other.idx_;
    }

    ///Same as operator<, except also true if the iterators are equal
    bool operator<=(const IndexItr& other)const noexcept
    {
        return (*this)<other || (*this)==other;
    }

    ///Same as operator<=, except return values are negated
    bool operator>(const IndexItr& other)const noexcept
    {
        return !((*this)<=other);
    }

    ///Same as operator<, except return values are negated
    bool operator>=(const IndexItr& other)const noexcept
    {
        return !((*this)<other);
    }

    ///Returns the current index
    const_reference operator*()const noexcept{
        return idx_;
    }

    const_pointer operator->()const noexcept{
        return &idx_;
    }

    ///Increments the current index, returning the resulting instance
    IndexItr& operator++()noexcept{
        next_(idx_,first_,last_,row_major_);
        return *this;
    }

    ///Increments the current index returning a copy of the instance pre incrment
    IndexItr operator++(int)noexcept{
        IndexItr rv(*this);
        ++(*this);
        return rv;
    }

private:
    ///The first index
    value_type first_;

    ///The max index
    value_type last_;

    ///The current index
    value_type idx_;

    ///Is row_major?
    bool row_major_;

    ///Functor providing next functionality
    incrementor_type next_;
};

#ifdef BUILD_TWRAPPER_LIBRARY
extern template class IndexItr<0>;
extern template class IndexItr<1>;
extern template class IndexItr<2>;
extern template class IndexItr<3>;
extern template class IndexItr<4>;
#endif

}//End namespace
