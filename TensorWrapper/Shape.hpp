#pragma once
#include<array>
#include<numeric>
#include<tuple>

namespace TWrapper {

/** \brief A class to describe the shape of a tensor.
 *
 *  For the moment we assume the underlying memory is not block cyclic, but
 *  either row major or column major.  This class hides the iteration details
 *  over either one of those two layouts for us.  It also faciliates iterating
 *  over a subtensors of a tensor.  This is where things get tricky.  When
 *  considering sub-tensors there are two sorts of indices: relative and
 *  absolute.  Relative indices describe the index
 *  within the sub-tensor, whereas absolute describes the indices within the
 *  full tensor.  Which set of indices a shape instance iterates over is
 *  context dependent and is up to the user of this class.
 *
 *  \tparam rank The rank of the tensor we are describing.
 */
template<size_t rank>
class Shape{
    ///The dimensions of the tensor
    std::array<size_t,rank> dims_;

    ///Is the tensor laid out in RowMajor format?
    bool RowMajor_;

    ///Returns the product of dimensions in the range [i,j)
    size_t product_dims(size_t i, size_t j)const
    {
        return std::accumulate(dims_.data()+i,
                               dims_.data()+j,1,std::multiplies<size_t>());
    }

    ///A class for iterating over the indices in the order they are laide out in
    ///memory
    class ShapeItr{
        ///Type of the index
        using index_t=std::array<size_t,rank>;

        ///Who made me
        const Shape& parent_;

        ///The current index
        index_t idx_;

        /** \brief Increments the index stored in the iterator
         *
         *  This is really the guts of the ShapeItr class.  At the end of the
         *  day, no matter the rank, the tensor is laid out as one long chunk of
         *  memory.  This memory is intrinsicly rank 1 and thus higher order
         *  tensors must make a convention for how their rows and columns are
         *  "flattened".  For a tensor of say rank 3, whose three
         *  dimensions are of lengths "m","n", and "k" respectively if the
         *  indices are laid out in row major format they appear in the order:
           \verbatim
           {0,0,0},{0,0,1},...,{0,0,k},{0,1,0},...{0,n,k},{1,0,0},...{m,n,k}
           \endverbatim
         *  On the other hand, if the tensor is laid out in column major format
         *  then the indices are in the order:
           \verbatim
           {0,0,0},{1,0,0},...,{m,0,0},{0,1,0},...{m,n,0},{0,0,1},...{m,n,k}
           \endverbatim
         * This means for row(column) major we need to find the first index
         * starting from the right(left) which can be incremented.  Say we
         * determine that said index is index "i", then we need to reset all
         * indices appearing to right(left) of index "i"  before returning.
         *
         * \throws No throw guarantee.
         */
        void next() noexcept
        {
            const auto& end=parent_.dims_;
            if(parent_.RowMajor_)
            {
                for(size_t i=rank;i>0;--i)
                    if(++idx_[i-1]<end[i-1])
                    {
                        std::fill(idx_.data()+i,idx_.data()+rank,0);
                        return;
                    }
            }
            else
            {
                for(size_t i=0;i<rank;++i)
                    if(++idx_[i]<end[i])
                    {
                        std::fill(idx_.data(),idx_.data()+i,0);
                        return;
                    }
            }
        }
    public:
        /** \brief Makes a new shape iterator that is tied to \p parent
         *
         *  This constructor is capable of making an iterator that points to
         *  either the first element or just past the last element.
         *
         *  \param[in] parent The Shape instance that this iterator is iterating
         *                    over.
         *  \param[in] begin Is this iterator pointing to the first or just past
         *                   the last element?

         *
         */
        ShapeItr(const Shape& parent,bool begin):
            parent_(parent),
            idx_(begin?index_t{}:parent.dims_)
        {}

        /** \brief Checks if two ShapeItrs point to the same index
         *
         *   This operator compares only the index of the two iterators.  It
         *   does not ensure they stem from the same shape instance, nor does it
         *   ensure they iterate in the same order (row major/column major), nor
         *   if the iteration ranges are equivalent.
         *
         *   \param[in] other The ShapeItr instance to compare to.
         *
         *   \return True if the iterators point to the same index
         *
         *   \throw Never throws
         */
        bool operator==(const ShapeItr& other)const noexcept
        {
            return idx_==other.idx_;
        }

        ///Same as operator==, except returns true if iterators are not equal
        bool operator!=(const ShapeItr& other)const noexcept
        {
            return !((*this)==other);
        }

        /** \brief Compares the indices contained within two ShapeItr instances
         *         lexographically.
         *
         * \param[in] other The ShapeItr instance to compare to.
         *
         * \return True if the index in this instance is less than that in
         *              \p other in a lexographic sense.
         *
         * \throws No throw guarantee.
         */
        bool operator<(const ShapeItr& other)const noexcept
        {
            return idx_<other.idx_;
        }

        ///Same as operator<, except also true if the iterators are equal
        bool operator<=(const ShapeItr& other)const noexcept
        {
            return (*this)<other || (*this)==other;
        }

        ///Same as operator<=, except return values are negated
        bool operator>(const ShapeItr& other)const noexcept
        {
            return !((*this)<=other);
        }

        ///Same as operator<, except return values are negated
        bool operator>=(const ShapeItr& other)const noexcept
        {
            return !((*this)<other);
        }

        ///Returns the current index
        const index_t& operator*()const noexcept{
            return idx_;
        }

        ///Increments the current index, returning the resulting instance
        ShapeItr& operator++()noexcept{
            next();
            return *this;
        }

        ///Increments the current index returning a copy of the instance pre incrment
        ShapeItr operator++(int)noexcept{
            ShapeItr rv(*this);
            ++(*this);
            return rv;
        }
    };

public:

    /** \brief Constructs a class that describes the layout of a tensor.
     *
     *  \param[in] dims The length of each of the "rank" dimensions of the
     *                  tensor described by this class.
     *  \param[in] RowMajor Is the memory of the described tensor laid out in
     *                      row major format?
     *
     */
    explicit Shape(const std::array<size_t,rank>& dims,
                   bool RowMajor=true):
        dims_(dims),
        RowMajor_(RowMajor){}

    ///Returns the total number of elements in this block
    const size_t size()const noexcept{return product_dims(0,rank);}

    /** \brief Returns the length of each dimension of this block
     *
     *  \warning This is not the end point of the block unless the origin vector
     *           is the zero vector.  To obtain the end point add origin to the
     *           return of this function
     *
     *  \return The number of elements this shape describes along each
     *          dimension.
     *
     *  \throws No throw guarantee.
     */
    const std::array<size_t,rank>& dims()const noexcept{return dims_;}

    ///The type of an iterator over this shape
    using const_iterator=ShapeItr;

    ///Returns an interator to the first index of the block
    const_iterator begin()const{
        return ShapeItr(*this,true);
    }

    ///Returns an iterator just past the last offset of the block
    const_iterator end()const{
        return ShapeItr(*this,false);
    }

    ///Checks if two shapes of the same rank are equal
    bool operator==(const Shape<rank>& other)const noexcept
    {
        return std::tie(RowMajor_,dims_)==
               std::tie(other.RowMajor_,other.dims_);
    }

    ///General check, will be false b/c same ranks resolve to the above fxn
    template<size_t rank_other>
    bool operator==(const Shape<rank_other>&)const noexcept
    {
        return false;
    }

    ///Same as operator==, except it negates the result
    template<size_t rank_other>
    bool operator!=(const Shape<rank_other>& other)const noexcept
    {
        return !(*this==other);
    }

    ///Returns the pointer offset of an index
    template<typename T>
    size_t flat_index(const T& idx)const
    {
        size_t result=0;
        for(size_t i=0;i<rank;++i)
            result+=idx[i]*product_dims(RowMajor_? i+1: 0,RowMajor_? rank: i);
        return result;

    }

};
}
