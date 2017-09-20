#pragma once
#include "TensorWrapper/IndexItr.hpp"
#include<array>
#include<tuple> //For tie
#include<functional> //For minus, multiplies
#include<algorithm> //For transform,accumulate

namespace TWrapper {

/** \brief A class to describe the shape of a block of a tensor.
 *
 *
 *  \tparam rank The rank of the tensor we are describing.
 */
template<size_t rank>
class Shape{
public:
    using index_type=std::array<size_t,rank>;

    using const_iterator=IndexItr<rank>;

    /** \brief Constructs a class that describes the layout of a tensor.
     *
     *  \param[in] end The last element in the this block of the tensor
     *  \param[in] RowMajor Is the memory of the described tensor laid out in
     *                      row major format? Default: true.
     *  \param[in] start The first element of the tensor.  Default: the "rank"-
     *                   element zero vector.
     *
     */
    explicit Shape(const index_type& end,
                   bool row_major=true,
                   const index_type& start=index_type{}):
        Shape(end,row_major,start,
              const_iterator(end,true,row_major,start),
              const_iterator(end,false,row_major,start))
    {}

    explicit Shape(const index_type &end,
                   bool row_major,
                   const index_type &start,
                   const IndexItr<rank>& begin_itr,
                   const IndexItr<rank>& end_itr):
        first_(start),last_(end),dims_(),row_major_(row_major),
        begin_(begin_itr),end_(end_itr)
    {
        std::transform(last_.begin(),last_.end(),
                       first_.begin(),dims_.begin(),
                       std::minus<size_t>());
    }

    ///Returns the total number of elements in this block
    size_t size()const noexcept{return product_dims(0,rank);}

    ///Returns true if the shape is row_major
    bool is_row_major()const noexcept{return row_major_;}

    /** \brief Returns the length of each dimension of this block
     *
     *  \warning This is not the end point of the block unless first vector
     *           is the zero vector.  To obtain the end point add origin to the
     *           return of this function
     *
     *  \return The number of elements this shape describes along each
     *          dimension.
     *
     *  \throws No throw guarantee.
     */
    const index_type& dims()const noexcept{return dims_;}

    ///Returns an interator to the first index of the block
    const_iterator begin()const{return begin_;}

    ///Returns an iterator just past the last offset of the block
    const_iterator end()const{return end_;}

    ///Checks if two shapes of the same rank are equal
    bool operator==(const Shape& other)const noexcept
    {
        return std::tie(row_major_,dims_,begin_,end_)==
               std::tie(other.row_major_,other.dims_,other.begin_,other.end_);
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
        {
            const size_t start_rank=(row_major_ ? i+1 : 0);
            const size_t end_rank=(row_major_ ? rank : i);
            result+=idx[i]*product_dims(start_rank,end_rank);
        }
        return result;

    }

    ///Unflattens an index see @ref md_FlatteningTensors for derivation
    index_type unflatten_index(size_t idx)const
    {
        index_type rv{};
        for(size_t i=0;i<rank;++i)
        {
            const size_t start_rank=(row_major_ ? i+1 : 0);
            const size_t end_rank=(row_major_ ? rank : i);
            const size_t prod=product_dims(start_rank,end_rank);
            rv[i]=((idx-idx%prod)/prod)%dims_[i];
        }
        return rv;
    }

private:
    ///The first index of this block of the tensor
    index_type first_;

    ///The last index of this block of the tensor
    index_type last_;

    ///The dimensions of this block of the tensor
    index_type dims_;

    ///Is the tensor laid out in RowMajor format?
    bool row_major_;

    ///An iterator to the beginning of the block
    const_iterator begin_;

    ///An iterator just past the end of the block
    const_iterator end_;

    ///Returns the product of dimensions in the range [i,j)
    size_t product_dims(size_t i, size_t j)const
    {
        return std::accumulate(dims_.data()+i,
                               dims_.data()+j,1,
                               std::multiplies<size_t>());
    }

};

}//End namespace TWrapper
