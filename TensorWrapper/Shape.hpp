#pragma once
#include<array>
#include<numeric>
#include<tuple>

namespace TWrapper {

template<size_t rank>
class Shape{
    ///The dimensions of the tensor
    std::array<size_t,rank> dims_;

    ///Is the tensor laid out in RowMajor format?
    bool RowMajor_;

    ///Returns the product of dimensions in the range [i,j)
    size_t product_dims(size_t i, size_t j)const
    {
        return std::accumulate(dims_.data()+i,dims_.data()+j,1,std::multiplies<size_t>());
    }

    class ShapeItr{
        using index_t=std::array<size_t,rank>;

        ///Who made me
        const Shape& parent_;

        ///The current index
        index_t idx_;

        ///Increments the index
        void next()
        {
            if(parent_.RowMajor_)
            {
                for(size_t i=rank;i>0;--i)
                    if(++idx_[i-1]<parent_.dims_[i-1])
                    {
                        std::fill(idx_.data()+i,idx_.data()+rank,0.0);
                        return;
                    }
            }
            else
            {
                for(size_t i=0;i<rank;++i)
                    if(++idx_[i]<parent_.dims_[i])
                    {
                        std::fill(idx_.data(),idx_.data()+i,0.0);
                        return;
                    }
            }
        }
    public:
        ShapeItr(const Shape& parent,bool begin):
            parent_(parent),
            idx_(begin?index_t():index_t(parent_.dims_))
        {}

        bool operator==(const ShapeItr& other)const
        {
            return idx_==other.idx_;
        }

        bool operator!=(const ShapeItr& other)const
        {
            return !((*this)==other);
        }

        bool operator<(const ShapeItr& other)const
        {
            return idx_<other.idx_;
        }

        bool operator<=(const ShapeItr& other)const
        {
            return (*this)<other || (*this)==other;
        }

        bool operator>(const ShapeItr& other)const
        {
            return !((*this)<=other);
        }

        bool operator>=(const ShapeItr& other)const
        {
            return !((*this)<other);
        }

        index_t operator*()const{
            return idx_;
        }

        ShapeItr& operator++(){
            next();
            return *this;
        }
    };

public:

    using const_iterator=ShapeItr;

    Shape(const std::array<size_t,rank>& dims,bool RowMajor=true):
        dims_(dims),RowMajor_(RowMajor){}

    const_iterator begin()const{
        return ShapeItr(*this,true);
    }

    const_iterator end()const{
        return ShapeItr(*this,false);
    }

    ///Checks if two shapes of the same rank are equal
    bool operator==(const Shape<rank>& other)const{
        return std::tie(RowMajor_,dims_)==std::tie(other.RowMajor_,other.dims_);
    }

    ///General check, will be false b/c same ranks resolve to the above fxn
    template<size_t rank_other>
    bool operator==(const Shape<rank_other>& other)const
    {
        return false;
    }

    ///Returns true if the two shapes are not equal, negates equivalence check.
    template<size_t rank_other>
    bool operator!=(const Shape<rank_other>& other)const
    {
        return !(*this==other);
    }

    ///Returns the pointer offset for an index
    template<typename T>
    size_t flat_index(const T& idx)
    {
        size_t result=0;
        for(size_t i=0;i<rank;++i)
            result+=idx[i]*product_dims(RowMajor_? i+1: 0,RowMajor_? rank: i);
        return result;

    }

};

}
