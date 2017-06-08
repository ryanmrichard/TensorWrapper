#pragma once
#include <array>
#include "TensorWrapper/TensorTypes.hpp"

namespace TWrapper {

template<size_t rank, typename T, typename Tensor_t>
class TensorWrapper{
private:

    ///The rank of the tensor
    const size_t rank_=rank;

    ///The wrapped tensor
    Tensor_t tensor_;

    ///The thing that will give us information
    detail_::TensorWrapperImpl<rank,T,Tensor_t> impl_;

public:
    ///The type of an instance of this class
    using my_t=TensorWrapper<rank,T,Tensor_t>;
    ///The type wrapped by this class
    using wrapped_t=Tensor_t;

    ///Copies, and evaluates expression templates, of other
    ///(deep vs. shallow depends on other's copy semantics)
    template<typename RHS_t>
    TensorWrapper(const RHS_t& other):
        rank_(rank),tensor_(other)
    {}

    template<typename...Args>
    TensorWrapper(const Contraction<Args...>& rhs)
    {
        tensor_=impl_.contract(rhs);
    }

    TensorWrapper():rank_(rank){}
    TensorWrapper(my_t&&)=default;
    my_t& operator=(const my_t&)=default;
    my_t& operator=(my_t&&)=default;


    ///Returns the tensor instance inside this wrapper
    const Tensor_t& tensor()const{return tensor_;}

    ///Sets this equal to rhs
    template<typename RHS_t>
    my_t& operator=(const RHS_t& rhs)
    {
        tensor_=rhs;
        return *this;
    }


    template<typename...Args>
    my_t& operator=(const Contraction<Args...>& rhs)
    {
        tensor_=impl_.contract(rhs);
        return *this;
    }


    ///Returns the shape of the tensor
    Shape<rank> dims()const{
        return impl_.dims(tensor_);
    }

    ///Returns an element of a tensor
    T operator()(std::array<size_t,rank>& idx)const
    {
        return impl_.get_value(tensor_,idx);
    }

    ///API for contraction
    template<size_t N> constexpr
    IndexedTensor<rank,Tensor_t> operator()(const char(&idx)[N])const
    {
        return IndexedTensor<rank,Tensor_t>(tensor_,idx);
    }

    ///Returns true if all elements of two tensors are equal
    template<typename RHS_t>
    bool operator==(const RHS_t& other)const;

    ///Returns true if any element of two tensors differs
    template<typename RHS_t>
    bool operator!=(const RHS_t& other)const;

    ///Scales this tensor by rhs
    decltype(auto) operator*(double rhs)const
    {
        return impl_.scale(tensor_,rhs);
    }

    ///Adds rhs
    template<typename RHS_t>
    decltype(auto) operator+(const RHS_t& rhs)const
    {
        return impl_.add(tensor_,rhs);
    }

    ///Subtracts rhs
    template<typename RHS_t>
    decltype(auto) operator-(const RHS_t& rhs)const
    {
        return impl_.subtract(tensor_,rhs);
    }

};

}
//All the operator overload definitions
#include "TensorWrapper/TWOperators.hpp"


