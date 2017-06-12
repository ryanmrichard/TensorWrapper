#pragma once
#include "TensorWrapper/TensorWrapperImpl.hpp"
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <map>
namespace TWrapper {
namespace detail_ {

template<typename T>
struct TensorWrapperImpl<2,T,Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>> {
    using wrapped_t=Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
    Shape<2> dims(const wrapped_t& impl)const{
        const size_t r(impl.rows()),c(impl.cols());
        return Shape<2>(std::array<size_t,2>({r,c}),false);
    }

    T get_value(const wrapped_t& impl,
                const std::array<size_t,2>& idx)const{
        return impl(idx[0],idx[1]);
    }

    ///Returns true if two tensors are equal
    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t& lhs, const RHS_t& rhs)const
    {
        return lhs == rhs;
    }

    auto scale(const wrapped_t& lhs,double val)const->decltype(lhs*val)
    {
        return lhs*val;
    }

    ///Adds to the tensor
    template<typename LHS_t,typename RHS_t>
    decltype(auto) add(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs+rhs;
    }

    ///Subtracts from the tensor
    template<typename LHS_t,typename RHS_t>
    decltype(auto) subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs-rhs;
    }

    template<typename...Args> constexpr
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    contract(const Contraction<Args...>& ct)const{
        static_assert(std::tuple_size<decltype(ct.tensors_)>::value==2,
                      "Eigen can not contract more than two tensors at a time");
        const auto& free=get_free_list(ct);
        bool transpose1=ct.get_position(free[0],0)!=0;
        bool transpose2=ct.get_position(free[1],1)!=1;
        const auto& lhs=std::get<0>(ct.tensors_).tensor_;
        const auto& rhs=std::get<1>(ct.tensors_).tensor_;
        if(transpose1 && !transpose2)
            return lhs.transpose()*rhs;
        else if(transpose2 && !transpose1)
            return lhs*rhs.transpose();
        else if(transpose1 && transpose2)
            return lhs.transpose()*rhs.transpose();
        else
            return lhs*rhs;
    }
};



template<>
template<>
Eigen::Tensor<double,2>
TensorConverter<Eigen::Tensor<double,2>>::operator()(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& other)const
{
    Eigen::Tensor<double,2> rv(other.rows(),other.cols());
    std::copy(other.data(),other.data()+other.size(),rv.data());
    return rv;
}


}}//End namespaces
