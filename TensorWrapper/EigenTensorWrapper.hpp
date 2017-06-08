#pragma once
#include "TensorWrapper/TensorWrapperImpl.hpp"
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
namespace TWrapper {
namespace detail_ {

template<typename T>
std::vector<std::pair<int,int>> contraction_helper(const T& ct){
    const auto& contraction_list=get_contraction_list(ct);
    const size_t n_contractions=contraction_list.size();
    std::vector<std::pair<int,int>> rv(n_contractions);
    for(size_t i=0;i<n_contractions;++i)
    {
        const int t1indexi=(int)ct.get_position(contraction_list[i],0);
        const int t2indexi=(int)ct.get_position(contraction_list[i],1);
        rv[i]=std::make_pair(t1indexi,t2indexi);
    }
    return rv;
}


template<size_t rank, typename T>
struct TensorWrapperImpl<rank,T,Eigen::Tensor<T,rank>> {
    using wrapped_t = Eigen::Tensor<T,rank>;
    Shape<rank> dims(const Eigen::Tensor<T,rank>& impl)const{
        std::array<size_t,rank> dims;
        auto dim=impl.dimensions();
        for(size_t i=0;i<rank;++i)dims[i]=dim[i];
        return Shape<rank>(dims,false);
    }

    T get_value(const Eigen::Tensor<T,rank>& impl,
                const std::array<size_t,rank>& idx)const{
        return impl(idx);
    }

    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t& lhs, const RHS_t& other)const
    {
        Eigen::Tensor<bool,0> rv= (lhs==other).all().eval();
        return rv(0);
    }


    template<typename...Args> constexpr
    Eigen::Tensor<T,rank> contract(const Contraction<Args...>& ct)const{
        static_assert(std::tuple_size<decltype(ct.tensors_)>::value==2,
                      "Eigen can not contract more than two tensors at a time");
        const auto& lhs=std::get<0>(ct.tensors_).tensor_;
        const auto& rhs=std::get<1>(ct.tensors_).tensor_;
        const auto cs=contraction_helper(ct);
        using idx_t=std::pair<int,int>;
        if(cs.size()==1)
            return lhs.contract(rhs,std::array<idx_t,1>({cs[0]}));
        else if(cs.size()==2)
            return lhs.contract(rhs,std::array<idx_t,2>({cs[0],cs[1]}));
        //Not sure what's going on here, but it seems to infinetly recurse beyond here
//        else if(cs.size()==3)
//            return lhs.contract(rhs,std::array<idx_t,3>({cs[0],cs[1],cs[2]}));
//        else if(cs.size()==4)
//            return lhs.contract(rhs,std::array<idx_t,4>({cs[0],cs[1],cs[2],cs[3]}));
//        else if(cs.size()==5)
//            return lhs.contract(rhs,std::array<idx_t,5>({cs[0],cs[1],cs[2],cs[3],cs[4]}));
        else
            throw std::out_of_range("I didn't go above unrolling 2 indices");
    }

    template<typename LHS_t>
    auto scale(const LHS_t& lhs,double val)const->decltype(lhs*val)
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
};
}}//End namespaces
