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
};

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

    ///End point of contraction recursion
    template<size_t lhs_rank, size_t rhs_rank,typename LHS_t, typename RHS_t>
    auto contract(const Contraction<IndexedTensor<lhs_rank,LHS_t>,
                                    IndexedTensor<rhs_rank,RHS_t>>& ct)const->
        decltype(ct.lhs_.tensor_.contract(ct.rhs_.tensor_,std::array<Eigen::IndexPair<int>,1>()))
    {
        std::array<Eigen::IndexPair<int>,1>  idx;
        for(const auto x : ct.idx2contract_)
        {
            size_t lpos=0,rpos=0;
            for(;lpos<lhs_rank;++lpos)
                if(ct.lhs_.idx_[lpos]==x)
                    break;
            for(;rpos<rhs_rank;++rpos)
                if(ct.rhs_.idx_[rpos]==x)
                    break;
            idx[0]=(Eigen::IndexPair<int>({(int)lpos,(int)rpos}));
        }
        return ct.lhs_.tensor_.contract(ct.rhs_.tensor_,idx);
    }

};

//template<size_t rank, typename T>
//template<>
//Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
//TensorConverter<rank,T>::convert(const Eigen::Tensor<T,rank>& other)
//{

//}

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
