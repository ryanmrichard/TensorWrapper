#pragma once
#include "TensorWrapper/TensorImpl/TensorWrapperImpl.hpp"
#include "TensorWrapper/TensorImpl/ContractionHelper.hpp"
#include <ga_cxx/GATensor.hpp>

namespace TWrapper {
namespace detail_ {

template<bool lt, bool rt,size_t nfree>
struct GAContract{
    template<typename lhs_t, typename rhs_t>
    static auto eval(const lhs_t& lhs, const rhs_t& rhs)
    {
        if(lt && rt)
            return lhs.transpose()*rhs.transpose();
        if(lt && !rt)
            return lhs.transpose()*rhs;
        if(!lt && rt)
            return lhs*rhs.transpose();
        return lhs *rhs;
    }
};

template<bool lt, bool rt>
struct GAContract<lt,rt,0>{
    template<typename lhs_t, typename rhs_t>
    static auto eval(const lhs_t& lhs, const rhs_t& rhs)
    {
        if(lt && rt)
            return lhs.transpose().dot(rhs.transpose());
        if(lt && !rt)
            return lhs.transpose().dot(rhs);
        if(!lt && rt)
            return lhs.dot(rhs.transpose());
        return lhs.dot(rhs);
    }
};

template<size_t rank, typename T>
struct TensorWrapperImpl<rank,T,TensorTypes::GlobalArrays> {
    using type = GATensor<rank,T>;
    using array_t = std::array<size_t,rank>;

    Shape<rank> dims(const type& impl)const{
        return Shape<rank>(impl.dims(),true);
    }

    template<typename Tensor_t>
    auto get_memory(Tensor_t& impl)const{
        array_t start,end;
        std::tie(start,end)=impl.my_slice();
        auto mem=impl.get_values(start,end);
        Shape<rank> my_shape(end,true,start);
        MemoryBlock<rank,T> rv;
        std::unique_ptr<T[]> ptr(new T[my_shape.size()]);
        std::copy(mem.begin(),mem.end(),ptr.get());
        rv.add_block(std::move(ptr),my_shape);
        return rv;
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<rank,T>& block)const
    {
        for(size_t i=0;i<block.nblocks();++i)
        {
            impl.set_values(*block.begin(i),*block.end(i),block.block(i));
        }
    }

    type allocate(const array_t& dims)const{
        return type(dims);
    }


    template<typename LHS_Idx,typename LHS_t>
    auto trace(const LHS_t& lhs)const
    {
        static_assert(LHS_Idx().size()==2,"Trace only available for matrix");
        return lhs.trace();
    }

    template<typename Tensor_t>
    auto permute(const Tensor_t& t, const array_t&)const
    {
        return t.transpose();
    }

    template<typename Tensor_t>
    auto slice(const Tensor_t& impl,
               const array_t& start,
               const array_t& end)const{
        auto data=impl.get_values(start,end);
        array_t new_dims,zero{};
        for(size_t i=0;i<start.size();++i)
            new_dims[i]=end[i]-start[i];
        type rv(new_dims);
        rv.set_values(zero,new_dims,data.data());
        return rv;
    }

    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t& lhs, const RHS_t& other)const
    {
        return lhs==other;
    }

    template<typename Index_t, typename LHS_t>
    auto scale(const LHS_t& lhs,double val)const
    {
        return lhs*val;
    }

    template<typename Op_t>
    type eval(const Op_t& op,const array_t&)const
    {
        type c=op;
        return c;
    }

    ///Adds to the tensor
    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t>
    auto add(const LHS_t& lhs,const RHS_t&rhs)const
    {
        if(std::is_same<LHS_Idx,RHS_Idx>::value)
            return lhs+rhs;
        return lhs+rhs.transpose();
    }

    ///Subtracts from the tensor
    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t>
    auto subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        if(std::is_same<LHS_Idx,RHS_Idx>::value)
            return lhs-rhs;
        return lhs-rhs.transpose();
    }

    template<typename,typename Op_t>
    auto eval(const Op_t& op,const array_t&)const
    {
        type c=op;
        return c;
    }

    ///Contraction
    template<typename LHS_Idx, typename RHS_Idx,typename LHS_t, typename RHS_t>
    auto contraction(const LHS_t& lhs, const RHS_t& rhs)const
    {
        static_assert(LHS_Idx::size()<3&&RHS_Idx::size()<3,
                      "GA does not support real contraction");
        using traits=
            ContractionTraits<LHS_Idx,RHS_Idx,LHS_Idx::size(),RHS_Idx::size()>;
        return GAContract<traits::ltranspose,traits::rtranspose,traits::nfree>::
                eval(lhs,rhs);
    }


//    template<typename My_t>
//    auto self_adjoint_eigen_solver(const My_t& tensor)const
//    {
//        const Shape<2> shape=dims(tensor);
//        const size_t n=shape.dims()[0];
//        Eigen::MatrixXd temp(n,n);
//        std::copy(tensor.data(),tensor.data()+n*n,temp.data());
//        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(temp);
//        GATensor<1,T> evals(std::array<size_t,1>{n});
//        GATensor<2,T> evecs(index_t{n,n});
//        const double* from=solver.eigenvalues().data();
//        double* to=evals.data();
//        std::copy(from,from+n,to);
//    const double* from2=solver.eigenvectors().data();
//    to=evecs.data();
//    std::copy(from2,from2+n*n,to);
//    return std::make_pair(evals,evecs);
//    }
};
}}//End namespaces
