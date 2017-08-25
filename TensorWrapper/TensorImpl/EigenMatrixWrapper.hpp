//This file meant from inclusion only from TensorImpls.hpp
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <map>
#include "TensorWrapper/TensorImpl/ContractionHelper.hpp"
namespace TWrapper {
namespace detail_ {

template<size_t R, typename T, typename Tensor_t>
struct MemoryGetterHelper
{

    template<size_t...I>
    MemoryBlock<R,T> eval_guts(const Shape<R>& shape,
                                     const std::array<size_t,R> dims,
                                     Tensor_t& t,
                                     std::index_sequence<I...>)const
         {
             return MemoryBlock<R,T>(shape,dims,
                 [&](const std::array<size_t,R>& idx)->
                                     T&{return t(idx[I]...);});
         }


    MemoryBlock<R,T> eval(const Shape<R>& shape,
                                const std::array<size_t,R> dims,
                                Tensor_t& t)const
    {
        return eval_guts(shape,dims,t,std::make_index_sequence<R>());
    }

};


//Matrix specialization
template<typename T>
struct TensorWrapperImpl<2,T,TensorTypes::EigenMatrix> {

    using array_t=std::array<size_t,2>;
    using type=Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;

    template<typename Tensor_t>
    Shape<2> dims(const Tensor_t& impl)const{
        size_t rows=(size_t)impl.rows();
        size_t cols=(size_t)impl.cols();
        return Shape<2>(array_t{rows,cols},impl.IsRowMajor);
    }

    template<typename Tensor_t>
    auto get_memory(Tensor_t& impl)const{
        MemoryGetterHelper<2,T,Tensor_t> helper;
        return helper.eval(dims(impl),dims(impl).dims(),impl);
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<2,T>& block)const
    {
        //Check if it's actually the T* of this tensor
        if(&block(0,0)==&impl(block.start[0],block.start[1]))
            return;
        for(const auto& idx:block.local_shape)
                impl(idx[0]+block.start[0],idx[1]+block.start[1])=block(idx);
    }

    type allocate(const array_t& dims)const{
        return type(dims[0],dims[1]);
    }

    template<typename Tensor_t>
    auto permute(const Tensor_t& t,
                 const array_t&)const
    {
        //Only one possibility {1,0}
        return t.transpose();
    }

    template<typename Tensor_t>
    auto slice(const Tensor_t& impl,
               const array_t& start,
               const array_t& end)const{
        return type(impl.block(start[0],start[1],end[0]-start[0],end[1]-start[1]));
    }

    ///Returns true if two tensors are equal
    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t& lhs, const RHS_t& rhs)const
    {
        return lhs == rhs;
    }

    template<typename Tensor_t>
    auto scale(const Tensor_t& lhs,double val)const
    {
        return lhs*val;
    }

    ///Adds to the tensor
    template<typename LHS_t,typename RHS_t>
    auto add(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs+rhs;
    }

    ///Subtracts from the tensor
    template<typename LHS_t,typename RHS_t>
    auto subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs-rhs;
    }

    template<typename LHS_t,typename RHS_t,typename LHS_Idx,typename RHS_Idx>
    auto contraction(const LHS_t& lhs, const RHS_t& rhs,
                     const LHS_Idx,const RHS_Idx&)const
    {
        constexpr bool isvec=RHS_Idx::size()==1;
        constexpr bool rrow=RHS_t::RowsAtCompileTime==1;
        using contract=ContractionTraits<LHS_Idx,RHS_Idx,2,RHS_Idx::size()>;
        constexpr bool rtranspose=(!isvec?contract::rtranspose:
                                          contract::rtranspose!=rrow);
        return ContractionHelper<contract::nfree,
                                 contract::ndummy,
                                 contract::ltranspose,
                                 rtranspose>().contract(lhs,rhs);
    }

    template<typename My_t>
    auto self_adjoint_eigen_solver(const My_t& tensor)const
    {
         Eigen::SelfAdjointEigenSolver<My_t> solver(tensor);
         return std::make_pair(solver.eigenvalues(),solver.eigenvectors());
    }

};

//Vector specialization
template<typename T>
struct TensorWrapperImpl<1,T,TensorTypes::EigenMatrix> {

    using array_t=std::array<size_t,1>;

    using type=Eigen::Matrix<T,Eigen::Dynamic,1>;

    template<typename Tensor_t>
    Shape<1> dims(const Tensor_t& impl)const{
        return Shape<1>(array_t{(size_t)impl.rows()},impl.IsRowMajor);
    }

    template<typename Tensor_t>
    auto get_memory(Tensor_t& impl)const{
        MemoryGetterHelper<1,T,Tensor_t> helper;
        return helper.eval(dims(impl),dims(impl).dims(),impl);
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<1,T>& block)const
    {
        for(const auto& idx:block.local_shape)
                impl(idx[0]+block.start[0])=block(idx);
    }

    type allocate(const array_t& dims)const{
        return type(dims[0]);
    }

    template<typename Tensor_t>
    auto permute(const Tensor_t& t,const array_t&)
    {
        return t.transpose();
    }

    template<typename Tensor_t>
    auto slice(const Tensor_t& impl,
               const array_t& start,
               const array_t& end)const{
        return type(impl.segment(start[0],end[0]-start[0]));
    }

    ///Returns true if two tensors are equal
    template<typename LHS_t, typename RHS_t>
    bool are_equal(LHS_t&& lhs, RHS_t&& rhs)const
    {
        return lhs == rhs;
    }

    template<typename Tensor_t>
    auto scale(Tensor_t&& lhs,double val)const
    {
        return lhs*val;
    }

    ///Adds to the tensor
    template<typename LHS_t,typename RHS_t>
    auto add(LHS_t&& lhs,RHS_t&& rhs)const
    {
        return lhs+rhs;
    }

    ///Subtracts from the tensor
    template<typename LHS_t,typename RHS_t>
    auto subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs-rhs;
    }

    template<typename LHS_t,typename RHS_t,typename LHS_Idx,typename RHS_Idx>
    auto contraction(const LHS_t& lhs, const RHS_t& rhs,
                     const LHS_Idx&,const RHS_Idx&)const
    {
        constexpr bool lrow=LHS_t::RowsAtCompileTime==1;
        constexpr bool is_vec=(RHS_Idx::size()==1);
        constexpr bool rrow=RHS_t::RowsAtCompileTime==1;
        using contract=ContractionTraits<LHS_Idx,RHS_Idx,1,RHS_Idx::size()>;
        constexpr bool rtranspose=(!is_vec?contract::rtranspose:
                                           rrow!=contract::rtranspose);
        return ContractionHelper<contract::nfree,
                                 contract::ndummy,
                                 contract::ltranspose!=lrow,
                                 rtranspose>().contract(lhs,rhs);
    }
};

//Scalar specialization
template<typename T>
struct TensorWrapperImpl<0,T,TensorTypes::EigenMatrix> {

    using array_t=std::array<size_t,0>;

    using type=Eigen::Matrix<T,1,1>;

    template<typename Tensor_t>
    Shape<0> dims(const Tensor_t& impl)const{
        return Shape<0>(array_t{},impl.IsRowMajor);
    }

    template<typename Tensor_t>
    auto get_memory(Tensor_t& impl)const{
        return MemoryBlock<0,T>(dims(impl),dims(impl).dims(),
                [&](const array_t&)->T&{return impl.data()[0];});
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<0,T>& block)const
    {
        impl.data()[0]=block(array_t{});
    }

    type allocate(const array_t&)const{
        return type{};
    }

    template<typename Tensor_t>
    auto permute(const Tensor_t& t,
                 const array_t&)
    {
        t.transpose();
    }

    template<typename Tensor_t>
    auto slice(const Tensor_t& impl,
               const array_t&,
               const array_t&)const{
        return impl;
    }

    ///Returns true if two tensors are equal
    template<typename LHS_t, typename RHS_t>
    bool are_equal(LHS_t&& lhs, RHS_t&& rhs)const
    {
        return lhs == rhs;
    }

    template<typename Tensor_t>
    auto scale(Tensor_t&& lhs,double val)const
    {
        return lhs*val;
    }

    ///Adds to the tensor
    template<typename LHS_t,typename RHS_t>
    auto add(LHS_t&& lhs,RHS_t&& rhs)const
    {
        return lhs+rhs;
    }

    ///Subtracts from the tensor
    template<typename LHS_t,typename RHS_t>
    auto subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs-rhs;
    }

    template<typename LHS_t,typename RHS_t,typename LHS_Idx,typename RHS_Idx>
    auto contraction(const LHS_t& lhs, const RHS_t& rhs,
                     const LHS_Idx&,const RHS_Idx&)const
    {
        return lhs*rhs;
    }
};

}}//End namespaces
