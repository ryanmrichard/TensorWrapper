//This file meant from inclusion only from TensorImpls.hpp
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <map>
namespace TWrapper {
namespace detail_ {

template<size_t NFree, size_t NDummy, bool rows_same, bool cols_same>
struct ContractionHelper{};

#define CHelperSpecial(NFree,NDummy,ltranspose,rtranspose,guts)\
template<>\
struct ContractionHelper<NFree,NDummy,ltranspose,rtranspose>{\
template<typename LHS_t,typename RHS_t>\
auto contract(const LHS_t& lhs, const RHS_t& rhs){\
   return guts;\
}}

//i,j * i,j or j,i * j,i
CHelperSpecial(0,2,false,false,lhs.cwiseProduct(rhs).sum());
//i,j * j,i or j,i * i,j
CHelperSpecial(0,2,false,true,lhs.cwiseProduct(rhs.transpose()).sum());
//i,j * k,j
CHelperSpecial(1,1,false,true,lhs*rhs.transpose());
//j,i * j,k
CHelperSpecial(1,1,true,false,lhs.transpose()*rhs);
//i,j * j,k
CHelperSpecial(1,1,false,false,lhs*rhs);
//j,i * k,j
CHelperSpecial(1,1,true,true,lhs.transpose()*rhs.transpose());
//i * i
CHelperSpecial(0,1,false,false,lhs.dot(rhs));
//i * j or j*i
CHelperSpecial(1,0,false,true,lhs*rhs.transpose());

#undef CHelperSpecial


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
        constexpr bool rows_same=
                (LHS_Idx::template get<0>()==RHS_Idx::template get<0>());
        constexpr bool cols_same=
                (LHS_Idx::template get<1>()==RHS_Idx::template get<1>());
        constexpr bool rowcol=
                (LHS_Idx::template get<0>()==RHS_Idx::template get<1>());
        constexpr bool all_same=(rows_same && cols_same);

        constexpr bool transpose1=(rows_same || rowcol) && !all_same;
        constexpr bool transpose2=(cols_same || rowcol) && !all_same;

        constexpr size_t nfree=LHS_Idx::nunique(RHS_Idx());
        constexpr size_t ndummy=LHS_Idx::ncommon(RHS_Idx());
        return ContractionHelper<nfree,ndummy,transpose1,transpose2>().
                contract(lhs,rhs);
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
        constexpr bool transpose=
                (LHS_Idx::template get<0>()!=RHS_Idx::template get<0>());
        constexpr size_t nfree=LHS_Idx::nunique(RHS_Idx());
        constexpr size_t ndummy=LHS_Idx::ncommon(RHS_Idx());
        return ContractionHelper<nfree,ndummy,false,transpose>().
                contract(lhs,rhs);
    }
};


}}//End namespaces
