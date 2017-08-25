//This file meant from inclusion only from TensorImpls.hpp
#include <ga_cxx/GATensor.hpp>
#include "TensorWrapper/TensorImpl/ContractionHelper.hpp"

namespace TWrapper {
namespace detail_ {

template<size_t rank, typename T>
struct MemoryFunctor{
    std::vector<T> buffer_;
    GATensor<rank,T>& parent_;
    Shape<rank> shape_;
    MemoryFunctor(std::vector<T>&& buffer,
                  GATensor<rank,T>& parent,
                  Shape<rank> shape):
        buffer_(std::move(buffer)),
        parent_(parent),
        shape_(shape)
    {}


    T& operator()(const std::array<size_t,rank>& idx)
    {
        return buffer_[shape_.flat_index(idx)];
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
        array_t dims;
        std::transform(end.begin(),end.end(),
                       start.begin(),dims.begin(),std::minus<size_t>());
        Shape<rank> my_shape(dims,true);
        MemoryFunctor<rank,T> functor(
            std::move(impl.get_values(start,end)),
            impl,my_shape
        );
        auto fxn=std::bind(&MemoryFunctor<rank,T>::operator(),functor,
                           std::placeholders::_1);
        return MemoryBlock<rank,T>(my_shape,end,fxn,start);
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<rank,T>& block)const
    {
        std::vector<T> buffer(block.local_shape.size());
        for(const auto& idx: block.local_shape)
            buffer[block.local_shape.flat_index(idx)]=block(idx);
        impl.set_values(block.start,block.end,buffer.data());
    }

    type allocate(const array_t& dims)const{
        return type(dims);
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

    template<typename LHS_t>
    auto scale(const LHS_t& lhs,double val)const->decltype(lhs*val)
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

    ///Contraction
    template<typename LHS_t, typename RHS_t, typename LHS_Idx, typename RHS_Idx>
    auto contraction(const LHS_t& lhs, const RHS_t& rhs,
                     const LHS_Idx,const RHS_Idx)const
    {
        static_assert(LHS_Idx::size()<3&&RHS_Idx::size()<3,
                      "GA does not support real contraction");
        using contract=ContractionTraits<LHS_Idx,RHS_Idx,LHS_Idx::size(),
                                                         RHS_Idx::size()>;
        return ContractionHelper<contract::nfree,
                                 contract::ndummy,
                                 contract::ltranspose,
                                 contract::rtranspose>().
                contract(lhs,rhs);
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
