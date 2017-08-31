//This file meant from inclusion only from TensorImpls.hpp
#define EIGEN_USE_THREADS //Enables threading for eigen::tensor
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/ThreadPool>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "TensorWrapper/Indices.hpp"
namespace TWrapper {
namespace detail_ {
template<size_t rank,typename T,TensorTypes LHS_t,TensorTypes RHS_t>
struct TensorConverter;

//Eigen tensor only supports comma seperated initialization and not array
//This and the next function unroll the array
template <typename Tensor_t, size_t rank, std::size_t... I>
auto allocate_guts(const std::array<size_t,rank>& idx, std::index_sequence<I...>)
{
     return Tensor_t(idx[I]...);
}

template<typename Tensor_t, size_t rank>
auto allocater(const std::array<size_t,rank>& idx)
{
     return allocate_guts<Tensor_t>(idx, std::make_index_sequence<rank>{});
}



template<size_t rank, typename T>
struct TensorWrapperImpl<rank,T,TensorTypes::EigenTensor> {
    using array_t=std::array<size_t,rank>;
    using type=Eigen::Tensor<T,rank>;

    template<typename My_t>
    Shape<rank> dims(const My_t& impl)const{
        array_t dims;
        auto dim=impl.dimensions();
        for(size_t i=0;i<rank;++i)dims[i]=dim[i];
        return Shape<rank>(dims,impl.Layout==Eigen::RowMajor);
    }

    template<typename Tensor_t>
    MemoryBlock<rank,T> get_memory(Tensor_t& impl)const{
        MemoryBlock<rank,T> rv;
        rv.add_block(impl.data(),dims(impl),array_t{},dims(impl).dims());
        return rv;
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<rank,T>& block)const
    {
        SetMemoryImpl<rank,T>::eval(impl,block,std::make_index_sequence<rank>());
    }

    type allocate(const array_t& dims)const{
        return allocater<type>(dims);
    }

    template<typename Tensor_t>
    auto permute(const Tensor_t& t, const array_t& permutation)const
    {
        return t.shuffle(permutation);
    }

    template<typename Tensor_t>
    auto slice(const Tensor_t& impl,
               const array_t& start,
               const array_t& end)const{
        array_t temp{};
        for(size_t i=0;i<rank;++i)temp[i]=end[i]-start[i];
        return type(impl.slice(start,temp));
    }

    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t& lhs, const RHS_t& other)const
    {
        Eigen::Tensor<bool,0> rv= (lhs==other).all().eval();
        return rv(0);
    }


    template<typename LHS_Idx, typename RHS_Idx,
             typename LHS_t, typename RHS_t>
    auto contraction(const LHS_t& lhs, const RHS_t& rhs)
    {
        constexpr size_t ndummy=LHS_Idx::ncommon(RHS_Idx());
        const auto dummy=detail_::get_dummy(LHS_Idx(),RHS_Idx());
        std::array<std::pair<size_t,size_t>,ndummy> temp;
        for(size_t i=0; i<ndummy;++i)
            temp[i]=std::make_pair(dummy.first[i],dummy.second[i]);
        return lhs.contract(rhs,temp);
    }

    template<typename,typename LHS_t>
    auto scale(const LHS_t& lhs,double val)const
    {
        return lhs*val;
    }

    template<typename L, typename R>
    using EnableIfSame=std::enable_if<std::is_same<L,R>::value,int>;

    template<typename L, typename R>
    using EnableIfNotSame=std::enable_if<!std::is_same<L,R>::value,int>;

    ///Adds to the tensor
    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t,
             typename EnableIfSame<LHS_Idx,RHS_Idx>::type=0>
    auto add(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs+rhs;
    }

    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t,
             typename EnableIfNotSame<LHS_Idx,RHS_Idx>::type=0>
    auto add(const LHS_t& lhs,const RHS_t&rhs)const
    {
        auto map=LHS_Idx::get_map(RHS_Idx());
        return lhs+rhs.shuffle(map);
    }

    ///Subtracts from the tensor
    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t,
             typename EnableIfSame<LHS_Idx,RHS_Idx>::type=0>
    auto subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs-rhs;
    }

    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t,
             typename EnableIfNotSame<LHS_Idx,RHS_Idx>::type=0>
    auto subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        auto map=LHS_Idx::get_map(RHS_Idx());
        return lhs-rhs.shuffle(map);
    }

    template<typename,typename Op_t>
    type eval(const Op_t& op,const array_t& dims)const
    {
        const int nthreads=omp_get_max_threads();
        Eigen::ThreadPool pool(nthreads);
        Eigen::ThreadPoolDevice my_device(&pool,nthreads);
        auto c=allocate(dims);
        c.device(my_device)=op;
        return c;
    }

    template<typename My_t>
    auto self_adjoint_eigen_solver(const My_t& tensor)const
    {

        const Shape<2> shape=dims(tensor);
        const size_t n=shape.dims()[0];
        Eigen::MatrixXd temp(n,n);
        std::copy(tensor.data(),tensor.data()+n*n,temp.data());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(temp);
        Eigen::Tensor<T,1> evals(n);
        Eigen::Tensor<T,2> evecs(n,n);
        const double* from=solver.eigenvalues().data();
        double* to=evals.data();
        std::copy(from,from+n,to);
        const double* from2=solver.eigenvectors().data();
        to=evecs.data();
        std::copy(from2,from2+n*n,to);
        return std::make_pair(evals,evecs);
    }
};
}}//End namespaces
