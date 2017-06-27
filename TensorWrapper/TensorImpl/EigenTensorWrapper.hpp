//This file meant from inclusion only from TensorImpls.hpp
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

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
        return MemoryBlock<rank,T>(dims(impl),dims(impl).dims(),
               [&](const array_t& idx)->T&{return impl(idx);});
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<rank,T>& block)const
    {
        for(const auto& idx:block.local_shape)
        {
            array_t full_idx;
            std::transform(idx.begin(),idx.end(),block.start.begin(),
                           full_idx.begin(),std::plus<size_t>());
            impl(full_idx)=block(idx);
        }
    }

    type allocate(const array_t& dims)const{
        return allocater<type>(dims);
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


    template<typename...Args> constexpr
    type contract(const Contraction<Args...>& ct)const{
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

    template<typename My_t>
    decltype(auto) self_adjoint_eigen_solver(const My_t& tensor)const
    {
        using mat_impl=TensorWrapperImpl<2,T,TensorTypes::EigenMatrix>;
        using mat_type=typename mat_impl::type;
        using mat_converter=TensorConverter<2,T,TensorTypes::EigenMatrix,
            TensorTypes::EigenTensor>;
        using tensor_converter1=TensorConverter<1,T,TensorTypes::EigenTensor,
        TensorTypes::EigenMatrix>;
        using tensor_converter2=TensorConverter<2,T,TensorTypes::EigenTensor,
        TensorTypes::EigenMatrix>;
        const mat_type ematrix=mat_converter::convert(tensor);
         Eigen::SelfAdjointEigenSolver<mat_type> solver(ematrix);
         //Need to copy for the moment as Eigen::Matrix owns memory
         Eigen::Tensor<T,1> vals=
                 tensor_converter1::convert(solver.eigenvalues());
         Eigen::Tensor<T,2> vecs=
                 tensor_converter2::convert(solver.eigenvectors());
         return std::make_pair(vals,vecs);
    }
};
}}//End namespaces
