//This file meant from inclusion only from TensorImpls.hpp
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <map>
namespace TWrapper {
namespace detail_ {

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
    MemoryBlock<2,T> get_memory(Tensor_t&& impl)const{
        return MemoryBlock<2,T>(dims(impl),dims(impl).dims(),
               [&](const array_t& idx)->T&{return impl(idx[0],idx[1]);});
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
        return MemoryBlock<1,T>(dims(impl),dims(impl).dims(),
               [&](const array_t& idx)->T&{return impl(idx[0]);});
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


}}//End namespaces
