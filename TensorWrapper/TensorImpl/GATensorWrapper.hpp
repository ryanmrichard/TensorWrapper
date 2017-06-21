//This file meant from inclusion only from TensorImpls.hpp
#include "TensorWrapper/TensorImpl/GA_CXX/GATensor.hpp"
#include <cassert>
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

    ~MemoryFunctor()
    {
        std::array<size_t,rank> temp{};
        for(size_t i=0;i<rank;++i)
            temp[i]=shape_.origin()[i]+shape_.dims()[i];
        parent_.set_values(shape_.origin(),temp,buffer_.data());
    }
};

template<size_t rank, typename T>
struct ConstMemoryFunctor{
    std::vector<T> buffer_;
    Shape<rank> shape_;
    ConstMemoryFunctor(std::vector<T>&& buffer,
                  Shape<rank> shape):
        buffer_(std::move(buffer)),
        shape_(shape)
    {}


    const T& operator()(const std::array<size_t,rank>& idx)
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
    auto get_memory(const Tensor_t& impl)const{
        array_t start,end;
        std::tie(start,end)=impl.my_slice();
        array_t dims;
        std::transform(end.begin(),end.end(),
                       start.begin(),dims.begin(),std::minus<size_t>());
        Shape<rank> my_shape(dims,true);
        ConstMemoryFunctor<rank,T> functor(
                    std::move(impl.get_values(start,end)),
                    my_shape
        );
        auto fxn=std::bind(&ConstMemoryFunctor<rank,T>::operator(),functor,
                           std::placeholders::_1);
        return MemoryBlock<rank,const T>(my_shape,end,fxn,start);
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
        return MemoryBlock<rank,const T>(my_shape,end,fxn,start);
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<rank,T>& block)const
    {
        std::vector<T> buffer;
        for(const auto& idx: block.local_shape)
            buffer[block.local_shape.flat_index(idx)]=block(idx);
        impl.set_values(block.start,block.end,buffer.data());
    }

    type allocate(const array_t& dims)const{
        return type(dims);
    }
    template<typename Tensor_t>
    auto slice(const Tensor_t& impl,
               const array_t& start,
               const array_t& end)const{
        return type(impl.get_values(start,end));
    }

    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t& lhs, const RHS_t& other)const
    {
        return lhs==other;
    }


    template<typename...Args> constexpr
    GATensor<rank,T> contract(const Contraction<Args...>& ct)const{
        static_assert(std::tuple_size<decltype(ct.tensors_)>::value==2,
                      "GA can not contract more than two tensors at a time");

        const auto& free=get_free_list(ct);
        bool transpose1=ct.get_position(free[0],0)!=0;
        bool transpose2=ct.get_position(free[1],1)!=1;
        const auto& lhs=std::get<0>(ct.tensors_).tensor_;
        const auto& rhs=std::get<1>(ct.tensors_).tensor_;
        assert(lhs.get_rank()<=2 &&
               rhs.get_rank()<=2 && "Can't handle higher order yet");

        if(transpose1 && !transpose2)
            return lhs.transpose()*rhs;
        else if(transpose2 && !transpose1)
            return lhs*rhs.transpose();
        else if(transpose1 && transpose2)
            return lhs.transpose()*rhs.transpose();
        return lhs*rhs;

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

//    template<typename My_t>
//    decltype(auto) self_adjoint_eigen_solver(const My_t& tensor)const
//    {
//        using mat_impl=TensorWrapperImpl<2,T,TensorTypes::EigenMatrix>;
//        using mat_type=typename mat_impl::type;
//        using mat_converter=TensorConverter<2,T,TensorTypes::EigenMatrix,
//            TensorTypes::GlobalArrays>;
//        using tensor_converter1=TensorConverter<1,T,TensorTypes::GlobalArrays,
//        TensorTypes::EigenMatrix>;
//        using tensor_converter2=TensorConverter<2,T,TensorTypes::GlobalArrays,
//        TensorTypes::EigenMatrix>;
//        const mat_type ematrix=mat_converter::convert(tensor);
//         Eigen::SelfAdjointEigenSolver<mat_type> solver(ematrix);
//         //Need to copy for the moment as Eigen::Matrix owns memory
//         GATensor<T,1> vals=
//                 tensor_converter1::convert(solver.eigenvalues());
//         GATensor<T,2> vecs=
//                 tensor_converter2::convert(solver.eigenvectors());
//         return std::make_pair(vals,vecs);
//    }
};
}}//End namespaces
