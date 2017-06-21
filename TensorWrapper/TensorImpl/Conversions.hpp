namespace TWrapper {
namespace detail_ {

///Eigen Tensor<2> to Eigen Matrix
template<typename T>
struct TensorConverter<2,T,TensorTypes::EigenMatrix,TensorTypes::EigenTensor>{
    template<typename Input_t>
    static decltype(auto) convert(const Input_t& rhs){
        using Tensor_t=typename TensorWrapperImpl<2,T,TensorTypes::EigenMatrix>::type;
        auto dims=rhs.dimensions();
        return Eigen::Map<Tensor_t>(const_cast<T*>(rhs.data()),dims[0],dims[1]);
    }
};

///Eigen Tensor<1> to Eigen Vector
template<typename T>
struct TensorConverter<1,T,TensorTypes::EigenMatrix,TensorTypes::EigenTensor>{
    template<typename Input_t>
    static decltype(auto) convert(const Input_t& rhs){
        using Tensor_t=typename TensorWrapperImpl<1,T,TensorTypes::EigenMatrix>::type;
        auto dims=rhs.dimensions();
        return Eigen::Map<Tensor_t>(const_cast<T*>(rhs.data()),dims[0]);
    }
};

///Eigen Matrix to Eigen Tensor<2>
template<typename T>
struct TensorConverter<2,T,TensorTypes::EigenTensor,TensorTypes::EigenMatrix>{
    template<typename Input_t>
    static decltype(auto) convert(const Input_t& rhs){
        using Tensor_t=typename TensorWrapperImpl<2,T,TensorTypes::EigenTensor>::type;
        return Eigen::TensorMap<Tensor_t>(const_cast<T*>(rhs.data()),rhs.rows(),rhs.cols());
    }
};

///Eigen Vector to Eigen Tensor<1>
template<typename T>
struct TensorConverter<1,T,TensorTypes::EigenTensor,TensorTypes::EigenMatrix>{
    template<typename Input_t>
    static auto convert(const Input_t& rhs){
        using Tensor_t=typename TensorWrapperImpl<1,T,TensorTypes::EigenTensor>::type;
        size_t ndims= Input_t::ColsAtCompileTime!=1 &&
                      Input_t::RowsAtCompileTime==1 ? rhs.cols() : rhs.rows();
        return Eigen::TensorMap<Tensor_t>(const_cast<T*>(rhs.data()),ndims);
    }

};
}}//End namespaces
