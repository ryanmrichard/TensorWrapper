namespace TWrapper {
namespace detail_ {

/** \brief Defines a conversion from RHS_t->LHS_t, the two sides are assumed to
 *         have the same data types and ranks
 *
 *  For the most part these conversions should be pretty straightforward, the
 *  biggest hiccup is with const.  To explain this situation let's consider two
 *  fictitious tensor backend classes InputTensor<T> and OutputTensor<T>.  These
 *  classes are respectively the types of the tensor we are converting from and
 *  the type we are converting to.  Furthermore, the class only depends on the
 *  data type stored inside it (this is the only relevant member for our const
 *  problem).  When we write our conversion we get something like:
 *
 *  \code{.cpp}
 *  OutputTensor<T> convert(const Tensor<T>& input)
 *  {
 *     //Get data
 *     const T* data=input.data();
 *     //Put data and return
 *     return OutputTensor<T>(data);
 *  }
 *  \endcode
 *
 *  See the problem?  Our output tensor has some buffer internally of type T*.
 *  We can't assign const T* to T*.  As a result these conversions can only be
 *  applied to non-const instances.  If you have a const instance you will need
 *  to const_cast it (I promise this class won't modify it); however, you'll be
 *  in charge of ensuring that the returned class is also treated in a const
 *  fashion.
 *
 *  Since if we support "N" backends there are approximately "N^2" conversions
 *  the primary template serves as an all else fails conversion that manually
 *  copies the data over.
 *
 */
template<size_t rank,typename T,TensorTypes LHS_t,TensorTypes RHS_t>
struct TensorConverter;

///Null operation (needed for compiling, never actually called)
template<size_t rank, typename T,TensorTypes Tensor_t>
struct TensorConverter<rank,T,Tensor_t,Tensor_t>{
    template<typename Input_t>
    static Input_t convert(Input_t&& rhs)
    {
        return rhs;
    }
};

///Eigen Tensor<2> to Eigen Matrix
template<typename T>
struct TensorConverter<2,T,TensorTypes::EigenMatrix,TensorTypes::EigenTensor>{
    template<typename Input_t>
    static auto convert(Input_t&& rhs){
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
