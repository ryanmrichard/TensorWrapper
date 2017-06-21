namespace TWrapper {
namespace detail_ {


template<size_t rank, typename T>
struct Downcaster{
    template<TensorTypes type>
    using Derived_t=TensorWrapper<rank,T,type>;
static TensorWrapper<rank,T>& downcast(TensorWrapper<rank,T>& tensor,
                                       const TensorType type)
{
    if(type==TensorTypes::EigenMatrix)
        return dynamic_cast<Derived_t<TensorTypes::EigenMatrix>&>(tensor);
    else if(type==TensorTypes::EigenTensor)
        return dynamic_cast<Derived_t<TensorTypes::EigenTensor>&>(tensor);
    else if(type==TensorTypes::GlobalArrays)
        return dynamic_cast<Derived_t<TensorTypes::GlobalArrays>&>(tensor);
    else
        throw std::out_of_range("Unrecognized TensorType");
}

}}//End namespaces
