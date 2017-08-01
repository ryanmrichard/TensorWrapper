#pragma once
#include <utility>
#include <stdexcept>

namespace TWrapper {
namespace detail_{
///Enumerations of the tensors
enum class TensorTypes {EigenMatrix,
                        EigenTensor,
                        GlobalArrays
};
///Macro for calling a function with one of the TensorTypes
#define TTGuts(name)\
    fxn_t().template eval<name>(std::forward<Args>(args)...)

///Macro for wrapping the call in an if statement
#define TTEntry(name)\
if(type==name){return TTGuts(name);}

/** \brief Code factorization for running through all of the possible tensor
 *  backends and then calling a function based on the the backend.
 *
 *
 * \param[in] type The tensortype (known at runtime) we want to apply.
 * \param[in] args The arguments that will be passed to the function.
 *
 * \tparam fxn_t The type of a functor that has a member function "eval" that is
 *               templated on TensorTypes parameter.
 */
template<typename fxn_t,typename...Args>
auto apply_TensorTypes(TensorTypes type,Args&&...args)->
decltype(TTGuts(TensorTypes::EigenMatrix))
{
    TTEntry(TensorTypes::EigenMatrix)
    TTEntry(TensorTypes::EigenTensor)
    throw std::logic_error("I don't know what crazy tensor you're trying to"
                            " get, but I don't know how to make it.");
}

#undef TTEntry
#undef TTGuts
}}
