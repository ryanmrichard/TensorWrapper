#pragma once

namespace TWrapper {
namespace detail_{
    ///Enumerations of the tensors
    enum class TensorTypes {EigenMatrix,
                            EigenTensor,
                            GlobalArrays};


/** \brief Defines a conversiont from RHS_t->LHS_t, the two sides are assumed to
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
 *  We can't assign const T* to T*.  Casts are meant to be done internally where
 *  we can assure we are not modifying the contents of a tensor.  At the moment
 *  all casts involiving const tensors being cast occur on the right side of
 *  some operator where, after the cast, the result is forwarded to another
 *  function via constant reference.  This means we can circumvent this problem
 *  with a const cast.
 *
 *  Since if we support "N" backends there are approximately "N^2" conversions
 *  the primary template serves as an all else fails conversion that manually
 *  copies the data over.
 *
 */
template<size_t rank,typename T,TensorTypes LHS_t,TensorTypes RHS_t>
struct TensorConverter;

}}
