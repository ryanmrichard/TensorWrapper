#pragma once
#include <utility>
#include <memory>
#include "TensorWrapper/TensorImpl/TensorTypes.hpp"

namespace TWrapper {
namespace detail_ {

/** \brief This is a pointer that can hold any type of tensor.
 *
 *  This class works off of type erasure.  Basically  we have a dummy base
 *  class that we hold the data as, then when we need the data we downcast to
 *  a dummy derived class which is templated on the data's type and actually
 *  stores the data.  From this class we can easily retrieve the data.  Note
 *  that this is fully type safe because you need to know the data's type to
 *  get it back.  If you're curious, and because this is perhaps a somewhat
 *  obscure fact, the little bit of template metaprogramming
 *  is because of the fact:
 *  \code{.cpp}
 *  template<typename T>
 *  void function(T&& value);
 *  \endcode
 *  Does not take an rvalue reference to an instance of type T, but rather
 *  takes a universal reference to an instance of type T.
 *
 */
class TensorPtr{
private:
    ///This class literally exists just to hold the next class
    struct Placeholder{
      ///Ensure we have a virtual destructor to get proper deletion
      virtual ~Placeholder()=default;
      ///This function will implment a polymorphic copy for us
      virtual std::unique_ptr<Placeholder> clone()const=0;
    };

    ///Layer on top of base class that actually holds the instance
    template <typename Tensor_t>
    struct Wrapper : public Placeholder
    {
      ///Copies tensor \p t into this instance
      explicit Wrapper(const Tensor_t& t):
            tensor(t)
      {}

      ///Moves tensor \p t into this instance
      explicit Wrapper(Tensor_t&& t):
          tensor(std::forward<Tensor_t>(t))
      {}

      ///Returns a deep copy of tensor, avoiding slicing
      std::unique_ptr<Placeholder> clone()const override
      {
          return std::make_unique<Wrapper<Tensor_t>>(tensor);
      }

      ///The instance in the pointer
      Tensor_t tensor;

     };

    ///The actual instance is wrapped in this member
    std::unique_ptr<Placeholder> tensor_;


public:
    ///Makes an empty TensorPtr instance
    TensorPtr()=default;

    /** \brief Makes a TensorPtr by copying a tensor instance.
     *
     *  Whether this is a deep or shallow copy of \p tensor is up to the
     *  tensor library backend.
     *
     *  \param[in] tensor The instance to copy.
     *  \tparam Tensor_t The type of the tensor we are copying.
     */
    template<typename Tensor_t>
    explicit TensorPtr(const Tensor_t& tensor):
        tensor_(std::make_unique<Wrapper<
                typename std::remove_reference<Tensor_t>::type>>(
                    tensor))
    {}


    ///Same as above except for rvalues and non-const lvalues
    template<typename Tensor_t>
    explicit TensorPtr(typename std::remove_reference<Tensor_t>::type && tensor):
        tensor_(std::make_unique<Wrapper<
                typename std::remove_reference<Tensor_t>::type>>(
                    std::forward<Tensor_t>(tensor)))
    {}

    ///Deep copies another TensorPtr
    TensorPtr(const TensorPtr& other):
      tensor_(std::move(other.tensor_->clone()))
    {
    }

    ///Takes ownership of other TensorPtr's tensor
    TensorPtr(TensorPtr&& other):
        tensor_(std::move(other.tensor_))
    {}

    ///Allows one to get the tensor back, assuming they know its type
    template<typename Tensor_t>
    Tensor_t& cast()
    {
        return const_cast<Tensor_t&>(
            const_cast<const TensorPtr&>(*this).cast<Tensor_t>()
        );
    }

    ///Allows one to get a read-only version of the tensor
    template<typename Tensor_t>
    const Tensor_t& cast()const
    {
        return static_cast<const Wrapper<Tensor_t>*>(tensor_.get())->tensor;
    }

    /** \brief Implicit conversion to boolean representing whether or not this
     *         pointer is holding something.
     *
     *  \return true if this pointer holds a tensor and false otherwise
     */
    operator bool()const noexcept
    {
        return static_cast<bool>(tensor_);
    }
};

}}//End namespaces
