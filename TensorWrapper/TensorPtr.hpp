#pragma once
#include <utility>
#include <memory>
#include "TensorWrapper/TensorImpl/TensorTypes.hpp"
#include "TensorWrapper/TensorImpls.hpp"
namespace TWrapper {
namespace detail_ {

/** \brief This is a pointer that can hold any type of tensor.
 *
 *  This class works off of type erasure.  Basically  we have a dummy base
 *  class that we hold the data as, then when we need the data we downcast to
 *  a dummy derived class which is templated on the data's type and actually
 *  stores the data.  From this class we can easily retrieve the data.  Note
 *  that this is fully type safe because you need to know the data's type to
 *  get it back.
 *
 */
template<size_t R, typename T>
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
            tensor(std::make_unique<Tensor_t>(t))
      {}

      ///Moves tensor \p t into this instance
      explicit Wrapper(Tensor_t&& t):
          tensor(std::make_unique<Tensor_t>(std::move(t)))
      {}

      ///Returns a deep copy of tensor, avoiding slicing
      std::unique_ptr<Placeholder> clone()const override
      {
          return std::make_unique<Wrapper<Tensor_t>>(*tensor);
      }

      ///The instance in the pointer
      std::unique_ptr<Tensor_t> tensor;

     };

    ///The actual instance is wrapped in this member
    std::unique_ptr<Placeholder> tensor_;

    TensorTypes type_;


    template<typename Tensor_t>
    const Wrapper<Tensor_t>* downcast()const{
        auto rv=dynamic_cast<const Wrapper<Tensor_t>*>(tensor_.get());
        if(!rv)
            throw std::bad_cast();
        return rv;
    }

    template<typename Tensor_t>
    Wrapper<Tensor_t>* downcast()
    {
        return const_cast<Wrapper<Tensor_t>*>(
                    const_cast<const TensorPtr<R,T>*>(*this)->downcast());
    }

    template<TensorTypes T1, TensorTypes T2>
    auto caster()const{
        using tensor_type=typename TensorWrapperImpl<R,T,T2>::type;
        auto temp=static_cast<const Wrapper<tensor_type>*>(tensor_.get());
        return TensorConverter<R,T,T1,T2>::convert(temp);
    }


public:
    ///Makes an empty TensorPtr instance
    TensorPtr()=default;

    ~TensorPtr()=default;

    /** \brief Makes a TensorPtr from a backend's instance.
     *
     *  Whether this is a deep or shallow copy of \p tensor is up to the
     *  tensor library backend.
     *
     *  \param[in] tensor The instance to copy.
     *  \tparam Tensor_t The type of the tensor we are copying.
     */
    template<typename Tensor_t>
    explicit TensorPtr(TensorTypes T1,Tensor_t&& tensor):
        tensor_(std::make_unique<
                    Wrapper<typename std::remove_reference<Tensor_t>::type>
                >(std::forward<Tensor_t>(tensor))),
        type_(T1)
    {}

    ///Deep copies another TensorPtr
    TensorPtr(const TensorPtr& other):
      tensor_(other.tensor_?
              std::move(other.tensor_->clone()):nullptr),
      type_(other.type_)
    {
    }

    /** \brief Assigns a deep copy of other to the current instance.
     *
     *  \param[in] other The instance to  deep copy.
     *  \return The current instance now containing the data of other.
     *  \throws std::bad_alloc if allocation fails.  Strong throw guarantee.
     */
    TensorPtr& operator=(const TensorPtr& other)
    {
        tensor_=std::move(TensorPtr(other).tensor_);
        type_=other.type_;
        return *this;
    }

    /** \brief Takes ownership of other TensorPtr's tensor
     *
     *  \param[in] other The tensor to take ownership of.
     *  \throws No throw guarantee.
     */
    TensorPtr(TensorPtr&&)noexcept=default;

    /** \brief Takes ownership of other TensorPtr's tensor
     *
     *  \param[in] other The tensor to take ownership of.
     *  \throws No throw guarantee.
     */
    TensorPtr& operator=(TensorPtr&&)noexcept=default;

    ///Allows one to get the tensor back, assuming they know its type
    template<TensorTypes T1>
    auto& cast()
    {
        using tensor_type=typename TensorWrapperImpl<R,T,T1>::type;

        return const_cast<tensor_type&>(
            const_cast<const TensorPtr&>(*this).cast<T1>()
        );
    }

    ///Allows one to get a read-only version of the tensor
    template<TensorTypes T1>
    const auto& cast()const
    {
        using tensor_type=typename TensorWrapperImpl<R,T,T1>::type;
        if(type_==T1)//Was the type in here already
        {
            //TODO: change to static cast when I know this works
            auto tensordown=downcast<tensor_type>();
            auto rv=tensordown->tensor.get();
            return *rv;
        }
//        //Need to cast
//        if(type_==TensorTypes::EigenMatrix)
//            return caster<T1,TensorTypes::EigenMatrix>();
//        if(type_==TensorTypes::EigenTensor)
//            return caster<T1,TensorTypes::EigenTensor>();
//        if(type_==TensorTypes::GlobalArrays)
//            return caster<T1,TensorTypes::GlobalArrays>();
        throw std::logic_error("I don't know what crazy tensor you're trying to"
                               " get, but I don't know how to make it.");
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
