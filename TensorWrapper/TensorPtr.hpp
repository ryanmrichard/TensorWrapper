#pragma once
#include <utility>
#include <memory>
#include "TensorWrapper/Operations.hpp"
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

    ///Functor for calling the conversions
    template<TensorTypes T1>
    struct Convert{
        template<TensorTypes T2>
        TensorPtr<R,T> eval(const TensorPtr& ptr)
        {              
            const auto& temp=ptr.template cast<T2>();
            TensorWrapperImpl<R,T,T1> impl;
            TensorWrapperImpl<R,T,T2> impl2;

            /* TODO: remove const_cast when MemoryBlock is made into an iterator
             */

            auto& t=const_cast<typename TensorWrapperImpl<R,T,T2>::type&>(temp);
            auto rv=impl.allocate(impl2.dims(t).dims());
            impl.set_memory(rv,impl2.get_memory(t));
            return TensorPtr<R,T>(T1,std::move(rv));
        }

    };

    ///This class literally exists just to hold the next class
    struct Placeholder{
      ///Ensure we have a virtual destructor to get proper deletion
      virtual ~Placeholder()=default;
      ///This function will implment a polymorphic copy for us
      virtual std::unique_ptr<Placeholder> clone()const=0;
    };

    ///Layer on top of PlaceHolder that actually holds the instance
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

    ///The type of the instance wrapped in \p tensor_
    TensorTypes type_;


    ///Wraps the downcast to clean the code up a bit.
    template<typename Tensor_t>
    const Wrapper<Tensor_t>* downcast()const{
        auto rv=dynamic_cast<const Wrapper<Tensor_t>*>(tensor_.get());
        if(!rv)
            throw std::bad_cast();
        return rv;
    }

    ///\copydoc downcast()const
    template<typename Tensor_t>
    Wrapper<Tensor_t>* downcast()
    {
        return const_cast<Wrapper<Tensor_t>*>(
                    const_cast<const TensorPtr<R,T>*>(*this)->downcast());
    }

public:
    /** \brief Makes an empty TensorPtr instance
     *
     *  This constructor makes a TensorPtr instance that is, roughly speaking,
     *  an overdressed null pointer.
     *
     *  \throws No throw guarantee.
     */
    TensorPtr()noexcept=default;

    /** \brief Releases the memory associated with this instance.
     *
     *
     *
     * \throws If the backend's destructor throws.
     */
    ~TensorPtr()=default;

    /** \brief Makes a TensorPtr from a backend's instance.
     *
     *  Whether this is a deep or shallow copy of \p tensor is up to the
     *  tensor library backend.
     *
     *  \param[in] tensor The instance to copy.
     *
     *  \tparam Tensor_t The type of the tensor we are copying.
     *
     *  \throws If the backend's invoked constructor (either the copy or move
     *  depending on the type of \p tensor) throws.
     */
    template<typename Tensor_t>
    explicit TensorPtr(TensorTypes T1,const Tensor_t& tensor):
        tensor_(std::make_unique<Wrapper<Tensor_t>>(tensor)),
        type_(T1)
    {}

    /** \brief Makes a TensorPtr from a backend's instance.
     *
     */
    template<typename Tensor_t,
             typename =typename std::enable_if<
                          std::is_rvalue_reference<Tensor_t&&>::value
                       >::type
             >
    explicit TensorPtr(TensorTypes T1,Tensor_t&& tensor):
        tensor_(std::make_unique<Wrapper<Tensor_t>>(std::move(tensor))),
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
        tensor_=std::move(other.tensor_->clone());
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

    /** \brief Returns the type of the wrapped tensor.
     *
     *
     *  \returns the type of the tensor
     *
     *  \throws No throw guarantee.
     */
    TensorTypes type()const noexcept{return type_;}

    ///\copydoc cast()const
    template<TensorTypes T1>
    auto& cast()
    {
        using tensor_type=typename TensorWrapperImpl<R,T,T1>::type;
        return const_cast<tensor_type&>(
            const_cast<const TensorPtr&>(*this).cast<T1>()
        );
    }

    /** \brief Allows one to get a read-only version of the tensor back
     *   assuming they know its type.
     *
     *   Storing tensor's from multiple backends is half the battle, one must
     *   also be able to get them back.  This function is half of the machinery
     *   for doing that.  If you know the TensorType of the pointer (at compile
     *   time) then you can call this function to get the tensor back in the
     *   backend's native format.
     *
     *   \returns The instance inside this pointer
     *
     *   \throws std::bad_cast if the tensor inside of this pointer is not of
     *    type T1.  Strong throw guarantee.
     *
     *   \tparam T1 The type of the tensor contained in this pointer.
     *
     */
    template<TensorTypes T1>
    const auto& cast()const
    {
        using tensor_type=typename TensorWrapperImpl<R,T,T1>::type;
        if(T1==type_)//Was the type in here already
        {
            //TODO: change to static cast when I know this works
            auto tensordown=downcast<tensor_type>();
            auto rv=tensordown->tensor.get();
            return *rv;
        }
        throw std::bad_cast();
    }

    template<TensorTypes T1>
    TensorPtr<R,T> convert()const
    {
        return apply_TensorTypes<Convert<T1>>(type_,*this);
    }


    /** \brief Implicit conversion to boolean representing whether or not this
     *         pointer is holding something.
     *
     *  \return true if this pointer holds a tensor and false otherwise
     *
     *   \throws No throw guarantee.
     */
    operator bool()const noexcept
    {
        return static_cast<bool>(tensor_);
    }
};

/* Explicit instantiation of common use cases */
template class TensorPtr<1,double>;
template class TensorPtr<2,double>;
template class TensorPtr<3,double>;

template<size_t R, typename T>
struct Convert<TensorPtr<R,T>> :
        public OperationBase<Convert<TensorPtr<R,T>>>
{

    const TensorPtr<R,T>& data_;

    Convert(const TensorPtr<R,T>& data):
        data_(data)
    {}

    constexpr static size_t rank=R;
    using scalar_type=T;
    using indices=IdxNotSet;

    template<TensorTypes TT>
    const typename TensorWrapperImpl<R,T,TT>::type& eval()const
    {
        return data_.template cast<TT>();
    }
};


}}//End namespaces
