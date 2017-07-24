#pragma once
#include "TensorWrapper/Contraction.hpp"
#include "TensorWrapper/Shape.hpp"
#include "TensorWrapper/MemoryBlock.hpp"
#include "TensorWrapper/TensorPtr.hpp"
#include "TensorWrapper/TensorImpl/TensorTypes.hpp"
#include "TensorWrapper/Operation.hpp"
#include "TensorWrapper/OperationImpls.hpp"
#include "TensorWrapper/Traits.hpp"
#include<random>

namespace TWrapper {
/** \brief  The class that actually does tensor-y stuff.
 *
 *  \tparam R The rank of the tensor
 *  \tparam T The type of the scalars in the tensor.
 */
template<size_t R,typename T>
class TensorWrapperBase{
protected:
    ///A type erased tensor
    detail_::TensorPtr<R,T> tensor_;

    ///That tensor's type
    detail_::TensorTypes ttype_;

    /** \brief Constructor used by derived classes after wrapping a tensor.
     *
     *
     *   \param[in] tensor The wrapped tensor we now own.  Memory allocation
     *                     etc. occurred in instantiating \p tensor.
     *   \param[in] ttype  The enumeration corresponding to the backend to
     *                     use for this tensor.
     *
     *   \throws No throw guarantee.
     */
    TensorWrapperBase(detail_::TensorPtr<R,T>&& tensor,
                      detail_::TensorTypes ttype)noexcept:
        tensor_(std::move(tensor)),
        ttype_(ttype)
    {}

    /** \brief The constructor for when the tensor isn't ready yet.
     *
     *  Unlike the other protected constructor this constructor assumes
     *  the derived class will be building
     *
     */
    TensorWrapperBase(detail_::TensorTypes ttype):
        ttype_(ttype)
    {}

    /** \brief Sets up an operation that can derefence the TensorPtr
     *
     * This function is primarily used to start the operation chaining.
     *
     * \return An operation that when evaluated will return the actual
     *  tensor by reference.
     */
    auto de_ref()const->
        decltype(detail_::make_op<R,T,detail_::DeRef<R,T>>(tensor_))
    {
        return detail_::make_op<R,T,detail_::DeRef<R,T>>(tensor_);
    }

    ///\copydoc de_ref()const
    auto de_ref()->
        decltype(detail_::make_op<R,T,detail_::DeRef<R,T>>(tensor_))
    {
        return detail_::make_op<R,T,detail_::DeRef<R,T>>(tensor_);
    }


    template<detail_::TensorTypes T1>
    using ConvertedType=typename detail_::TensorWrapperImpl<R,T,T1>::type;

    /** \brief Returns the TensorType of the backend used to create this
     *  instance.
     *
     *  This function is largely intended as an implementation detail, but
     *  I wanted to avoid declaring lots of operators as friends of this class.
     *
     *  \return The enumeration corresponding to the backend used to create this
     *   instance.
     *
     *  \throws No throw guarantee.
     */
    detail_::TensorTypes type()const noexcept{return ttype_;}

public:

    template<detail_::TensorTypes T1>
    ConvertedType<T1> eval()const
    {
        return de_ref().template eval<T1>();
    }

    ///\copydoc eval()const
    template<detail_::TensorTypes T1>
    ConvertedType<T1> eval()
    {
       return de_ref().template eval<T1>();
    }

    ///The type of a "rank"-dimensional vector of indices
    using index_t=std::array<size_t,R>;

    /** \brief Constructs a null tensor instance
     *
     *  The resulting instance is essentially a placeholder and can only be made
     *  usable by assigning or moving a legit instance into it.
     *
     *  \throws Never throws.
     */
    TensorWrapperBase()noexcept=default;

    /** \brief Constructs a new instance via a deep copy of \p other
     *
     *  \note This constructor relies on the derived classes having no state
     *        because it slices (in the C++ sense of the word) the class.
     *
     *  \param[in] other The tensor to deep copy.
     *
     *  \throws std::bad_alloc if there is insufficient memory for the copy.
     *          Strong throw guarantee.
     *
     */
    TensorWrapperBase(const TensorWrapperBase&)=default;

    /** \brief Assigns a deep copy of \p other to the current instance.
     *
     *  \param[in] other The tensor to deep copy.
     *  \return The current instance containing a deep copy of other.
     *  \throws std::bad_alloc if the allocation fails.  Strong throw guarantee.
     */
    TensorWrapperBase& operator=(const TensorWrapperBase&)=default;

    /** \brief Takes ownership of another TensorWrapper.
     *
     *  \param[in] other The instance we are taking over.
     *
     *  \throw No throw guarantee.
     */
    TensorWrapperBase(TensorWrapperBase&&)noexcept=default;

    /** \brief Takes ownership of another TensorWrapper freeing up current
     *         resources.
     *
     *  \param[in] other The tensor to take ownership of.
     *  \returns The current instance, now with the contents of other.
     *  \throws No throw guarantee.
     */
    TensorWrapperBase& operator=(TensorWrapperBase&&)noexcept=default;


    /** \brief Destructs the current TensorWrapperBase instance.
     *
     *  All members of this class are managed and thus the default destructor is
     *  fine.  That said, although the TensorWrapper library promises not to
     *  throw in a destructor it may be the case that the backend does throw.
     *
     */
    virtual ~TensorWrapperBase()=default;

    ///@{
    ///Returns the rank of the wrapped tensor
    constexpr size_t rank()const{return R;}

    ///Returns the shape of the wrapped tensor
    virtual Shape<R> shape()const=0;

    ///@}


    /** \brief Returns an element of the tensor. This is a convenience API and
     *         is slow.
     *
     *  For production level code use get_memory/set_memory or slice functions.
     *
     *  \return The requested element.
     */
    virtual T operator()(const index_t& idx)const=0;


    /** \brief Syntactic sugar for accessing an element of the tensor using a
     *         comma separated list of indices rather than an object of type
     *         index_t.
     *
     *  This is a simple wrapper around the other operator() member function.
     *  The signature may look scary, but usage is simple:
     *
     *  \code{.cpp}
     *  TensorWrapper<3,T> tensor;//Assume this initialized already
     *  tensor(0,3,6);//<-Usage.  Returns the element at index {0,3,6}
     *  \endcode
     *
     *  \param[in] elem1 The index along the first dimension to retrieve
     *  \param[in] args The remaining "rank-1" indices to retrieve
     *
     *  \note The first element is split off to avoid this function's signature
     *  reducing to that of the other operator().
     *
     *  \return The value of the tensor at the input index
     */
    template<typename...Args>
    T operator()(size_t elem1,Args...args)const
    {
        static_assert(sizeof...(args)==R-1, "Number of indices != rank");
        return (*this)(index_t{elem1,args...});
    }


    /** \brief Returns an operation that can grab a slice of the current tensor.
     *
     *
     *
     */
    auto slice(const index_t& start,const index_t& end)const->
        decltype(detail_::make_op<R,T,detail_::SliceOp<R,T>>(de_ref(),start,end))
    {
        return detail_::make_op<R,T,detail_::SliceOp<R,T>>(de_ref(),start,end);
    }


    /** \brief Together with set_memory allows reading and writing to the local
     *         elements of the tensor
     *
     *   Reading and writing to a tensor is complicated as the backends hold the
     *   data in all sorts of interesting ways.  This call returns an API to
     *   the memory the current process is responsible for.  For tensor backends
     *   where all of the tensor is local this is the entire tensor, whereas for
     *   distributed tensor backends this is only the slice you hold locally.
     *   Whereas the slice API returns a new tensor, this call, roughly
     *   speaking, returns the raw pointer.  That is to say writes to the
     *   resulting MemoryBlock instance will (see note below) actually change
     *   the memory of the current TensorWrapperBase instance.  You should not
     *   write to the resulting instance unless you intend for your changes to
     *   modify the tensor (use slice if you do not want your writes to do
     *   this).  Finally, since this call always returns the local memory it
     *   can not be used to write to remote parts of the tensor; to do this
     *   you'll need to make your own MemoryBlock instance, write to it and then
     *   call set_memory.
     *
     *   \note If the backend is local your writes will likely take effect
     *   immediatly; however, depending on the backend, this may not be true for
     *   distributed tensors.  This is why you should always call set_memory.
     *   As long as your backend is hooked in well, the set_memory call will
     *   determine whether it really needs to copy the elements over.
     *
     *   \returns A slightly wrapped raw pointer to the data inside the tensor.
     *
     */
    virtual MemoryBlock<R,T> get_memory()=0;

    virtual void set_memory(const MemoryBlock<R,T>& other)=0;

    template<typename RHS_t>
    auto operator+(RHS_t&& lhs)const->
    decltype(de_ref()+(std::forward<RHS_t>(lhs)))
    {
        return de_ref()+std::forward<RHS_t>(lhs);
    }

    template<typename RHS_t>
    auto operator-(RHS_t&& rhs)const->
        decltype(de_ref()-std::forward<RHS_t>(rhs))
    {
        return de_ref()-std::forward<RHS_t>(rhs);
    }

    auto operator*(T&& value)const->decltype(de_ref()*std::forward<T>(value))
    {
        return de_ref()*std::forward<T>(value);
    }

//    ///API for contraction
//    template<size_t N> constexpr
//    IndexedTensor<Rank,wrapped_t> operator()(const char(&idx)[N])const
//    {
//        return IndexedTensor<Rank,wrapped_t>(tensor_,idx);
//    }
};

template<size_t R, typename T>
TensorWrapperBase<R,T>& FillRandom(TensorWrapperBase<R,T>& tensor)
{
         auto mem=tensor.get_memory();
         std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<> dis(0,10);
         for(auto idx : mem.local_shape)
             mem(idx)=dis(gen);
         tensor.set_memory(mem);
         return tensor;
}


template class TensorWrapperBase<1,double>;
template class TensorWrapperBase<2,double>;
template class TensorWrapperBase<3,double>;

}//End namespace

template<typename T, size_t R>
auto operator*(T&& value, const TWrapper::TensorWrapperBase<R,T>& t)->
  decltype(t*std::forward<T>(value))
{
    return t*std::forward<T>(value);
}


