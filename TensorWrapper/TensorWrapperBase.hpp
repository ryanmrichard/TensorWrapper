#pragma once
#include "TensorWrapper/TensorPtr.hpp"
#include "TensorWrapper/MemoryBlock.hpp"
#include<random>

namespace TWrapper {

template<typename...Args>
struct IndexHelper{
    static constexpr bool value=detail_::AllUnique<Args...>::value;
};

/** \brief  The common base class of all the various implementations.
 *
 *  By time we are in this class we can't actually return values because we
 *  have lost the details of how to compute them.  Rather, the base class
 *  returns an operation that can be evaluated to get the result.  Typically
 *  this evaluation happens in the constructor/assignment operation of a
 *  TensorWrapper instance.
 *
 *  \note Copy/Move construction/assignment functions are protected to avoid
 *  slicing.  Deep copies can be made from the base using the clone member
 *  function.
 *
 *  \tparam R The rank of the tensor
 *  \tparam T The type of the scalars in the tensor.
 */
template<size_t R,typename T>
class TensorWrapperBase{
protected:
    ///The type of a type erased tensor
    using pTensor=detail_::TensorPtr<R,T>;

    ///The type erased tensor
    pTensor tensor_;

    ///The type of an operation that converts the type-erased pointer
    template<typename Index_t=typename detail_::GenericIndex<R>::type>
    auto de_ref_op()const
    {
        using Convert_t=detail_::Convert<pTensor>;
        return detail_::IndexedTensor<T,Convert_t,Index_t>(
                    Convert_t(tensor_,shape().dims()));
    }

    ///The type of an operation that converts the type-erased pointer and traces
    template<typename Index_t>
    auto de_ref_with_tr()const
    {
        using Nested_t=decltype(de_ref_op<Index_t>());
        //return de_ref_op<Index_t>();
        return detail_::Trace<Nested_t>(de_ref_op<Index_t>());
    }

    template<bool good,typename Index_t>
    struct DeRefHelper;

    template<typename Index_t>
    struct DeRefHelper<true,Index_t>{
        template<typename This_t>
        static auto eval(const This_t* t)
        {
            return t->de_ref_op<Index_t>();
        }
    };

    template<typename Index_t>
    struct DeRefHelper<false,Index_t>{
        template<typename This_t>
        static auto eval(const This_t* t)
        {
            return t->de_ref_with_tr<Index_t>();
        }
    };

    /** \brief Constructor used by derived classes after wrapping a tensor.
     *
     *
     *   \param[in] tensor The wrapped tensor we now own.  Memory allocation
     *                     etc. occurred in instantiating \p tensor.
     *
     *   \throws None No throw guarantee.
     */
    TensorWrapperBase(pTensor&& tensor)noexcept:
        tensor_(std::move(tensor))
    {}

    /** \brief Deep copies another instance.
     *
     *   \throws std::bad_alloc if memory allocation fails.
     */
    TensorWrapperBase(const TensorWrapperBase&)=default;

    /** \brief Takes ownership of another instance
     *
     *
     */
    TensorWrapperBase(TensorWrapperBase&&)noexcept=default;



    TensorWrapperBase& operator=(const TensorWrapperBase&)=default;

    TensorWrapperBase& operator=(TensorWrapperBase&&)noexcept=default;


    //For the moment I do not want to expose the TensorPtr class
    //this allows Convert to grab it
    friend class detail_::Convert<TensorWrapperBase<R,T>>;
public:

    ///The type of a "rank"-dimensional vector of indices
    using index_t=std::array<size_t,R>;

    /** \brief Constructs a null tensor instance
     *
     *  The only member of this class is the pointer to the tensor
     *  implementation.  For all intents and purposes that instance will point
     *  to nullptr after a call to this constructor.
     *
     *  \throws None No throw guarantee.
     */
    TensorWrapperBase()noexcept=default;

    virtual std::unique_ptr<TensorWrapperBase<R,T>> clone()const=0;


    /** \brief Destructs the current TensorWrapperBase instance.
     *
     *  All members of this class are managed and thus the default destructor is
     *  fine.
     *
     *  \throws ??? The TensorWrapper library does not throw during
     *  destruction; however, the implementations we are wrapping may.  If the
     *  wrapped implementation throws during destruction this function will too.
     */
    virtual ~TensorWrapperBase()=default;

    ///Returns the rank of the wrapped tensor
    static constexpr size_t rank(){return R;}

    ///Returns the shape of the wrapped tensor
    virtual Shape<R> shape()const=0;

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
     *  TensorWrapper<3,T> tensor;//Assume this is initialized already
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


    /** \brief Starts the lazy evaluation chain when first operation is addition
     *
     *  This function allows for two tensors to be added without specifying
     *  their indices.  For example via:
     *
     *  \code
     *  TensorWrapper A,B,C;
     *  C=A+B;
     *  \endcode
     *
     *  If indices are not included, it is assumed that all tensors have the
     *  same indices.  For example the above code is interpreted as being the
     *  same as:
     *
     *  \code
     *  TensorWrapper A,B,C;
     *  auto i=make_index("i");
     *  auto j=make_index("j");
     *  C=A(i,j)+B(i,j);
     *  \endcode
     *
     *  \param[in] rhs The tensor or tensor expression to add to this tensor.
     *
     *  \tparam RHS_t The type of \p rhs.  Expected to be a TensorWrapper or
     *          TensorWrapperBase instance for the same backend or a lazy
     *          evaluation expression involving tensors satisfying this
     *          condition.
     *
     */
    template<typename RHS_t>
    auto operator+(const RHS_t& rhs)const
    {
        return de_ref_op()+rhs;
    }

    template<typename RHS_t>
    auto operator-(const RHS_t& rhs)const
    {
        return de_ref_op()-rhs;
    }

    auto operator*(T rhs)const
    {
        return de_ref_op()*rhs;
    }



    /** \brief API for associating a set of indices with a tensor
     *
     */
    template<char...Chars,typename...Args>
    constexpr auto operator()(const detail_::C_String<Chars...>&,Args...)const
    {
        static_assert(sizeof...(Args)+1==R,"#indices != rank");
        using Index_t=detail_::make_indices<detail_::C_String<Chars...>,Args...>;
        return DeRefHelper<IndexHelper<Index_t>::value,Index_t>::eval(this);
    }
};

/** \relates TensorWrapper
 *
 *  \brief Fills a TensorWrapper instance with random numbers.
 *
 *  \param[in] tensor The tensor to fill with random numbers.
 *  \returns The tensor now with random numbers.
 */
template<size_t R, typename T>
TensorWrapperBase<R,T>& fill_random(TensorWrapperBase<R,T>& tensor)
{
         auto mem=tensor.get_memory();
         std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<> dis(0,10);
         for(size_t i=0;i<mem.nblocks();++i)
         {
             auto idx=mem.begin(i),end=mem.end(i);
             size_t counter=0;
             T* buffer=mem.block(i);
             while(idx!=end)
             {
                 buffer[counter++]=dis(gen);
                 ++idx;
             }
         }
         tensor.set_memory(mem);
         return tensor;
}


template class TensorWrapperBase<1,double>;
template class TensorWrapperBase<2,double>;
template class TensorWrapperBase<3,double>;

namespace detail_ {

template<size_t R, typename T>
struct Convert<TensorWrapperBase<R,T>> : public
        OperationBase<Convert<TensorWrapperBase<R,T>>>
{
    using scalar_type=T;
    constexpr static size_t rank=R;
    using indices=IdxNotSet;
    const TWrapper::detail_::TensorPtr<R,T>& data_;
    Convert(const TensorWrapperBase<R,T>& data):
        data_(data.tensor_)
    {}

    template<TensorTypes TT>
    auto eval()const->decltype(data_.template cast<TT>())
    {
        return data_.template cast<TT>();
    }
};
}



}//End namespace

template<typename T, size_t R>
auto operator*(T value, const TWrapper::TensorWrapperBase<R,T>& t)
{
    return t*value;
}


