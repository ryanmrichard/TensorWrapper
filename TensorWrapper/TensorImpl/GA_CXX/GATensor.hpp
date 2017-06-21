#pragma once
#include<ga.h>
#include<utility>
#include<complex>
#include<memory>
#include<array>
#include<vector>
#include<numeric>
#include<algorithm>
namespace TWrapper {

/** \brief Imparts a RAII model to intializing and finalizing Global Arrays
 *
 *  Here's how this works, there exists a global unique pointer to a class
 *  GAEnv (the details of which are completely irrelevant) in GATensorImpl.
 *  At first this pointer is null, when this function is called it allcates
 *  the unique_ptr.  When the pointer goes out of scope GA is finalized.  So
 *  long as GAInitialize is called at some point in the program all is well.
 *  Subsequent calls to GAInitialize will have no effect.
 *
 *  \param[in] heap The minimum size of the heap, -1 lets GA decide
 *  \param[in] stack The minimum size of the stack, -1 lets GA decide
 */
void GAInitialize(int heap=-1, int stack=-1);

/** \brief A C++ wrapper to the Global Arrays library
 *
 *  Global Arrays, has C++ bindings, but they are just very thin wrappers around
 *  the C bindings.  This class is my take on a more proper Tensor class powered
 *  by Global Arrays.
 *
 *  A couple of things to note:
 *
 *  As far as I can tell GA seems to be designed for contiguous blocks as far as
 *  indices go
 *  that is to say that the elements in the flattened buffer are something like
 *  (0,1), (0,2), ... and not (0,1), (0,11), (0,21)... that is the fast index has
 *  period 1 and not say 10.  Regardless of whether or not GA supports block
 *  cyclic or not, I assume the blocks have period 1.
 *
 *  In C++ it is canonical to specify ranges [begin,end), that is ranges are
 *  inclusive on the starting element and exclusive on the ending element.  When
 *  this class asks for a range I expect a C++-like range.  Conversion to GA's
 *  range standard, i.e. [begin,end], happens under the hood.
 *
 *  On the topic of ranges, in C++ ranges that can't be negative are always
 *  given in terms of an integral type size_t, which for all intents and
 *  purposes can be thought of as unsigned long int (there's some technical
 *  details prohibiting one from actually calling size_t unsigned long int, but
 *  we can ignore those if we just agree to use size_t).
 *
 *  At the moment the shape of an instance can't be reshaped.  If you want to
 *  change its shape you'll have to make a new instance.
 *
 *  This class uses RAII, what this means is when you make an instance it
 *  obtains the memory it will need.  When the instance goes out of scope it
 *  releases the memory.  In other words, you're not responsible for the memory
 *  managed by this class.
 *
 *  For functions making copies we follow the usual C++ paradigm that such
 *  copies are deep copies.  That is the resulting GATensor is not simply an
 *  alias for the copied tensor, but rather has its own allocated memory, GA
 *  handle, and lifetime.
 *
 *  For functions that have a template type parameter Cont_t, this can be any
 *  STL container that stores its elements contigouously (i.e. vector, array,
 *  deque, etc.)
 *
 *  Despite the GA library supporting arbitrary rank tensors, most of it
 *  actually focuses on matrices.  As a result in a few places (notably for
 *  multiplication and transpose) I have used a nomeclature indicative of that.
 *  Should more general support be added then these names should change (to
 *  contract and permute/shuffle respectively).
 *
 *  Implementation in more detail.
 *
 *  To keep the class declaration relatively clean the actual implementation is
 *  done in an auxillary header file GATensorImpl.hpp.
 *
 *  We need a null value for the handle, as I'm not sure of what values GA may
 *  use we do this with a pointer, basically if the pointer is null the instance
 *  is not allocatd.
 *
 *  GA does add, subtract, etc. very BLAS like.  These calls are where scaling
 *  of the tensor are actually done.  Until one of these calls is called we
 *  just carry the scale factor along for the ride.  If that call is assigning,
 *  then we reset the scale factor.  The current scale factor is automatically
 *  applied at the get_value function.  The only tricky situation is what
 *  happens when set_values is called for a scale factor that is not equal to 1?
 *  Since as far as the user is concerned we have already scaled the array we
 *  go ahead and tell GA to scale the array and then set the values.
 *
 *  We should do a similar trick for the transpose, but at the moment I do not
 *  as this moves more into expression templating than I care to do at the
 *  moment.  Basically how this would work is that calling transpose
 *  flips  a flag.  The transpose is then only applied when needed.
 *
 *  \tparam rank The rank of the tensor
 *  \tparam T The type of an element
 */
template<size_t rank,typename T>
class GATensor{
public:

    /** \brief Makes a placeholder GATensor
     *
     *  Many algorithms assume that a class is default constructable simply to
     *  get a placeholder variable.  This constructor will create a GATensor
     *  that is not useable aside from being a placeholder.  To make the
     *  instance usable one must assign an already useable instance to it.
     *
     */
    GATensor()=default;

    /** \brief Makes a GATensor from an existing handle
     *
     *  There's not much to say about this constructor aside that invoking it
     *  transfers ownersip of the handle to this class.  More specifically
     *  freeing and allocating memory in the tensor with the given handle is
     *  now managed by the resulting instance.  This includes the handle itself,
     *  i.e. when this instance goes out of scope the handle will be freed. This
     *  could be changed in the future, if requested.
     *
     *  \param[in] handle The handle this instance will take control of.
     *
     */
    explicit GATensor(int handle):handle_(std::make_unique<int>(handle)){}

    /** \brief The most general constructor for a GATensor
     *
     *   This constructor provides full flexibility in initializing a GATensor.
     *   GA works by distributing blocks of the entire tensor on each node. To
     *   do this it needs to know how big the full tensor is.  It can then
     *   automatically determine how much of the tensor to put on each node.
     *   For most algorithms this will not be the optimal distribution of the
     *   tensor thus this constructor also allows you to provide hints about
     *   how large these blocks should be.  Consult the GA documentation for
     *   more information about how these hints are used.
     *
     *   \param [in] dims A "rank" element container where element "i" is the
     *                    length of dimension "i", "i" in the range [0,rank).
     *                    These are the lengths of the full tensor.
     *   \param [in] chunks A "rank" element container where element "i" is the
     *                    minimum (TODO: verify) length of a block's dimension
     *                    "i", "i" in the range [0,rank)
     *   \param[in] name An optional name to call the tensor.  Only used for
     *                   printing.
     *
     *   \tparam Cont_t1 The type of the container for \p dims
     *   \tparam Cont_t2 The type of the container for \p chunks
     */
    template<typename Cont_t1, typename Cont_t2>
    GATensor(const Cont_t1& dims, const Cont_t2& chunks,
             const char * name=nullptr);

    /** \brief A basic constructor for a GATensor.  Only needs the sizes.
     *
     *   This constructor is similar to the most general constructor aside from
     *   the block sizes will be determined automatically.
     *
     *   \param [in] dims A "rank" element container where element "i" is the
     *                    length of dimension "i", "i" in the range [0,rank)
     *   \param [in] name An optional name to call the tensor.  Only used for
     *                    printing.
     *   \tparam Dims_t The type of the container for \p dims
     */
    template<typename Cont_t>
    GATensor(const Cont_t& dims,const char * name=nullptr):
        GATensor(dims,Cont_t(),name){}

    /** \brief Basic constructor that additionally fills the tensor.
     *
     *   Given the dimensions of the resultign tensor goes ahead and sets
     *   all elements to a particular value.
     *
     *   \param[in] dims A "rank" element container where element "i" is the
     *                    length of dimension "i", "i" in the range [0,rank)
     *   \param[in] value The value
     *   \param[in] name An optional name to call the tensor.  Only used for
     *                    printing.
     *   \tparam Dims_t The type of the container for \p dims
     */
    template<typename Cont_t>
    GATensor(const Cont_t& dims,T value,const char* name=nullptr):
        GATensor(dims,name)
    {
        fill(value);
    }

    /** \brief Constructs an instance by deep copynig the tensor \p other.
     *
     *  \param[in] other The GATensor to copy.  Must have the same rank and type
     *                   of elements.
     */
    GATensor(const GATensor<rank,T>& other);

    /** \brief Assigns a deep copy of \p other to this instance.
     *
     *  \param[in] other The GATensor to copy.  Must have the same rank and type
     *                   of elements.
     *
     *  \return This instance now populated by a deep copy of \p other.
     */
    GATensor& operator=(const GATensor<rank,T>& other);

    /** \brief Takes ownership of another GATensor
     *
     *   \param[in] other The GATensor instance we are taking ownership of.
     *                    After this call \p other is in a valid, but
     *                    unusable state, i.e. let it go out of scope
     *
     */
    GATensor(GATensor<rank,T>&& other);

    /** \brief Assigns another GATensor to the current instance
     *
     *   Take note that the tensor wrapped by the current instance will
     *   be deallocated before this instance takes control of the other
     *   instance.
     *
     *   \param[in] other The GATensor instance we are taking ownership of.
     *                    After this call \p other is in a valid, but
     *                    unusable state, i.e. let it go out of scope
     *
     *   \return The current instance
     */
    GATensor& operator=(GATensor<rank,T>&& other);

    /** \brief Frees the current instance, releasing any of its acquired
     *         resources.
     *
     *
     */
    ~GATensor();

    /** \brief Sets a single index
     *
     *  This function will set a single index.  There is no restriction on where
     *  the memory for that index may reside.  This is here for convenience, it
     *  should probably never be used in production code.  Instead see the next
     *  overload.
     *
     *  \param[in] idx A "rank" element long container where element "i" is the
     *                 offset along the "i"-th dimension of the tensor.
     *  \param[in] value The value that will be assigned to the element with the
     *                   index \p idx.
     *
     *  \tparam Cont_T The type of the container
     */
    template<typename Cont_T>
    void set_values(const Cont_T& idx, T value){
        std::array<size_t,rank> high(idx);
        std::transform(high.begin(),high.end(),
                       high.begin(),[](T val){return ++val;});
        set_values(idx,high,&value);
    }

    /** \brief Sets a block from a buffer, uses normal C++ [lo,hi) range
     *
     *  This is the set call that should be used in a production level code
     *  (along with the local_shape function).  Like the other overload, this
     *  will allow you to set an arbitrary chunk of the tensor; however, unlike
     *  the other call this can be done with a minimal amount of communication.
     *
     *  Usage:
     *    Say we want to set the data block that has (x,y) as the top
     *    left element and (z,w) as its bottom right element.  In this
     *    case low would be a container with elements "x" and "y" (in that
     *    order) and high would be a container with elements "z+1" and "w+1"
     *    (again in that order).
     *
     *  \param[in] low A "rank" long list of the starting indices of for the
     *                 block of data to set.  Element "i" is the start of the
     *                 block along the "i"-th dimension
     *  \param[in] high A "rank" long list of the starting indices of for the
     *                  block of data to set.  Element "i" is the end of the
     *                  block along the "i"-th dimension.
     *  \param[in] buffer The data that will be copied into the tensor.
     *
     *
     *  \tparam Cont_t1 The type of the container used to store the low part of
     *                  the range.
     *  \tparam Cont_t2 The type of the container used to store the high part of
     *                  the range.
     */
    template<typename Cont_t1, typename Cont_t2>
    void set_values(const Cont_t1& low,
                    const Cont_t2& high,
                    const T* buffer);

    /** \brief Sets all elements of the tensor to a value
     *
     *  \param[in] value The value each element of the tensor will be set to
     */
    void fill(T value);

    /** \brief Returns a single value
     *
     *  Like the single element set_value function, this function is provided
     *  as a convenience.  It should not be used in production level code.  The
     *  index may be anywhere in the tensor.
     *
     *  \param[in] idx A "rank" long list of where element "i" is the offset
     *                 along the "i"-th dimension of the tensor.
     *
     *  \return The requested index
     *
     *  \tparam Cont_t The type of the container used for the index.
     *
     */
    template<typename Cont_t>
    T get_values(const Cont_t& idx)const{
        std::array<size_t,rank> high(idx);
        std::transform(high.begin(),high.end(),
                       high.begin(),[](T val){return ++val;});
        return get_values(idx,high)[0];
    }

    /** \brief Returns a chunk of the tensor
     *
     *  This is the get call that should be used in a production level code
     *  (along with the local_shape function).  Like the other overload, this
     *  will allow you to get an arbitrary chunk of the tensor; however, unlike
     *  the other call this can be done with a minimal amount of communication.
     *
     *  Usage:
     *    Say we want to get the data block that has (x,y) as the top
     *    left element and (z,w) as its bottom right element.  In this
     *    case low would be a container with elements "x" and "y" (in that
     *    order) and high would be a container with elements "z+1" and "w+1"
     *    (again in that order).
     *
     *  \param[in] low A "rank" long list of the starting indices of for the
     *                 block of data to get.  Element "i" is the start of the
     *                 block along the "i"-th dimension
     *  \param[in] high A "rank" long list of the starting indices of for the
     *                  block of data to get.  Element "i" is the end of the
     *                  block along the "i"-th dimension.
     *  \return The requested block of the tensor.
     *
     *  \tparam Cont_t1 The type of the container used to store the low part of
     *                  the range.
     *  \tparam Cont_t2 The type of the container used to store the high part of
     *                  the range.
     */
    template<typename Cont_t1, typename Cont_t2>
    std::vector<T> get_values(const Cont_t1& low, const Cont_t2& high)const;

    ///The type of a pair of starting and stopping indices
    using slice_range_t=std::pair<std::array<size_t,rank>,
                                  std::array<size_t,rank>>;

    /** \brief Returns the starting and stopping indices of the slice of the
     *         tensor held in local memory.
     *
     *
     *   In a distributed environment only part of the tensor will actually be
     *   in the memory of the current node.  This function tells you want range
     *   that is.  This is useful for getting/setting as it allows you to only
     *   work with local memory.
     *
     *  \return Two "rank" element arrays where element "i" in the first array
     *          is the first offset along the "i" direction held in local memory
     *          and element "i" in the second array is the first element not
     *          held in local memory.
     */
    slice_range_t my_slice()const;

    /** \brief Returns the transpose of the current matrix
     *
     *  This operation should actually be done via lazy evalution, by flipping
     *  the transpose flag in some form of wrapper class and then waiting until
     *  we need to actually do the transpose.  As implementing expression
     *  templating is beyond the scope of the current class, we simply actually
     *  do the transpose here.
     *
     *  \note Although, we could do the check for rank>2 at compile time, doing
     *        so would significantly limit the utility of this class.
     *
     *  \throws If transpose is not defined for this tensor's rank.
     *
     *  \return The current matrix, transposed.
     */
    GATensor<rank,T> transpose()const;

    /** \brief Adds two GATensors
     *
     *  Note, in modern C++ tensor libraries addition is typically handled by
     *  expression templates.  This removes the need for temporaries and also
     *  improves cache reuse.  GA's add function is pairwise and thus we will
     *  not obtain any performance gains by writing this operation as an
     *  expression template.
     *
     *  \param[in] rhs The tensor to add to this.
     *  \return Ths sum of this and \p rhs.
     */
    GATensor<rank,T> operator+(const GATensor<rank,T>& rhs)const;

    /** \brief Accumulates another GATensor into this one.
     *
     *  Note, in modern C++ tensor libraries addition is typically handled by
     *  expression templates.  This removes the need for temporaries and also
     *  improves cache reuse.  GA's add function is pairwise and thus we do will
     *  not obtain any performance gains by writing this operation as an
     *  expression template.
     *
     *  \param[in] rhs The tensor to add to this.
     *  \return The current tensor after accumulation
     */
    GATensor<rank,T>& operator+=(const GATensor<rank,T>& rhs);

    /** \brief Subtracts two GATensors
     *
     *  \note This function is implemented in terms of GA's add function with
     *  a negative scale factor.
     *
     *  Note, in modern C++ tensor libraries subtraction is typically handled by
     *  expression templates.  This removes the need for temporaries and also
     *  improves cache reuse.  GA's add function is pairwise and thus we do will
     *  not obtain any performance gains by writing this operation as an
     *  expression template.
     *
     *  \param[in] rhs The tensor to subtract from this.
     *  \return Ths difference of this and \p rhs.
     */
    GATensor<rank,T> operator-(const GATensor<rank,T>& rhs)const;

    /** \brief Negative accumulates another GATensor into this one.
     *
     *  \note This function is implemented in terms of GA's add function with
     *  a negative scale factor.
     *
     *  Note, in modern C++ tensor libraries subtraction is typically handled by
     *  expression templates.  This removes the need for temporaries and also
     *  improves cache reuse.  GA's add function is pairwise and thus we do will
     *  not obtain any performance gains by writing this operation as an
     *  expression template.
     *
     *  \param[in] rhs The tensor to subtract from this.
     *  \return The current tensor after accumulation
     */
    GATensor<rank,T>& operator-=(const GATensor<rank,T>& rhs);

    ///Scalar multiplication
    GATensor<rank,T> operator*(T value)const{
        GATensor temp(*this);
        temp.scale_*=value;
        return temp;
    }

    ///Scalar multiplication with assignment
    GATensor<rank,T>& operator*=(T value){
        scale_*=value;
        return *this;
    }

    /** \brief Matrix multiplication
     *
     *  This function is for multiplying tensors of rank 2.  It could be
     *  specialized to other ranks, but I think that is best handled by a more
     *  general contract function.
     *
     */
    typename std::enable_if<rank==2,GATensor<2, T>>::type
    operator*(const GATensor<2,T>& rhs)const;

    /** \brief Returns true if the underlying tensor is exactly equal to rhs
     *
     *   This call will perform an element by element comparision to ensure
     *   that the two tensors are the same.  There is no tolerance on this scan,
     *   i.e. usual double comparision caveats apply.  The primary
     *   template accounts for all cases where rhs's rank is not equal to
     *   this tensor's rank.  The specialization to equal ranks actually
     *   performs the equality comparison.
     *
     *   \param[in] rhs The other tensor we are comparing too
     *
     *   \return True if all elements in this tensor are equal to those inside
     *                \p rhs.
     *
     *   \tparam rank2 The rank of the tensor on the right side
     */
    bool operator==(const GATensor<rank,T>& rhs)const;

    template<size_t rank2>
    bool operator==(const GATensor<rank2,T>& /*rhs*/)const
    {
        return false;
    }

    ///Returns the full dimensions of the tensor
    std::array<size_t,rank> dims()const{return dims_;}

    /** \brief Returns the rank of the tensor
     *
     *  Since the rank is a compile time constant we make this a static
     *  function to facilitate eventual template metaprogramming.
     *
     *  \return The rank of the current tensor
     */
    static constexpr size_t get_rank(){return rank;}

    ///Prints the tensor for debuggin'
    void print_out() const;


private:
    ///The handle of the underlying tensor
    std::unique_ptr<int> handle_;

    ///The dimensions of the full tensor
    std::array<size_t,rank> dims_;

    ///The minimum size of each block (stored for copying purposes)
    std::array<size_t,rank> chunks_;

    ///The current scale factor
    T scale_;
};

#include "TensorWrapper/TensorImpl/GA_CXX/GATensorImpl.hpp"

}//End namespace TWrapper

template<size_t rank, typename T, typename T1>
typename std::enable_if<std::is_fundamental<T1>::value,
                        TWrapper::GATensor<rank,T>>::type
operator*(T1 lhs, const TWrapper::GATensor<rank,T>& rhs)
{
    return rhs*lhs;
}
