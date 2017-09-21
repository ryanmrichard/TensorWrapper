#pragma once
#include "TensorWrapper/Shape.hpp"
#include <vector>
#include <functional>
#include <memory>
namespace TWrapper {

/** \brief A class for wrapping the memory access semantics of the underlying
 *         tensor implementation.
 *
 *  At the end of the day all tensors store their elements contigiously to some
 *  extent.  What we mean by this is the entire tensor can be thought of as
 *  being composed of many blocks of contigious memory (up to the possiblility
 *  that each element constitutes one such block).  Particularly when it comes
 *  to initialization of the tensors it becomes necessary to access the data
 *  stored locally.  That's what this class wraps.
 *
 *  \note To access data, regardless of whether it is local or remote, use the
 *  slice API.
 *
 *  Typical usage of this class is something like:
 *  \code
 *  //Request a tensor's local memory
 *  auto mem = tensor.get_memory();
 *
 *  //Loop over all the contigious blocks of memory stored locally
 *  for(size_t blocki=0;mem.nblocks();++blocki)
 *  {
 *      //Get the pointer to the i-th block's memory
 *      T* buffer=mem.block(blocki);
 *
 *      //Get the shape of the block
 *      auto shapei=mem.shape(blocki);
 *
 *      //Fill the block using the shape for the data layout
 *      size_t counter=0;
 *      for(const auto& idx : shapei)
 *      {
 *          buffer[counter++]=//set to data for idx;
 *      }//End loop over block's indices
 *
 *  }//End loop over blocks
 *
 *  //Give the newly set memory back to the tensor
 *  tensor.set_memory(mem);
 *
 *  //Consider all pointers in mem invalidated from now on
 *  \endcode
 *
 *  Implementation notes.
 *
 *  I have decied to delete the copy constructor and copy assignment operators
 *  because their semantics are somewhat non-intuitive.  From the user's
 *  perspective this class is a uniform API to the raw memory so copying should
 *  be shallow to ensure that two instances of this class which are associated
 *  with the same tensor really are providing an API to the same memory.  On the
 *  other hand, if this class is managing memory then we would be expected to
 *  shallow copy everything, but the unique pointers.  This however would
 *  produce a data race in that if the original instance is deleted the pointers
 *  of all copies would be invalidated.
 *
 *
 *  \tparam rank The rank of the block we wrap
 *  \tparam T  The type of an element in the block we wrap
 *
 */
template<size_t rank, typename T>
class MemoryBlock{
public:
    ///The type of an index
    using index_type=std::array<size_t,rank>;

    ///The type of an iterator over the managed blocks
    using const_iterator=IndexItr<rank>;

    /** \brief Makes a new MemoryBlock instance that contains no blocks of
     *  memory.
     *
     *  \throws None.  No throw guarantee.
     */
    MemoryBlock()noexcept{}

    //See class documentation for why these are deleted
    MemoryBlock(const MemoryBlock<rank,T>&)=delete;
    MemoryBlock<rank,T>& operator=(const MemoryBlock&)=delete;

    /** \brief Makes a new MemoryBlock by taking the contents of another
     *   instance.
     *
     *  \param[in] other The instance to take the state of.
     *  \throws None.  No throw gurantee.
     */
    MemoryBlock(MemoryBlock<rank,T>&& other)noexcept:
        buffers_(std::move(other.buffers_)),
        memory_(std::move(other.memory_)),
        shapes_(std::move(other.shapes_))
    {}

    /** \brief Makes a new MemoryBlock by taking the contents of another
     *   instance.
     *
     *  \param[in] other The instance to take the state of.
     *
     *  \returns The current instance after taking the state of \p other.
     *  \throws None.  No throw gurantee.
     */
    MemoryBlock<rank,T>& operator=(MemoryBlock&& other)noexcept
    {
        buffers_=std::move(other.buffers_);
        memory_=std::move(other.memory_);
        shapes_=std::move(other.shapes_);
        return *this;
    }

    /** \brief Frees up resources of the class.
     *
     *  If this class owned the memory all pointers to it are now invalidated.
     *  \throws None. No throw guarantee.
     */
    ~MemoryBlock()noexcept{}

    /** \brief Adds a new block to the instance.
     *
     * This overload is for the case when the tensor is able to return a pointer
     * to the raw memory.
     *
     * \param[in] mem A pointer to the contigious memory.
     * \param[in] shape A Shape instance describing the layout of the memory

     * \throws std::bad_alloc if resisizing the internal \p std::vectors fails.
     *         Weak throw guarantee.
     *
     */
    void add_block(T* mem,
                   const Shape<rank>& shape)
    {
        buffers_.push_back(mem);
        shapes_.push_back(shape);
    }

    /** \brief Adds a new block to the instance transferring memory ownership.
     *
     *  This overload is to be used when the tensor backend does not expose the
     *  raw memory buffer.  In that event it is expected that the routine adding
     *  a block to this instance will allocate a contigious chunk of memory in
     *  a std::unique_ptr and this instance will take ownership of it.
     *
     *  \param[in] mem The memory we are taking ownership of.
     *  \param[in] shape A Shape instance describing the layout of the memory.
     *  \param[in] start The index of the overall tensor that the first element
     *                  of \p mem maps to.
     * \param[in] end The index in the overall tensor that the last element of
     *                \p mem maps to.
     * \throws std::bad_alloc if resisizing the internal \p std::vectors fails.
     *         Weak throw guarantee.
     */
    void add_block(std::unique_ptr<T[]>&& mem, const Shape<rank>& shape)
    {
        T* _mem=mem.get();
        memory_.push_back(std::move(mem));
        add_block(_mem,shape);
    }

    /** \brief Returns the number of blocks currently managed by this instance.
     *
     * \returns The number of blocks currently managed by this instance.
     * \throw None.  No throw guarantee.
     */
    size_t nblocks()const noexcept{return buffers_.size();}

    /** \brief Returns a pointer to the raw memory in which data should be
     *  placed.
     *
     *  \param[in] blocki The block of data you want. \p blocki should be in the
     *             range [0,nblocks()); however, no check is made to ensure that
     *             this is the case and values of \p blocki outside this range
     *             will result in undefined behavior.
     *  \returns A pointer to the raw memory in which data should be placed.
     *  \throws None. No throw guarantee.
     */
    T* block(size_t blocki)const noexcept{return buffers_[blocki];}

    /** \brief Returns an iterator to the first index within a specified block.
     *
     *  \param[in] blocki The block whose indices are to be iterated over.
     *             \p blocki should be in the range [0,nblocks()); however, no
     *             check is made to ensure that this is the case and values of
     *             \p blocki outside this range will result in undefined
     *             behavior.
     *  \returns An iterator set to the first index of a block.
     *  \throws None.  No throw gurantee.
     */
    virtual const_iterator begin(size_t blocki)const noexcept
    {
        return shapes_[blocki].begin();
    }

    /** \brief Returns an iterator to just past the last index of a specified
     *  block.
     *
     *  \param[in] blocki The block whose indices are to be iterated over.
     *             \p blocki should be in the range [0,nblocks()); however, no
     *             check is made to ensure that this is the case and values of
     *             \p blocki outside this range will result in undefined
     *             behavior.
     *  \returns An iterator set to just past the last index of a block.
     *  \throws None.  No throw gurantee.
     */
    const_iterator end(size_t blocki)const noexcept
    {
        return shapes_[blocki].end();
    }

private:

    ///The type of an index
    using array_t=std::array<size_t,rank>;

    ///The pointers to the local memory
    std::vector<T*> buffers_;

    ///If the tensor doesn't expose T* these are the managed blocks
    std::vector<std::unique_ptr<T[]>> memory_;

    ///These are the shapes of each block
    std::vector<Shape<rank>> shapes_;

};

#ifdef BUILD_TWRAPPER_LIBRARY
extern template class MemoryBlock<1,double>;
extern template class MemoryBlock<2,double>;
extern template class MemoryBlock<3,double>;
extern template class MemoryBlock<4,double>;
#endif

}
