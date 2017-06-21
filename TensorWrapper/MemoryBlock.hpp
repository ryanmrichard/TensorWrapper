#pragma once
#include "TensorWrapper/Shape.hpp"
#include <vector>
#include <functional>
namespace TWrapper {

/** \brief A class for wrapping the memory access semantics of the underlying
 *         tensor implementation.
 *
 *
 *  \tparam rank The rank of the block we wrap
 *  \tparam T  The type of an element in the block we wrap
 *
 */
template<size_t rank, typename T>
class MemoryBlock{
    using array_t=std::array<size_t,rank>;
    std::function<T&(const array_t&)> fxn_;
public:
    ///The shape of the current buffer.  Read-only.
    const Shape<rank> local_shape;

    ///The first index of the current buffer.
    const array_t start;

    ///The last index of the current buffer;
    const array_t end;

    /** \brief Wraps a block of contigious memory or (one something that looks
     *         like one)
     *
     *  \param[in] shape The shape of the block
     *  \param[in] fxn The functor provided by the backend to give us an element
     */
    template<typename fxn_t>
    MemoryBlock(const Shape<rank>& shape,
                const array_t& end_in,
                fxn_t fxn,
                const array_t& start_in=array_t{})
:
        fxn_(fxn),
        local_shape(shape),
        start(start_in),
        end(end_in)
    {
    }

    /** \brief Copies another MemoryBlock.
     *
     *  Copying is a bit complicated.  If we own the memory then we deep copy.
     *  Otherwise this is a shallow copy.
     *
     *  \param[in] other The MemoryBlock to copy.  Must contain the same data
     *                   and have the same rank.
     *
     */
    MemoryBlock(const MemoryBlock<rank,T>& other)=default;

    MemoryBlock(MemoryBlock<rank,T>&&)=default;
    MemoryBlock<rank,T>& operator=(MemoryBlock&&)=default;
    MemoryBlock<rank,T>& operator=(const MemoryBlock& other)=default;

    ~MemoryBlock()=default;

    ///Returns a read/write element
    T& operator()(const std::array<size_t,rank>& idx)
    {
        return fxn_(idx);
    }

    ///Returns a read-only element
    const T& operator()(const std::array<size_t,rank>& idx)const
    {
        return fxn_(idx);
    }

    ///Syntactic sugar for accessing read/write element
    template<typename...Args>
    T& operator()(size_t el1, Args...idx)
    {
        return operator()(std::array<size_t,rank>({el1,idx...}));
    }

    ///Sytactic sugar for accessing read-only element
    template<typename...Args>
    const T& operator()(size_t el1,Args...idx)const
    {
        return operator()(std::array<size_t,rank>({el1,idx...}));
    }
};

}
