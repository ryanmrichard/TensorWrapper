#pragma once
#include <tiledarray.h>

namespace TWrapper {
namespace detail_ {

///Little function for making an arbitrary rank R index
template<size_t R>
std::string make_an_index()
{
    std::string idx="0";
    for(size_t i=1;i<R;++i)
        idx+=","+std::to_string(i);
    return idx;
}

///Struct for getting at the base of a TA tensor
template<size_t R, typename Index_t, typename Tensor_t>
struct TADerefer
{
    static auto eval(const Tensor_t& expr)
    {
        return expr;
    }
};

template<size_t R, typename Index_t, typename T>
struct TADerefer<R,Index_t, TiledArray::TArray<T>>
{
    static auto eval(const TiledArray::TArray<T>& tensor)
    {
        return tensor(detail_::stringify(Index_t()));
    }
};

template<size_t R, typename T, typename Range_t>
std::tuple<std::array<size_t,R>,
           std::array<size_t,R>,
           std::array<size_t,R>> TA2TW(const Range_t& range)
{
    std::array<size_t,R> start,end,sizes;
    for(size_t i=0;i<R;++i)
    {
        start[i]=range.lobound()[i];
        end[i]=range.upbound()[i];
        sizes[i]=end[i]-start[i];
    }
    return std::make_tuple(start,end,sizes);
}


template<size_t R,typename T>
struct TensorWrapperImpl<R,T,TensorTypes::TiledArray> {

    using array_t=std::array<size_t,R>;
    using type=TiledArray::TArray<T>;

    template<typename Tensor_t>
    Shape<R> dims(const Tensor_t& impl)const
    {
        array_t start,end,sizes;
        std::tie(start,end,sizes)=TA2TW<R,T>(impl.elements_range());
        return Shape<R>(sizes,true);
    }

    template<typename Tensor_t>
    MemoryBlock<R,T> get_memory(const Tensor_t& impl)const
    {
        MemoryBlock<R,T> rv;
        for(auto x:impl)
        {
            auto tile=x.get();
            array_t start,end,sizes;
            std::tie(start,end,sizes)=TA2TW<R,T>(impl.range());
            std::unique_ptr<T[]> mem(new T[tile.size()]);
            const T* origin=&tile[0];
            std::copy(origin,origin+tile.size(),mem.get());
            rv.add_block(std::move(mem),Shape<R>(sizes,true),start,end);
        }
        return rv;
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<R,T>& mem)const
    {
        size_t counter=0;
        using Tile_t=TiledArray::Tensor<T, Eigen::aligned_allocator<T>> ;
        for(auto tile=impl.begin(); tile!=impl.end(); ++tile)
        {
            auto range=impl.trange().make_tile_range(tile.ordinal());
            Tile_t newtile(range);
            T* buffer=mem.block(counter);
            for(int i=0; i<newtile.size(); ++i)
              newtile[i]=buffer[i];
            *tile=newtile;
            ++counter;
        }
    }

    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t>
    auto add(const LHS_t& lhs,const RHS_t&rhs)const
    {
        auto _l=TADerefer<R,LHS_Idx,LHS_t>::eval(lhs);
        auto _r=TADerefer<R,RHS_Idx,RHS_t>::eval(rhs);
        return _l+_r;
    }

    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t>
    auto subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        auto _l=TADerefer<R,LHS_Idx,LHS_t>::eval(lhs);
        auto _r=TADerefer<R,RHS_Idx,RHS_t>::eval(rhs);
        return _l-_r;
    }

    template<typename LHS_Idx,typename RHS_Idx,typename LHS_t,typename RHS_t>
    auto contraction(const LHS_t& lhs, const RHS_t& rhs)const
    {
        auto _l=TADerefer<R,LHS_Idx,LHS_t>::eval(lhs);
        auto _r=TADerefer<R,RHS_Idx,RHS_t>::eval(rhs);
        return _l*_r;
    }

    template<typename LHS_Idx,typename Tensor_t>
    auto trace(const Tensor_t& lhs)const
    {
        return TADerefer<R,LHS_Idx,Tensor_t>::eval(lhs);
    }

    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t& lhs,const RHS_t&rhs)const
    {
        auto idx=make_an_index<R>();
        return (lhs(idx)-rhs(idx)).min() == 0;
    }

    template<typename Index_t,typename LHS_t>
    auto scale(const LHS_t& lhs,T c)const
    {
        auto _l=TADerefer<R,Index_t,LHS_t>::eval(lhs);
        return _l*c;
    }

    template<typename Idx_t,typename Op_t>
    type eval(const Op_t& op,const array_t&)const
    {
            type c;
            c(detail_::stringify(Idx_t()))=op;
            return c;
    }

    template<typename Tensor_t>
    auto permute(const Tensor_t& t, const array_t& perm)const
    {
        auto idx=make_an_index<R>();
        std::string rv=std::to_string(perm[0]);
        for(size_t i=1;i<R;++i)
            rv+=","+std::to_string(perm[i]);
        type c;
        c(rv)=t(idx);
        return c;
    }

    type allocate(const array_t& dims)const{
        TA::World& world = TA::get_default_world();
        std::array<TiledArray::TiledRange1,R> ranges;
        for(size_t i=0;i<R;++i)
            ranges[i]=TiledArray::TiledRange1(0,dims[i]);
        TiledArray::TiledRange trange(ranges.begin(),ranges.end());
        type rv(world,trange);
        return rv;
    }

    template<typename Tensor_t>
    type slice(const Tensor_t& impl,const array_t& start,
               const array_t& end)const
    {
       auto idx=make_an_index<R>();
       type rv;
       rv(idx)=impl(idx).block(start,end);
       return rv;
    }


 };



//    template<typename My_t>
//    auto self_adjoint_eigen_solver(const My_t& tensor)const
//    {
//         Eigen::SelfAdjointEigenSolver<My_t> solver(tensor);
//         return std::make_pair(solver.eigenvalues(),solver.eigenvectors());
//    }
}}//End namespaces
