#pragma once
#include "TensorWrapper/TensorImpl/TensorWrapperImpl.hpp"
#ifdef __GNUC__
#pragma GCC system_header
#endif
#include<ctf/ctf.hpp>

namespace TWrapper {
namespace detail_ {

///Struct for getting at the base of a CTF tensor
template<size_t R, typename Index_t, typename Tensor_t>
struct CTFDerefer
{
    static Tensor_t& eval(Tensor_t& expr)
    {
        return expr;
    }
};

template<size_t R, typename Index_t, typename T>
struct CTFDerefer<R,Index_t, CTF::Tensor<T>>
{
    static auto eval(CTF::Tensor<T>& tensor)
    {
        return tensor[detail_::stringify(Index_t(),"").c_str()];
    }
};


template<typename T>
struct CTFItr{
    //Shared pointers b/c captured by value in lambda
    std::shared_ptr<std::vector<size_t>> idxs_;
    size_t npair_;
    std::shared_ptr<std::vector<T>> data_;

    CTFItr(const CTF::Tensor<T>& A)
    {
        int64_t* idx_temp;
        T* data_temp;
        int64_t temp_npair;
        A.read_local(&temp_npair,&idx_temp,&data_temp);
        npair_=temp_npair;
        idxs_=std::make_shared<std::vector<size_t>>(temp_npair);
        data_=std::make_shared<std::vector<T>>(temp_npair);
        std::copy(idx_temp,idx_temp+npair_,idxs_->begin());
        std::copy(data_temp,data_temp+npair_,data_->begin());
        delete [] idx_temp;
        delete [] data_temp;
    }
};

template<size_t rank, typename T>
std::array<IndexItr<rank>,2> CTF_make_itr(const CTFItr<T>& temp,
                                          const Shape<rank>& shape)
{
    using array_t=std::array<size_t,rank>;
    array_t begin{},end=shape.dims();

    //Won't be incremented so default next is fine.
    IndexItr<rank> end_itr(end,false,false,begin);
    IndexItr<rank> begin_itr(end,true,false,begin,
        [=](array_t& idx,const array_t&,const array_t& end_in,bool)
        {
            const size_t offset=shape.flat_index(idx);
            auto ptr=std::find(temp.idxs_->begin(),temp.idxs_->end(),offset);
            idx=(*ptr!=temp.idxs_->back()) ? shape.unflatten_index(ptr[1]) :
                                             end_in;
        }

    );
    return {begin_itr,end_itr};
}


template<size_t rank, typename T>
struct TensorWrapperImpl<rank,T,TensorTypes::CTF> {
    using array_t=std::array<size_t,rank>;
    using type=CTF::Tensor<T>;

    template<typename My_t>
    Shape<rank> dims(const My_t& impl)const{
        array_t dims;
        for(size_t i=0;i<rank;++i)dims[i]=impl.lens[i];
        return Shape<rank>(dims,false);
    }

    template<typename Tensor_t>
    MemoryBlock<rank,T> get_memory(Tensor_t& impl)const
    {
        MemoryBlock<rank,T> rv;
        CTFItr<T> temp(impl);
        auto shape=dims(impl);
        array_t begin{},end=shape.dims();
        auto itrs=CTF_make_itr(temp,shape);
        //Lifetime of data pointer is tied to begin_itr which is stored in class
        rv.add_block(temp.data_->data(),
            Shape<rank>(end,false,begin,itrs[0],itrs[1]));
        return rv;
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<rank,T>& block)const
    {
        Shape<rank> shape=dims(impl);
        for(size_t i=0;i<block.nblocks();++i)
        {
            std::vector<int64_t> idxs;
            for(auto idx=block.begin(i);idx!=block.end(i);++idx)
                idxs.push_back(shape.flat_index(*idx));
            impl.write(idxs.size(),idxs.data(),block.block(i));
        }

    }

    type allocate(const array_t& dims)const{
        std::array<int,rank> idims{};
        for(size_t i=0;i<rank;++i)
            idims[i]=dims[i];
        return type(rank,idims.data());
    }

//    template<typename Tensor_t>
//    auto permute(const Tensor_t& t, const array_t& permutation)const
//    {
//        return t.shuffle(permutation);
//    }

    template<typename LHS_Idx,typename LHS_t>
    auto trace(const LHS_t& clhs)const
    {
        LHS_t& lhs=const_cast<LHS_t&>(clhs);
        auto lhswidx=CTFDerefer<rank,LHS_Idx,LHS_t>::eval(lhs);
        return lhswidx;
    }

    template<typename Tensor_t>
    auto slice(const Tensor_t& impl,
               const array_t& start,const array_t& end)const
    {
        std::array<int,rank> temp_start,temp_end;
        for(size_t i=0;i<rank;++i)
        {
            temp_start[i]=start[i];
            temp_end[i]=end[i];
        }
        return impl.slice(temp_start.data(),temp_end.data());
    }

    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t& clhs, const RHS_t& crhs)const
    {
        if(dims(clhs)!=dims(crhs))return false;
        using idx_t=typename GenericIndex<rank>::type;
        type C(clhs);
        C=0.0;
        auto cwidx=CTFDerefer<rank,idx_t,CTF::Tensor<T>>::eval(C);
        cwidx=subtract<idx_t,idx_t>(clhs,crhs);
        return C.norm1()==0;
    }


    template<typename LHS_Idx, typename RHS_Idx,
             typename LHS_t, typename RHS_t>
    auto contraction(const LHS_t& clhs, const RHS_t& crhs)
    {
        LHS_t& lhs=const_cast<LHS_t&>(clhs);
        RHS_t& rhs=const_cast<RHS_t&>(crhs);
        auto lwidx=CTFDerefer<rank,LHS_Idx,LHS_t>::eval(lhs);
        auto rwidx=CTFDerefer<rank,RHS_Idx,RHS_t>::eval(rhs);
        return lwidx*rwidx;
    }

    template<typename LHS_idx,typename LHS_t>
    auto scale(const LHS_t& clhs,double val)const
    {
        LHS_t& lhs=const_cast<LHS_t&>(clhs);
        auto lhswidx=CTFDerefer<rank,LHS_idx,LHS_t>::eval(lhs);
        return lhswidx*val;
    }

    ///Adds to the tensor
    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t>
    auto add(const LHS_t& clhs,const RHS_t& crhs)const
    {
        LHS_t& lhs=const_cast<LHS_t&>(clhs);
        RHS_t& rhs=const_cast<RHS_t&>(crhs);
        auto lwidx=CTFDerefer<rank,LHS_Idx,LHS_t>::eval(lhs);
        auto rwidx=CTFDerefer<rank,RHS_Idx,RHS_t>::eval(rhs);
        return lwidx+rwidx;
    }

    ///Subtracts from the tensor
    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t>
    auto subtract(const LHS_t& clhs,const RHS_t& crhs)const
    {
        LHS_t& lhs=const_cast<LHS_t&>(clhs);
        RHS_t& rhs=const_cast<RHS_t&>(crhs);
        auto lwidx=CTFDerefer<rank,LHS_Idx,LHS_t>::eval(lhs);
        auto rwidx=CTFDerefer<rank,RHS_Idx,RHS_t>::eval(rhs);
        return lwidx-rwidx;
    }


    template<typename Out_Idx,typename Op_t>
    type eval(const Op_t& op,const array_t& dims)const
    {
        type rv=allocate(dims);
        auto rvwidx=CTFDerefer<rank,Out_Idx,type>::eval(rv);
        rvwidx=op;
        return rv;
    }

//    template<typename My_t>
//    auto self_adjoint_eigen_solver(const My_t& tensor)const
//    {

//        const Shape<2> shape=dims(tensor);
//        const size_t n=shape.dims()[0];
//        Eigen::MatrixXd temp(n,n);
//        std::copy(tensor.data(),tensor.data()+n*n,temp.data());
//        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(temp);
//        Eigen::Tensor<T,1> evals(n);
//        Eigen::Tensor<T,2> evecs(n,n);
//        const double* from=solver.eigenvalues().data();
//        double* to=evals.data();
//        std::copy(from,from+n,to);
//        const double* from2=solver.eigenvectors().data();
//        to=evecs.data();
//        std::copy(from2,from2+n*n,to);
//        return std::make_pair(evals,evecs);
//    }
};
}}
