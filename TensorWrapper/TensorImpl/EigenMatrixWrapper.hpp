//This file meant from inclusion only from TensorImpls.hpp
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <map>
#include "TensorWrapper/TensorImpl/ContractionHelper.hpp"
namespace TWrapper {
namespace detail_ {

//Unfortunately the Eigen API depends on the rank these first few structs are
//specialized on the rank to ensure the correct Eigen API is called.

///Primary template for selecting the backend type
template<size_t R, typename T>
struct ToEigenType;

///Specilaization selecting the type of a matrix
template<typename T>
struct ToEigenType<2,T>
{
    using type=Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
};

///Specialization selecting the type of a column vector
template<typename T>
struct ToEigenType<1,T>
{
    using type=Eigen::Matrix<T,Eigen::Dynamic,1>;
};

///Specialization selecting the type of a scalar
template<typename T>
struct ToEigenType<0,T>
{
    using type=Eigen::Matrix<T,1,1>;
};

///Primary template for making the correct shape instance.
template<size_t R>
struct DimMaker;

///Specialization of DimMaker to matrices
template<>
struct DimMaker<2>
{
    template<typename Tensor_t>
    static Shape<2> eval(const Tensor_t& impl)
    {
        size_t rows=(size_t)impl.rows();
        size_t cols=(size_t)impl.cols();
        return Shape<2>(std::array<size_t,2>{rows,cols},impl.IsRowMajor);
    }
};

///Specialization of DimMaker to vectors
template<>
struct DimMaker<1>
{
    template<typename Tensor_t>
    static Shape<1> eval(const Tensor_t& impl)
    {
        size_t rows=(size_t)impl.rows();
        return Shape<1>(std::array<size_t,1>{rows},impl.IsRowMajor);
    }
};

///Specialization of DimMaker to scalars
template<>
struct DimMaker<0>
{
    template<typename Tensor_t>
    static Shape<0> eval(const Tensor_t& impl)
    {
        return Shape<0>(std::array<size_t,0>{},impl.IsRowMajor);
    }
};

///Primary template for contraction
template<size_t R>
struct ContractionImpl;

template<>
struct ContractionImpl<2>
{
    template<typename LHS_Idx, typename RHS_Idx, typename LHS_t, typename RHS_t>
    static auto eval(const LHS_t& lhs, const RHS_t& rhs)
    {
        constexpr bool isvec=RHS_Idx::size()==1;
        constexpr bool rrow=RHS_t::RowsAtCompileTime==1;
        using contract=ContractionTraits<LHS_Idx,RHS_Idx,2,RHS_Idx::size()>;
        constexpr bool rtranspose=(!isvec?contract::rtranspose:
                                          contract::rtranspose!=rrow);
        return ContractionHelper<contract::nfree,
                                 contract::ndummy,
                                 contract::ltranspose,
                                 rtranspose>().contract(lhs,rhs);
    }
};

template<>
struct ContractionImpl<1> {

    template<typename LHS_Idx, typename RHS_Idx, typename LHS_t, typename RHS_t>
    static auto eval(const LHS_t& lhs, const RHS_t& rhs)
    {
        constexpr bool lrow=LHS_t::RowsAtCompileTime==1;
        constexpr bool is_vec=(RHS_Idx::size()==1);
        constexpr bool rrow=RHS_t::RowsAtCompileTime==1;
        using contract=ContractionTraits<LHS_Idx,RHS_Idx,1,RHS_Idx::size()>;
        constexpr bool rtranspose=(!is_vec?contract::rtranspose:
                                           rrow!=contract::rtranspose);
        return ContractionHelper<contract::nfree,
                                 contract::ndummy,
                                 contract::ltranspose!=lrow,
                                 rtranspose>().contract(lhs,rhs);
    }
};

template<>
struct ContractionImpl<0> {
    template<typename, typename, typename LHS_t, typename RHS_t>
    static auto eval(const LHS_t& lhs, const RHS_t& rhs)
    {
        return lhs*rhs;
    }
};

///Primary template for slicing an Eigen matrix
template<size_t R,typename T>
struct SlicingImpl;

template<typename T>
struct SlicingImpl<2,T> {
  template<typename Tensor_t>
  static auto eval(const Tensor_t& impl,
                   const std::array<size_t,2>& start,
                   const std::array<size_t,2>& end)
  {
    return typename ToEigenType<2,T>::type(
                impl.block(start[0],start[1],end[0]-start[0],end[1]-start[1]));
  }
};

template<typename T>
struct SlicingImpl<1,T> {
  template<typename Tensor_t>
  static auto eval(const Tensor_t& impl,
                   const std::array<size_t,1>& start,
                   const std::array<size_t,1>& end)
  {
    return typename ToEigenType<1,T>::type(
                    impl.segment(start[0],end[0]-start[0]));
  }
};

template<typename T>
struct SlicingImpl<0,T> {
  template<typename Tensor_t>
  static auto eval(const Tensor_t& impl,
                   const std::array<size_t,0>&,
                   const std::array<size_t,0>&)
  {
    return typename ToEigenType<0,T>::type(impl.segment(0,1));
  }
};


template<size_t R, typename T>
struct SetMemoryImpl{
    template<typename Tensor_t,size_t...Is>
    static void eval(Tensor_t& impl,const MemoryBlock<R,T>& block,
                     std::index_sequence<Is...>)
    {
        //Check if it's actually the T* of this tensor
        if(block.block(0)==impl.data() && block.nblocks()==1)
            return;
        for(size_t i=0;i<block.nblocks();++i)
        {
            auto index=block.begin(i),last=block.end(i);
            T* blocki=block.block(i);
            size_t counter=0;

            while(index!=last)
            {
                impl((*index)[Is]...)=blocki[counter++];
                ++index;
            }
        }
    }
};

template<typename T>
struct SetMemoryImpl<0,T>{
    template<typename Tensor_t, size_t...Is>
    static void eval(Tensor_t& impl,const MemoryBlock<0,T>& block,
                     std::index_sequence<Is...>)
    {
        impl(0,0)=block.block(0)[0];
    }
};


template<size_t R,typename T,size_t...Is>
auto allocate_impl(const std::array<size_t,R>& dims,std::index_sequence<Is...>)
{
    return typename ToEigenType<R,T>::type(dims[Is]...);
}

//Matrix specialization
template<size_t R, typename T>
struct TensorWrapperImpl<R,T,TensorTypes::EigenMatrix> {

    using array_t=std::array<size_t,R>;
    using type=typename ToEigenType<R,T>::type;

    template<typename LHS_Idx,typename RHS_Idx>
    using EnableIfSameIdx=std::enable_if<
            std::is_same<LHS_Idx,RHS_Idx>::value,int>;

    template<typename LHS_Idx,typename RHS_Idx>
    using EnableIfNotSameIdx=std::enable_if<
            !std::is_same<LHS_Idx,RHS_Idx>::value,int>;

    template<typename Tensor_t>
    Shape<R> dims(const Tensor_t& impl)const{
        return DimMaker<R>::eval(impl);
    }

    template<typename Tensor_t>
    auto get_memory(Tensor_t& impl)const{
        MemoryBlock<R,T> rv;
        rv.add_block(impl.data(),dims(impl),array_t{},dims(impl).dims());
        return rv;
    }

    template<typename Tensor_t>
    void set_memory(Tensor_t& impl,const MemoryBlock<R,T>& block)const
    {
        SetMemoryImpl<R,T>::eval(impl,block,std::make_index_sequence<R>());
    }

    auto allocate(const array_t& dims)const{
        return allocate_impl<R,T>(dims,std::make_index_sequence<R>());
    }

    template<typename Tensor_t>
    auto permute(const Tensor_t& t, const array_t&)const
    {
        //Only one possibility {1,0}
        return t.transpose();
    }

    template<typename Tensor_t>
    auto slice(const Tensor_t& impl,
               const array_t& start,const array_t& end)const{
        return SlicingImpl<R,T>::eval(impl,start,end);
    }

    ///Returns true if two tensors are equal
    template<typename LHS_t, typename RHS_t>
    bool are_equal(const LHS_t& lhs, const RHS_t& rhs)const
    {
        return lhs == rhs;
    }

    template<typename, typename Tensor_t>
    auto scale(const Tensor_t& lhs,double val)const
    {
        return lhs*val;
    }

    ///Adds to the tensor
    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t,
             typename EnableIfSameIdx<LHS_Idx,RHS_Idx>::type=0>
    auto add(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs+rhs;
    }


    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t,
             typename EnableIfNotSameIdx<LHS_Idx,RHS_Idx>::type=0>
    auto add(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs+rhs.transpose();
    }

    ///Subtracts from the tensor
    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t,
             typename EnableIfSameIdx<LHS_Idx,RHS_Idx>::type=0>
    auto subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs-rhs;
    }


    template<typename LHS_Idx,typename RHS_Idx,
             typename LHS_t,typename RHS_t,
             typename EnableIfNotSameIdx<LHS_Idx,RHS_Idx>::type=0>
    auto subtract(const LHS_t& lhs,const RHS_t&rhs)const
    {
        return lhs-rhs.transpose();
    }

    template<typename,typename Op_t>
    auto eval(const Op_t& op,const array_t&)const
    {
        typename ToEigenType<R,T>::type c=op;
        return c;
    }

    template<typename LHS_Idx,typename RHS_Idx,typename LHS_t,typename RHS_t>
    auto contraction(const LHS_t& lhs, const RHS_t& rhs)const
    {
        return ContractionImpl<R>::template eval<LHS_Idx,RHS_Idx>(lhs,rhs);
    }

    template<typename My_t>
    auto self_adjoint_eigen_solver(const My_t& tensor)const
    {
        static_assert(R==2,"Eigen solving only available for matrices");
        Eigen::SelfAdjointEigenSolver<My_t> solver(tensor);
        return std::make_pair(solver.eigenvalues(),solver.eigenvectors());
    }

};

}}//End namespaces
