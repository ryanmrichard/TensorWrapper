#include <TensorWrapper/Config.hpp>
#include <TensorWrapper/TensorImpl/CTFWrapper.hpp>
#include <TensorWrapper/RunTime.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using namespace TWrapper::detail_;


template<size_t R>
using impl_t=TensorWrapperImpl<R,double,TensorTypes::CTF>;

int main(int argc, char** argv)
{
    Tester tester("Testing wrapping of Eigen matrix library");
    const size_t dim=10;
    impl_t<2> impl;
#ifdef ENABLE_CTF
    using tensor_type=impl_t<2>::type;
    RunTime rt(argc,argv);
    std::array<size_t,2> dims{dim,dim};
    std::array<int,2> idims{dim,dim};
    tensor_type A(2,idims.data());
    A.fill_random(0.0,100.0);

    //Shape
    Shape<2> corr_shape(dims,false);
    tester.test("Shape",corr_shape==impl.dims(A));

    //Get/set memory
    int64_t npair=0;
    int64_t* idxs;
    double* data;
    A.read_local(&npair,&idxs,&data);
    auto mem=impl.get_memory(A);
    tester.test("Nblocks",mem.nblocks());
    tester.test("Memory is same",std::equal(data,data+npair,mem.block(0)));
    mem.block(0)[0]=9.9;
    impl.set_memory(A,mem);
    auto mem2=impl.get_memory(A);
    tester.test("Memory is set",mem2.block(0)[0]==9.9);

    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    using idx_ii=make_indices<decltype(i),decltype(i)>;
    using idx_ij=make_indices<decltype(i),decltype(j)>;
    using idx_ik=make_indices<decltype(i),decltype(k)>;
    using idx_kj=make_indices<decltype(k),decltype(j)>;

    tensor_type B(2,idims.data()),corr_C(2,idims.data()),C(2,idims.data());
    B.fill_random(0.0,100.0);

    //Subtraction
    corr_C["ij"]=A["ij"]-B["ij"];
    C["ij"]=impl.subtract<idx_ij,idx_ij>(A,B);
    mem=impl.get_memory(corr_C);
    mem2=impl.get_memory(C);
    tester.test("Subtraction",
                std::equal(mem.block(0),mem.block(0)+100,mem2.block(0)));

    //Equality (depends on subtraction)
    tester.test("Are equal",impl.are_equal(C,corr_C));
    tester.test("Not equal",!impl.are_equal(C,B));

    //Eval
    C=impl.eval<idx_ij>(A["ij"]-B["ij"],dims);
    tester.test("Eval",impl.are_equal(C,corr_C));

    //Slice
    std::array<size_t,2> slice_start{7,7};
    std::array<int,2> islice_start{7,7};
    auto corr_slice=A.slice(islice_start.data(),idims.data());
    auto slice=impl.slice(A,slice_start,dims);
    tester.test("Slice",impl.are_equal(corr_slice,slice));

    //Scale
    corr_C["ij"]=A["ij"]*0.5;
    C["ij"]=impl.scale<idx_ij>(A,0.5);
    tester.test("Scale",impl.are_equal(C,corr_C));

    //Add
    corr_C["ij"]=B["ij"]+A["ij"];
    C["ij"]=impl.add<idx_ij,idx_ij>(B,A);
    tester.test("Add",impl.are_equal(corr_C,C));

    //Trace
    std::array<int,1> scalar{1};
    tensor_type corr_tr(0,scalar.data()),tr(0,scalar.data());
    corr_tr[""]=B["ii"];

    impl_t<0> scalar_impl;
    tr[""]=impl.trace<idx_ii>(B);
    tester.test("Trace",scalar_impl.are_equal(corr_tr,tr));

    //Contraction
    corr_C["ij"]=B["ik"]*A["kj"];
    C["ij"]=impl.contraction<idx_ik,idx_kj>(B,A);
    tester.test("Contraction",impl.are_equal(corr_C,C));


#endif
    return tester.results();
}
