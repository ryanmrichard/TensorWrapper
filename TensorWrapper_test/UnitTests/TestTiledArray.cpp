#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"
using namespace TWrapper;

int main(int argc, char** argv)
{
    Tester tester("Testing TiledArray Tensor Wrapping");
    const size_t dim=10;
    const std::array<size_t,2> dims({dim,dim});
    RunTime rt(argc,argv);
#ifdef ENABLE_tiledarray
    using TATensor=TiledArray::TArrayD;
    detail_::TensorWrapperImpl<2,double,detail_::TensorTypes::TiledArray> impl;
    TA::World& world = TA::get_default_world();
    TiledArray::TiledRange trange{{0, 5, dim},
                                  {0, 5, dim}};
    TATensor A(world,trange);
    A.fill_local(3.0);

    Shape<2> corr_shape(dims,true);
    tester.test("Shape",impl.dims(A)==corr_shape);

    auto mem=impl.get_memory(A);
    tester.test("NBlocks",mem.nblocks()==4);
    std::vector<double> corr_mem(25,3.0);
    for(size_t i=0;i<4;++i)
    {
        double* buffer=mem.block(i);
        tester.test("Get Block "+std::to_string(i),std::equal(corr_mem.begin(),
                                                          corr_mem.end(),
                                                          buffer));
        if(i==2)
        {
            for(size_t j=0;j<25;++j)
                buffer[j]=4.0;
        }
    }

    TATensor B(world,trange);
    impl.set_memory(B,mem);
    std::vector<double> corr_mem2(25,4.0);
    auto mem2=impl.get_memory(B);
    for(size_t i=0;i<4;++i)
    {
        if(i==2)
            tester.test("Set Block 2",
                std::equal(corr_mem2.begin(),corr_mem2.end(),mem2.block(2)));
        else
            tester.test("Set Block "+std::to_string(i),
                std::equal(corr_mem.begin(),corr_mem.end(),mem2.block(i)));
    }

    //TATensor F,G;
    //F("i,j")=B("i,j").block({0,0},{5,5});
    //std::cout<<F<<std::endl;
    //G=impl.slice(B,{4,4},{6,6});
    //mem=impl.get_memory(G);
    //std::array<double,4> corr_slice({3.0,3.0,4.0,3.0});
    //tester.test("Slice",std::equal(corr_slice.begin(),corr_slice.end(),
    //                               mem.block(0));


    tester.test("Are equal",impl.are_equal(A,A));

    //Addition
    TATensor C,D,E;
    C("i,j")=A("i,j")+B("i,j");
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");
    using idx_ij=detail_::make_indices<decltype(i),decltype(j)>;
    using idx_ik=detail_::make_indices<decltype(i),decltype(k)>;
    using idx_ji=detail_::make_indices<decltype(j),decltype(i)>;
    using idx_jk=detail_::make_indices<decltype(j),decltype(k)>;
    using idx_kl=detail_::make_indices<decltype(k),decltype(l)>;
    D("i,j")=impl.add<idx_ij,idx_ij>(A,B);
    tester.test("A + B",impl.are_equal(C,D));
    D("i,j")=A("i,j")+B("i,j")+C("i,j");
    E("i,j")=impl.add<idx_ij,idx_ij>(impl.add<idx_ij,idx_ij>(A,B),C);
    tester.test("A + B + C",impl.are_equal(D,E));
    C("i,j")=A("i,j")+B("j,i");
    D("i,j")=impl.add<idx_ij,idx_ji>(A,B);
    tester.test("A + B^T",impl.are_equal(C,D));

    //Subtraction
    C("i,j")=A("i,j")-B("i,j");
    D("i,j")=impl.subtract<idx_ij,idx_ij>(A,B);
    tester.test("A - B",impl.are_equal(C,D));
    D("i,j")=A("i,j")-B("i,j")-C("i,j");
    E("i,j")=impl.subtract<idx_ij,idx_ij>(impl.subtract<idx_ij,idx_ij>(A,B),C);
    tester.test("A - B - C",impl.are_equal(D,E));
    C("i,j")=A("i,j")-B("j,i");
    D("i,j")=impl.subtract<idx_ij,idx_ji>(A,B);
    tester.test("A - B^T",impl.are_equal(C,D));

    //Scale
    C("i,j")=A("i,j")*0.5;
    D("i,j")=impl.scale<idx_ij>(A,0.5);
    tester.test("Scale",impl.are_equal(C,D));

    //Permute
    C("j,i")=B("i,j");
    D=impl.permute(B,{1,0});
    tester.test("Permutation",impl.are_equal(C,D));

    //Contraction
    D("i,k")=A("i,j")*B("j,k");
    E("i,k")=impl.contraction<idx_ij,idx_jk>(A,B);
    tester.test("A * B",impl.are_equal(D,E));
    D("i,k")=A("j,i")*B("j,k");
    E("i,k")=impl.contraction<idx_ji,idx_jk>(A,B);
    tester.test("A^T * B",impl.are_equal(D,E));
    D("i,l")=A("j,i")*B("j,k")*C("k,l");
    E("i,l")=
        impl.contraction<idx_ik,idx_kl>(impl.contraction<idx_ji,idx_jk>(A,B),C);
    tester.test("A^T * B * C",impl.are_equal(D,E));

    C("i,j")=A("i,j")+B("i,j");
    E=impl.eval<idx_ij>(impl.add<idx_ij,idx_ij>(A,B),dims);
    tester.test("Eval",impl.are_equal(C,E));
#endif
    return tester.results();
}
