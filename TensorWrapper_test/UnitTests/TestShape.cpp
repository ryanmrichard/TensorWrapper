#include <TensorWrapper/Shape.hpp>
#include<vector>
#include "TestHelpers.hpp"

using namespace TWrapper;

const std::vector<std::array<size_t,1>> corr_D1_rowmajor({
    {0,},{1,},{2,},{3,},{4,},{5,},{6,},{7,},{8,},{9,},
});

const std::vector<std::array<size_t,2>> corr_D2_rowmajor({
    {0,0,},{0,1,},{0,2,},{0,3,},{0,4,},{1,0,},{1,1,},{1,2,},{1,3,},{1,4,},
    {2,0,},{2,1,},{2,2,},{2,3,},{2,4,},{3,0,},{3,1,},{3,2,},{3,3,},{3,4,},
    {4,0,},{4,1,},{4,2,},{4,3,},{4,4,},{5,0,},{5,1,},{5,2,},{5,3,},{5,4,},
    {6,0,},{6,1,},{6,2,},{6,3,},{6,4,},{7,0,},{7,1,},{7,2,},{7,3,},{7,4,},
    {8,0,},{8,1,},{8,2,},{8,3,},{8,4,},{9,0,},{9,1,},{9,2,},{9,3,},{9,4,}
});

const std::vector<std::array<size_t,2>> corr_D2_colmajor({
{0,0},{1,0},{2,0},{3,0},{4,0},{5,0},{6,0},{7,0},{8,0},{9,0},{0,1},{1,1},{2,1},
{3,1},{4,1},{5,1},{6,1},{7,1},{8,1},{9,1},{0,2},{1,2},{2,2},{3,2},{4,2},{5,2},
{6,2},{7,2},{8,2},{9,2},{0,3},{1,3},{2,3},{3,3},{4,3},{5,3},{6,3},{7,3},{8,3},
{9,3},{0,4},{1,4},{2,4},{3,4},{4,4},{5,4},{6,4},{7,4},{8,4},{9,4},
});

const std::vector<std::array<size_t,3>> corr_D3_rowmajor(
{
{0,0,0},{0,0,1},{0,0,2},{0,1,0},{0,1,1},{0,1,2},{0,2,0},{0,2,1},{0,2,2},{1,0,0},
{1,0,1},{1,0,2},{1,1,0},{1,1,1},{1,1,2},{1,2,0},{1,2,1},{1,2,2},{2,0,0},{2,0,1},
{2,0,2},{2,1,0},{2,1,1},{2,1,2},{2,2,0},{2,2,1},{2,2,2}
});

const std::vector<std::array<size_t,3>> corr_D3_colmajor({
{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0},{2,1,0},{0,2,0},{1,2,0},{2,2,0},{0,0,1},
{1,0,1},{2,0,1},{0,1,1},{1,1,1},{2,1,1},{0,2,1},{1,2,1},{2,2,1},{0,0,2},{1,0,2},
{2,0,2},{0,1,2},{1,1,2},{2,1,2},{0,2,2},{1,2,2},{2,2,2}
});

int main()
{
    Tester tester("Testing shape class");

    //Row major testing
    Shape<0> D0(std::array<size_t,0>({}));
    Shape<1> D1(std::array<size_t,1>({10}));
    Shape<2> D2(std::array<size_t,2>({10,5}));
    Shape<3> D3(std::array<size_t,3>({3,3,3}));
    tester.test("Scalar size",D0.size()==1);
    tester.test("Vector size",D1.size()==10);
    tester.test("Matrix size",D2.size()==50);
    tester.test("Tensor size",D3.size()==27);
    for(auto idx: D0)//Should never actually get into this loop
    {
        tester.test("Rank 0 shouldn't iterate",false);
        std::cout<<idx.size()<<std::endl;//Just to silence the compiler warning
    }
    size_t counter=0;
    for(auto idx: D1)
    {
        const auto& corr=corr_D1_rowmajor[counter];
        tester.test(elem_name(corr),idx==corr);
        tester.test(elem_name(corr)+" flat",counter==D1.flat_index(corr));
        ++counter;
    }
    counter=0;
    for(auto idx: D2)
    {
        const auto& corr=corr_D2_rowmajor[counter];
        tester.test(elem_name(corr),idx==corr);
        tester.test(elem_name(corr)+" flat",counter==D2.flat_index(corr));
        ++counter;
    }
    counter=0;
    for(auto idx: D3)
    {
        const auto& corr=corr_D3_rowmajor[counter];
        tester.test(elem_name(corr),idx==corr);
        tester.test(elem_name(corr)+" flat",counter==D3.flat_index(corr));
        ++counter;
    }

    //Column major testing
    D0=Shape<0>(std::array<size_t,0>({}),false);
    D1=Shape<1>(std::array<size_t,1>({10}),false);
    D2=Shape<2>(std::array<size_t,2>({10,5}),false);
    D3=Shape<3>(std::array<size_t,3>({3,3,3}),false);
    for(auto idx: D0)//Should never actually get into this loop
    {
        tester.test("Rank 0 shouldn't iterate",false);
        std::cout<<idx.size()<<std::endl;//Just to silence the compiler warning
    }
    counter=0;
    for(auto idx: D1)
    {
        //Vector is the same row vs. col
        const auto& corr=corr_D1_rowmajor[counter];
        tester.test(elem_name(corr),idx==corr);
        tester.test(elem_name(corr)+" flat",counter==D1.flat_index(corr));
        ++counter;
    }
    counter=0;
    for(auto idx: D2)
    {
        const auto& corr=corr_D2_colmajor[counter];
        tester.test(elem_name(corr),idx==corr);
        tester.test(elem_name(corr)+" flat",counter==D2.flat_index(corr));
        ++counter;
    }
    counter=0;
    for(auto idx: D3)
    {
        const auto& corr=corr_D3_colmajor[counter];
        tester.test(elem_name(corr),idx==corr);
        tester.test(elem_name(corr)+" flat",counter==D3.flat_index(corr));
        ++counter;
    }
    return tester.results();
}
