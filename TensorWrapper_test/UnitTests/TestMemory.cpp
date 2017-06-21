#include <TensorWrapper/MemoryBlock.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;

int main()
{
    Tester tester("Testing memory class");
    Shape<1> D1(std::array<size_t,1>({10}),true);
    std::vector<double> not_yours(10);
    MemoryBlock<1,double> block(D1,D1.dims(),
                     [&](const std::array<size_t,1>& idx)->double&
       {return not_yours[idx[0]];});
    std::iota(not_yours.begin(),not_yours.end(),1.0);
    tester.test("Same memory",&block(std::array<size_t,1>{0})==not_yours.data());
    tester.test("Same memory",&block(0)==not_yours.data());
    block(3)=99.9;
    size_t counter=0;
    for(double x: not_yours)
    {
        tester.test("Element"+std::to_string(counter),x==block(counter));
        ++counter;
    }
    return tester.results();
}
