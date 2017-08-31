#include <TensorWrapper/MemoryBlock.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;

int main()
{
    Tester tester("Testing memory class");
    std::array<size_t,1> start{20},local{10},end{30};
    Shape<1> D1(local,true);
    std::vector<double> not_yours(10);
    MemoryBlock<1,double> block;
    block.add_block(not_yours.data(),D1,start,end);
    std::iota(not_yours.begin(),not_yours.end(),1.0);
    tester.test("Same memory",block.block(0)==not_yours.data());
    auto blocki=block.begin(0),last_block=block.end(0);
    size_t counter=0;
    while(blocki!=last_block)
    {
        tester.test("Iterator element "+std::to_string(counter),
                    (*blocki)[0]==20+counter);
        ++counter;
        ++blocki;
    }
    return tester.results();
}
