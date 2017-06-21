#include <TensorWrapper/Operation.hpp>
#include "TestHelpers.hpp"
#include <vector>
using namespace TWrapper;
using namespace detail_;

//Simulates a tensor backend specific operation
struct CounterGuts{
    template<typename T>
    auto eval(T val)
    {
        return val+1;
    }
};

//Simulates a TensorWrapper operation
struct Counter{
    template<typename Impl_t,typename T>
    auto eval(Impl_t impl,T val)
    {
        return impl.eval(val);
    }
};

//Simulates binary tensor backend operation
struct AddGuts{
    template<typename lhs_t,typename rhs_t>
    auto eval(lhs_t lhs, rhs_t rhs)
    {
        return lhs+rhs;
    }
};

//Simulates binary TensorWrapper operation
struct Add{
    template<typename Impl_t,typename lhs_t,typename rhs_t>
    auto eval(Impl_t impl,lhs_t lhs, rhs_t rhs)
    {
        return impl.eval(lhs,rhs);
    }
};

int main()
{
    Tester tester("Testing Operation class");
    size_t zero=0;

    //Nothing like unrolling expression templating...
    Operation<Counter,size_t> depth1(Counter(),zero);
    Operation<Counter,decltype(depth1)> depth2(Counter(),depth1);
//    Operation<Counter,decltype(depth2)> depth3(Counter(),depth2);
//    Operation<Counter,decltype(depth3)> depth4(Counter(),depth3);
//    Operation<Counter,decltype(depth4)> sum(Counter(),depth4);
    size_t total=depth2.eval(CounterGuts());
    tester.test("Unary",total==5);

//    Operation<Add,size_t,size_t> add_depth1(Add(),total,total);//10
//    Operation<Add,decltype(add_depth1),size_t> add_depth2(Add(),add_depth1,total);//15
//    Operation<Add,decltype(add_depth2),size_t> add_depth3(Add(),add_depth2,total);//20
//    Operation<Add,decltype(add_depth3),size_t> add_depth4(Add(),add_depth3,total);//25
//    Operation<Add,decltype(add_depth4),size_t> add_sum(Add(),add_depth4,total);//30

//    total=add_sum.eval(AddGuts());
//    tester.test("Binary",total==30);

    return tester.results();
}
