#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using tensor_type=TensorWrapper<2,double,detail_::TensorTypes::GlobalArrays>;
using wrapped_type=GATensor<2,double>;

void print_times(const std::string& msg, double native_time, double twtime)
{
    std::cout<<msg<<" Eigen time (s): "<<native_time<<std::endl;
    std::cout<<msg<<" TensorWrapper time (s): "<<twtime<<std::endl;
}


int main(int argc, char** argv)
{
    Tester tester("Stress Testing Global Arrays Wrapping");
    const size_t dim=argc>1?atoi(argv[1]):10000;
    const std::array<size_t,2> dims({dim,dim});
    tensor_type _A(dims),_B(dims),_C(dims);
    FillRandom(_A);
    FillRandom(_B);
    FillRandom(_C);

    wrapped_type A(_A.data()),B(_B.data()),C(_C.data());

    Timer timer;
    wrapped_type D=A+B+C;
    double native_time=timer.get_time();
    timer.reset();
    tensor_type _D=_A+_B+_C;
    double wrapper_time=timer.get_time();
    print_times("A+B+C",native_time,wrapper_time);
    tester.test("A+B+C",D==_D);

    timer.reset();
    D=A-B-C;
    native_time=timer.get_time();
    timer.reset();
    _D=_A-_B-_C;
    wrapper_time=timer.get_time();
    print_times("A-B-C",native_time,wrapper_time);
    tester.test("A-B-C",D==_D);

    timer.reset();
    D=A*B*C;
    native_time=timer.get_time();
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");

    timer.reset();
    _D=_A(i,k)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A*B*C",native_time,wrapper_time);
    tester.test("A*B*C",_D==D);

    timer.reset();
    D=A.transpose()*B*C;
    native_time=timer.get_time();

    timer.reset();
    _D=_A(k,i)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A^T*B*C",native_time,wrapper_time);
    tester.test("A^T*B*C",D==_D);

    return tester.results();
}
