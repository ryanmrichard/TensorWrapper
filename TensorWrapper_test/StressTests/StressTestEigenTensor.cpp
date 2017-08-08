#define EIGEN_USE_MKL_ALL
#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using namespace Eigen;
using tensor_type=EigenTensor<2,double>;
using idx_t=Eigen::IndexPair<int>;
template<size_t n>
using idx_array=std::array<idx_t,n>;

void print_times(const std::string& msg, double eigen_time, double twtime)
{
    std::cout<<msg<<" Eigen time (s): "<<eigen_time<<std::endl;
    std::cout<<msg<<" TensorWrapper time (s): "<<twtime<<std::endl;
}


int main(int argc, char** argv)
{
    Tester tester("Stress Testing Eigen Tensor Wrapping");
    const size_t dim=argc>1?atoi(argv[1]):10000;
    const std::array<size_t,2> dims({dim,dim});
    tensor_type _A(dims),_B(dims),_C(dims);
    FillRandom(_A);
    FillRandom(_B);
    FillRandom(_C);

    Tensor<double,2> A=_A.data(),B=_B.data(),C=_C.data();

    Timer timer;
    Tensor<double,2> D=A+B+C;
    double eigen_time=timer.get_time();
    timer.reset();
    tensor_type _D=_A+_B+_C;
    double wrapper_time=timer.get_time();
    print_times("A+B+C",eigen_time,wrapper_time);
    tester.test("A+B+C",D==_D);

    timer.reset();
    D=A-B-C;
    eigen_time=timer.get_time();
    timer.reset();
    _D=_A-_B-_C;
    wrapper_time=timer.get_time();
    print_times("A-B-C",eigen_time,wrapper_time);
    tester.test("A-B-C",D==_D);

    timer.reset();
    D=A.contract(B,idx_array<1>{idx_t{1,0}});
    Tensor<double,2>E=D.contract(C,idx_array<1>{idx_t{1,0}});
    eigen_time=timer.get_time();
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");

    timer.reset();
    _D=_A(i,k)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A*B*C",eigen_time,wrapper_time);
    tester.test("A*B*C",_D==E);

    timer.reset();
    D=A.contract(B,idx_array<1>{idx_t{0,0}});
    E=D.contract(C,idx_array<1>{idx_t{1,0}});
    eigen_time=timer.get_time();

    timer.reset();
    _D=_A(k,i)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A^T*B*C",eigen_time,wrapper_time);
    tester.test("A^T*B*C",E==_D);

    return tester.results();
}
