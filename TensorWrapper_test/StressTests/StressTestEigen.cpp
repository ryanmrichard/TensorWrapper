#define EIGEN_USE_MKL_ALL
#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using namespace Eigen;
using tensor_type=EigenMatrix<double>;

void print_times(const std::string& msg, double eigen_time, double twtime)
{
    std::cout<<msg<<" Eigen time (s): "<<eigen_time<<std::endl;
    std::cout<<msg<<" TensorWrapper time (s): "<<twtime<<std::endl;
}


int main(int argc, char** argv)
{
    Tester tester("Stress Testing Eigen Matrix Wrapping");
    const size_t dim=argc>1?atoi(argv[1]):10;
    const std::array<size_t,2> dims({dim,dim});
    MatrixXd A=MatrixXd::Random(dim,dim),
            B=MatrixXd::Random(dim,dim),
            C=MatrixXd::Random(dim,dim);
    tensor_type _A(A),_B(B),_C(C);

    Timer timer;
    MatrixXd D=A+B+C;
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
    D=A*B*C;
    eigen_time=timer.get_time();
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");

    timer.reset();
    _D=_A(i,k)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A*B*C",eigen_time,wrapper_time);
    tester.test("A*B*C",_D==D);

    timer.reset();
    D=A.transpose()*B*C;
    eigen_time=timer.get_time();

    timer.reset();
    _D=_A(k,i)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A^T*B*C",eigen_time,wrapper_time);
    tester.test("A^T*B*C",D==_D);

    return tester.results();
}
