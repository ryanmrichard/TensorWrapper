#include <TensorWrapper/TensorWrapper.hpp>
#include <iostream>
#include "TestHelpers.hpp"

using namespace TWrapper;
using namespace Eigen;
using tensor_type=TensorWrapper<2,double,MatrixXd>;

int main()
{
    Tester tester("Stress Testing Eigen Matrix Wrapping");
    const size_t dim=5000;
    const std::array<size_t,2> dims({dim,dim});
    MatrixXd A=MatrixXd::Random(dim,dim),
            B=MatrixXd::Random(dim,dim),
            C=MatrixXd::Random(dim,dim);
    tensor_type _A(A),_B(B),_C(C);
    Timer timer;
    MatrixXd D=A+B+C;
    double eigen_time=timer.get_time();
    timer.reset();
    tensor_type E=_A+_B+_C;
    double wrapper_time=timer.get_time();

    std::cout<<"Eigen time (s): "<<eigen_time<<std::endl;
    std::cout<<"Wrapper time (s): "<<wrapper_time<<std::endl;
    const double add_cost_pct=
            std::fabs(eigen_time-wrapper_time)/eigen_time*100.0;
    tester.test("Addition speed <1%",add_cost_pct<1.0);
    return tester.results();
}
