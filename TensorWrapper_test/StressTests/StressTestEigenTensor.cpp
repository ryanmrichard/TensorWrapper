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
    const size_t dim=argc>1?atoi(argv[1]):10;
    const int nthreads=omp_get_max_threads();
    Eigen::ThreadPool pool(nthreads);
    Eigen::ThreadPoolDevice my_device(&pool,nthreads);
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");
    {

    const std::array<size_t,2> dims({dim,dim});
    tensor_type _A(dims),_B(dims),_C(dims);
    fill_random(_A);
    fill_random(_B);
    fill_random(_C);

    Tensor<double,2> A=_A.data(),B=_B.data(),C=_C.data();

    Timer timer;
    Tensor<double,2> D(dim,dim);
    D.device(my_device)=A+B+C;
    double eigen_time=timer.get_time();
    timer.reset();
    tensor_type _D=_A+_B+_C;
    double wrapper_time=timer.get_time();
    print_times("A+B+C",eigen_time,wrapper_time);
    tester.test("A+B+C",D==_D);

    timer.reset();
    D.device(my_device)=A-B-C;
    eigen_time=timer.get_time();
    timer.reset();
    _D=_A-_B-_C;
    wrapper_time=timer.get_time();
    print_times("A-B-C",eigen_time,wrapper_time);
    tester.test("A-B-C",D==_D);

    timer.reset();
    auto contract1=A.contract(B,idx_array<1>{idx_t{1,0}});
    D.device(my_device)=contract1.contract(C,idx_array<1>{idx_t{1,0}});
    eigen_time=timer.get_time();


    timer.reset();
    _D=_A(i,k)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A*B*C",eigen_time,wrapper_time);
    tester.test("A*B*C",_D==D);

    timer.reset();
    auto contract2=A.contract(B,idx_array<1>{idx_t{0,0}});
    D.device(my_device)=contract2.contract(C,idx_array<1>{idx_t{1,0}});
    eigen_time=timer.get_time();

    timer.reset();
    _D=_A(k,i)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A^T*B*C",eigen_time,wrapper_time);
    tester.test("A^T*B*C",D==_D);
    }

    std::array<size_t,3> dims({dim,dim,dim});
    EigenTensor<3,double> _A(dims),_B(dims),_C(dims);
    fill_random(_A);
    fill_random(_B);
    fill_random(_C);
    Tensor<double,3> A=_A.data(),B=_B.data(),C=_C.data(),D(dim,dim,dim);

    Timer timer;
    //A(i,j,k)*B(i,k,l)=Temp(j,l)
    auto contract1=A.contract(B,idx_array<2>{idx_t{0,0},idx_t{2,1}});
    //Temp(j,l)*C(m,l,n)=D(j,m,n)
    D.device(my_device)=contract1.contract(C,idx_array<1>{idx_t{1,1}});
    double eigen_time=timer.get_time();

    auto m=make_index("m");
    auto n=make_index("n");
    timer.reset();
    EigenTensor<3,double> _D=_A(i,j,k)*_B(i,k,l)*_C(m,l,n);
    double wrapper_time=timer.get_time();
    print_times("A(i,j,k)*B(i,k,l)*C(m,l,n)",eigen_time,wrapper_time);
    tester.test("A(i,j,k)*B(i,k,l)*C(m,l,n)",_D==D);


    return tester.results();
}
