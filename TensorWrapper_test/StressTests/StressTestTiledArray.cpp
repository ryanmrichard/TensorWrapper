#include <TensorWrapper/TensorWrapper.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using namespace TiledArray;
using tensor_type=TensorWrapper<2,double,detail_::TensorTypes::TiledArray>;

void print_times(const std::string& msg, double eigen_time, double twtime)
{
    std::cout<<msg<<" TiledArray time (s): "<<eigen_time<<std::endl;
    std::cout<<msg<<" TensorWrapper time (s): "<<twtime<<std::endl;
}


int main(int argc, char** argv)
{
    Tester tester("Stress Testing TiledArray Wrapping");
    const size_t dim=argc>1?atoi(argv[1]):10000;
    const size_t nblocks=10;
    if(!dim%nblocks)
        throw std::runtime_error("length is not evenly divisible by nblocks");
    const size_t parts=dim/nblocks;
    RunTime rt(argc,argv);

    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");
#ifdef ENABLE_tiledarray
    TA::World& world = TA::get_default_world();
    {
    const std::array<size_t,2> dims({dim,dim});

    std::vector<size_t> cuts;
    for(size_t i=0,counter=0;i<=dim;i+=parts,++counter)
        cuts.push_back(i);
    std::vector<TiledArray::TiledRange1>
        ranges(2, TiledArray::TiledRange1(cuts.begin(), cuts.end()));
    TiledArray::TiledRange trange(ranges.begin(),ranges.end());

    TiledArray::TArrayD A(world,trange),B(world,trange),C(world,trange);
    A.fill_local(4.0);
    B.fill_local(5.0);
    C.fill_local(6.0);
    tensor_type _A(A),_B(B),_C(C);

    Timer timer;
    TiledArray::TArrayD D;
    D("i,j")=A("i,j")+B("i,j")+C("i,j");
    double eigen_time=timer.get_time();
    timer.reset();
    tensor_type _D=_A(i,j)+_B(i,j)+_C(i,j);
    double wrapper_time=timer.get_time();
    print_times("A+B+C",eigen_time,wrapper_time);
    tester.test("A+B+C",D==_D);

    timer.reset();
    D("i,j")=A("i,j")-B("i,j")-C("i,j");
    eigen_time=timer.get_time();
    timer.reset();
    _D=_A(i,j)-_B(i,j)-_C(i,j);
    wrapper_time=timer.get_time();
    print_times("A-B-C",eigen_time,wrapper_time);
    tester.test("A-B-C",D==_D);

    timer.reset();
    D("i,l")=A("i,j")*B("j,k")*C("k,l");
    eigen_time=timer.get_time();


    timer.reset();
    _D=_A(i,k)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A*B*C",eigen_time,wrapper_time);
    tester.test("A*B*C",_D==D);

    timer.reset();
    D("i,l")=A("j,i")*B("j,k")*C("k,l");
    eigen_time=timer.get_time();

    timer.reset();
    _D=_A(k,i)*_B(k,l)*_C(l,j);
    wrapper_time=timer.get_time();
    print_times("A^T*B*C",eigen_time,wrapper_time);
    tester.test("A^T*B*C",D==_D);
    }

    std::vector<size_t> cuts;
    for(size_t i=0,counter=0;i<=dim;i+=parts,++counter)
        cuts.push_back(i);
    std::vector<TiledArray::TiledRange1>
        ranges(3, TiledArray::TiledRange1(cuts.begin(), cuts.end()));
    TiledArray::TiledRange trange(ranges.begin(),ranges.end());

    std::array<size_t,3> dims({dim,dim,dim});
    TiledArray::TArrayD A(world,trange),B(world,trange),C(world,trange),D;
    A.fill_local(3.0);
    B.fill_local(4.0);
    C.fill_local(5.0);

    TensorWrapper<3,double,detail_::TensorTypes::TiledArray> _A(A),_B(B),_C(C),_D;

    Timer timer;
    D("j,m,n")=A("i,j,k")*B("i,k,l")*C("m,l,n");
    double eigen_time=timer.get_time();

    auto m=make_index("m");
    auto n=make_index("n");
    timer.reset();
    _D=_A(i,j,k)*_B(i,k,l)*_C(m,l,n);
    double wrapper_time=timer.get_time();
    print_times("A(i,j,k)*B(i,k,l)*C(m,l,n)",eigen_time,wrapper_time);
    tester.test("A(i,j,k)*B(i,k,l)*C(m,l,n)",_D==D);

#endif
    return tester.results();
}
