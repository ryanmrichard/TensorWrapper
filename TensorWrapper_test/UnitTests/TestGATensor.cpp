#include <TensorWrapper/TensorImpl/GATensor.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;
using GA_Matrix=GATensor<2,double>;
using dim_t=std::array<size_t,2>;

template<typename T, typename T1>
double block_error(const T& rvs,const T1& corr)
{
    return std::inner_product(rvs.begin(),
                       rvs.end(),
                       corr.begin(),
                       0.0,
                       std::plus<double>(),
                       [](double lhs, double rhs){ return std::fabs(lhs-rhs);
            });
}

std::vector<double> mm_corr={
    3355.0,7955.0,12555.0,47070.0,21755.0,26355.0,30955.0,35555.0,40155.0,44755.0,
    3410.0,8110.0,12810.0,48390.0,22210.0,26910.0,31610.0,36310.0,41010.0,45710.0,
    3465.0,8265.0,13065.0,49710.0,22665.0,27465.0,32265.0,37065.0,41865.0,46665.0,
    7380.0,21930.0,36480.0,1015065.0,65580.0,80130.0,94680.0,109230.0,123780.0,138330.0,
    3575.0,8575.0,13575.0,52350.0,23575.0,28575.0,33575.0,38575.0,43575.0,48575.0,
    3630.0,8730.0,13830.0,53670.0,24030.0,29130.0,34230.0,39330.0,44430.0,49530.0,
    3685.0,8885.0,14085.0,54990.0,24485.0,29685.0,34885.0,40085.0,45285.0,50485.0,
    3740.0,9040.0,14340.0,56310.0,24940.0,30240.0,35540.0,40840.0,46140.0,51440.0,
    3795.0,9195.0,14595.0,57630.0,25395.0,30795.0,36195.0,41595.0,46995.0,52395.0,
    3850.0,9350.0,14850.0,58950.0,25850.0,31350.0,36850.0,42350.0,47850.0,53350.0,
};

int main()
{
    Tester tester("Testing Global Arrays C++ Wrapping");
    GAInitialize();

    //Constructors (Destructors and copy constructor implicitly tested)
    GA_Matrix from_handle(0);
    GA_Matrix from_dims_and_chunk(dim_t{10,10},dim_t{1,1},"Tensor A");
    GA_Matrix from_dims(dim_t{10,10},"Tensor B");
    GA_Matrix filled(dim_t{10,10},1.0,"Tensor C");
    from_handle=from_dims_and_chunk;
    GA_Matrix Gimme_A(std::move(from_dims_and_chunk));
    from_handle=std::move(Gimme_A);

    //Get/Set values
    std::vector<double> numbers(100);
    std::iota(numbers.begin(),numbers.end(),1.0);
    from_dims.set_values(dim_t{0,0},dim_t{10,10},numbers.data());
    auto rvs=from_dims.get_values(dim_t{0,0},dim_t{10,10});
    double error=block_error(rvs,numbers);
    tester.test("Set/Get",error<1.0E-8);
    from_dims.set_values(dim_t{3,3},999);
    double el33=from_dims.get_values(dim_t{3,3});
    tester.test("Single element Set/Get",std::fabs(el33-999.0)<1.0E-8);

    //Addition
    std::iota(numbers.begin(),numbers.end(),2.0);
    numbers[33]=1000;//We set this one manually
    GA_Matrix C=from_dims+filled;
    rvs=C.get_values(dim_t{0,0},dim_t{10,10});
    error=block_error(rvs,numbers);
    tester.test("Addition",error<1E-8);
    from_dims+=filled;
    rvs=from_dims.get_values(dim_t{0,0},dim_t{10,10});
    error=block_error(rvs,numbers);
    tester.test("Accumulation",error<1E-8);

    //Subtraction
    std::iota(numbers.begin(),numbers.end(),1.0);
    numbers[33]=999;//We set this one manually
    C=from_dims-filled;
    rvs=C.get_values(dim_t{0,0},dim_t{10,10});
    error=block_error(rvs,numbers);
    tester.test("Subtraction",error<1E-8);
    from_dims-=filled;
    rvs=from_dims.get_values(dim_t{0,0},dim_t{10,10});
    error=block_error(rvs,numbers);
    tester.test("Negative Accumulation",error<1E-8);

    //Transpose
    GA_Matrix from_dimsT=from_dims.transpose();
    error=0.0;
    for(size_t i=0;i<10;++i)
        for(size_t j=0;j<10;++j)
            error+=std::fabs(from_dimsT.get_values(dim_t{i,j})-numbers[j*10+i]);
    tester.test("Transpose",error<1E-8);

    //Multiplication
    GA_Matrix D =from_dimsT*from_dims;
    rvs=D.get_values(dim_t{0,0},dim_t{10,10});
    error=block_error(rvs,mm_corr);
    tester.test("Matrix multiplication",error<1E-8);

    return tester.results();
}
