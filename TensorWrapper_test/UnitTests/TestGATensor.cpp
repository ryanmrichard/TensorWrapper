#include <TensorWrapper/TensorImpl/GA_CXX/GATensor.hpp>
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
    29410,29870,30330,60705,31250,31710,32170,32630,33090,33550,
    29870,30340,30810,62160,31750,32220,32690,33160,33630,34100,
    30330,30810,31290,63615,32250,32730,33210,33690,34170,34650,
    60705,62160,63615,1029105,66525,67980,69435,70890,72345,73800,
    31250,31750,32250,66525,33250,33750,34250,34750,35250,35750,
    31710,32220,32730,67980,33750,34260,34770,35280,35790,36300,
    32170,32690,33210,69435,34250,34770,35290,35810,36330,36850,
    32630,33160,33690,70890,34750,35280,35810,36340,36870,37400,
    33090,33630,34170,72345,35250,35790,36330,36870,37410,37950,
    33550,34100,34650,73800,35750,36300,36850,37400,37950,38500,
};

int main()
{
    Tester tester("Testing Global Arrays C++ Wrapping");
    GAInitialize();

    //Constructors (Destructors and copy constructor implicitly tested)
    int handle=GA_Create_handle();
    std::array<int,2> dims{10,10};
    GA_Set_data(handle,2,dims.data(),C_DBL);
    GA_Allocate(handle);
    GA_Matrix from_handle(handle);
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
    auto my_block=D.my_slice();
    tester.test("Slice start",my_block.first==dim_t{0,0});
    tester.test("Slice end",my_block.second==dim_t{10,10});

    return tester.results();
}
