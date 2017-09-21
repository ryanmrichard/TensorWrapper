#include <TensorWrapper/Shape.hpp>
#include "TestHelpers.hpp"

using namespace TWrapper;

template<size_t n>
using index_t=std::array<size_t,n>;

int main()
{
    Tester tester("Testing shape class");

    index_t<0> zero0{};
    index_t<1> vec{10};
    index_t<2> mat{10,5};
    index_t<3> tensor{3,3,3};

    //Row major testing
    Shape<0> D0(zero0),D0cm(zero0,false);
    Shape<1> D1(vec),D1cm(vec,false);
    Shape<2> D2(mat),D2cm(mat,false);
    Shape<3> D3(tensor),D3cm(tensor,false);
    tester.test("Scalar equality",D0==D0);
    tester.test("Vector equality",D1==D1);
    tester.test("Matrix equality",D2==D2);
    tester.test("Tensor equality",D3==D3);
    tester.test("Inequality",D0!=D1);

    tester.test("Scalar size",D0.size()==1);
    tester.test("Vector size",D1.size()==10);
    tester.test("Matrix size",D2.size()==50);
    tester.test("Tensor size",D3.size()==27);
    tester.test("Is row major",D0.is_row_major());


    IndexItr<0> D0begin(zero0),D0end(zero0,false);
    tester.test("Scalar begin",D0begin==D0.begin());
    tester.test("Scalar end",D0end==D0.end());

    IndexItr<1> D1begin(vec),D1end(vec,false);
    tester.test("Vector begin",D1begin==D1.begin());
    tester.test("Vector end",D1end==D1.end());

    IndexItr<2> D2begin(mat),D2end(mat,false);
    tester.test("Matrix begin",D2begin==D2.begin());
    tester.test("Matrix end",D2end==D2.end());

    IndexItr<3> D3begin(tensor),D3end(tensor,false);
    tester.test("Tensor begin",D3begin==D3.begin());
    tester.test("Tensor end",D3end==D3.end());

    //column-major testing


    IndexItr<0> D0cmbegin(zero0,true,false),D0cmend(zero0,false,false);
    tester.test("CM Scalar begin",D0cmbegin==D0cm.begin());
    tester.test("CM Scalar end",D0cmend==D0cm.end());
    tester.test("CM is row major",!D0cm.is_row_major());

    IndexItr<1> D1cmbegin(vec,true,false),D1cmend(vec,false,false);
    tester.test("CM Vector begin",D1cmbegin==D1cm.begin());
    tester.test("CM Vector end",D1cmend==D1cm.end());

    IndexItr<2> D2cmbegin(mat,true,false),D2cmend(mat,false,false);
    tester.test("CM Matrix begin",D2cmbegin==D2cm.begin());
    tester.test("CM Matrix end",D2cmend==D2cm.end());

    IndexItr<3> D3cmbegin(tensor,true,false),D3cmend(tensor,false,false);
    tester.test("CM Tensor begin",D3cmbegin==D3cm.begin());
    tester.test("CM Tensor end",D3cmend==D3cm.end());

    return tester.results();
}
