#include <TensorWrapper/MathLibs.hpp>
#include "TestHelpers.hpp"
#include <time.h>
#include <vector>
#include <stdlib.h>


void print_time(const std::string& name,double time)
{
    std::cout<<"Time for "<<name<<" : "<<time<<" s"<<std::endl;
}


int main(int argc, char** argv)
{
    Tester tester("Establishing BLAS base lines");
    srand (time(NULL));
    const int dim=argc>1?atoi(argv[1]):10;
    const int dim2=dim*dim;
    std::vector<double> A(dim2),B(dim2),C(dim2);
    for(size_t i=0;i<dim2;++i)
    {
        A[i]=rand();
        B[i]=rand();
        C[i]=rand();
    }

    Timer timer;
    cblas_daxpy(dim2,1.0,A.data(),1,B.data(),1);
    cblas_daxpy(dim2,1.0,B.data(),1,C.data(),1);
    double time=timer.get_time();
    print_time("A+B+C",time);

    timer.reset();
    cblas_daxpy(dim2,-1.0,B.data(),1,A.data(),1);
    cblas_daxpy(dim2,-1.0,C.data(),1,A.data(),1);
    time=timer.get_time();
    print_time("A-B-C",time);

    timer.reset();
    //For memory considerations, here we pretend that C is first an empty buffer
    //then we put C into A, and zero B
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                dim,dim,dim,
                1.0,A.data(),dim,B.data(),dim,1.0,C.data(),dim);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                dim,dim,dim,
                1.0,C.data(),dim,A.data(),dim,1.0,B.data(),dim);
    time=timer.get_time();
    print_time("A*B*C",time);

    timer.reset();
    //Same pretened as above
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
                dim,dim,dim,
                1.0,A.data(),dim,B.data(),dim,1.0,C.data(),dim);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                dim,dim,dim,
                1.0,C.data(),dim,A.data(),dim,1.0,B.data(),dim);
    time=timer.get_time();
    print_time("A^T*B*C",time);

    return tester.results();
}
