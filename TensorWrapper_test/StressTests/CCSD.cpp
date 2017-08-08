#include<TensorWrapper/TensorWrapper.hpp>

/* Disclaimer.  I copied these equations out of Helgaker's book.  I'm not sure
 *  if I copied them correctly nor does this file actually contain any way of
 *  updating the amplitudes.  Furthermore at the time of this writing my
 *  lazy evaluation can't infer permutations from indices nor can it distribute.
 *
 *
 * That all said the  point of this file is to simply see how long
 * it takes for this templated mess to compile.
 *
 *
 */
using namespace TWrapper;

int main()
{
    auto i=make_index("i");
    auto j=make_index("j");
    auto k=make_index("k");
    auto l=make_index("l");
    auto a=make_index("a");
    auto b=make_index("b");
    auto c=make_index("c");
    auto d=make_index("d");
    auto p=make_index("p");
    auto q=make_index("q");
    auto r=make_index("r");
    auto s=make_index("s");
    auto mu=make_index("mu");
    auto nu=make_index("nu");
    auto rho=make_index("rho");
    auto sigma=make_index("sigma");

    const size_t dims=10;
    const std::array<size_t,2> dims2{dims,dims};
    const std::array<size_t,4> dims4{dims,dims,dims,dims};

    //Stuff we'd get from elsewhere
    EigenTensor<2,double> H(dims2);
    EigenTensor<4,double> G(dims4);
    EigenTensor<2,double> Ones(dims2);
    EigenTensor<2,double> T1(dims2);
    EigenTensor<4,double> T2(dims4);



    EigenTensor<2,double> X=Ones-T1;
    EigenTensor<2,double> Y=Ones+T1;


    EigenTensor<2,double> Htilde=X(mu,p)*Y(nu,q)*H(mu,nu);
    EigenTensor<4,double> Gtilde=
            X(mu,p)*Y(nu,q)*X(rho,r)*Y(sigma,s)*G(mu,nu,rho,sigma);
    EigenTensor<4,double> Ltilde=
            Gtilde(p,q,r,s)*2.0-Gtilde(p,s,r,q);

    EigenTensor<2,double> Ftilde=Htilde+
            Gtilde(p,q,i,i)*2.0-Gtilde(p,i,i,q);


    //Singles residuals
    EigenTensor<4,double> u=T2(i,a,j,b)*2.0-T2(j,a,i,b);
    auto OmegaA1=u(k,c,i,d)*Gtilde(a,d,k,c);
    auto OmegaB1=u(k,a,l,c)*Gtilde(k,i,l,c)*-1.0;
    auto OmegaC1=u(i,a,k,c)*Ftilde(k,c);
    EigenTensor<2,double> Omega1=OmegaA1+OmegaB1+OmegaC1+Ftilde;

    //Doubles residuals
    auto OmegaA2=Gtilde+T2(i,c,j,d)*Gtilde(a,c,b,d);
    auto OmegaB2=T2(k,a,l,b)*Gtilde(k,i,l,j)+
                 T2(k,a,l,b)*T2(i,c,j,d)*Gtilde(k,c,l,d);
    auto OmegaC2=T2(k,b,j,c)*Gtilde(k,i,a,c)*-0.5+
                 T2(k,b,j,c)*T2(l,a,i,d)*Gtilde(k,d,l,c)*0.25-
                 T2(k,b,i,c)*Gtilde(k,j,a,c)+
                 T2(k,b,i,c)*T2(i,a,j,d)*Gtilde(k,d,l,c);
    auto OmegaD2=u(j,b,k,c)*Ltilde(a,i,k,c)*0.5+
                 u(j,b,k,c)*u(i,a,l,d)*Ltilde(l,d,k,c)*0.25;
    auto OmegaE2=T2(i,a,j,c)*Ftilde(b,c)-
                 T2(i,a,j,c)*u(k,b,l,d)*Gtilde(l,d,k,c)-
                 T2(i,a,k,b)*Ftilde(k,j)-
                 T2(i,a,k,b)*u(i,c,j,d)*Gtilde(k,d,l,c);

    //C2, D2, and E2 i,j and a,b should be permuted on their second apperance,
    //but I don't support that at the moment.
    EigenTensor<4,double> Omega2=OmegaA2+OmegaB2+OmegaC2+OmegaC2+
            OmegaD2+OmegaD2+
            OmegaE2+OmegaE2;

    EigenTensor<4,double> L=G(p,q,r,s)*2.0-G(p,s,r,q);
    auto CCSD_corr_egy=T2(i,a,j,b)*L(i,a,j,b)+T1(i,a)*L(i,a,j,b)*T1(j,b);
    return 0;
}
