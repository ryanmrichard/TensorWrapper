#include<TensorWrapper/TensorWrapper.hpp>

/* Disclaimer.  I copied these equations out of Helgaker's book.  I'm not sure
 *  if I copied them correctly nor does this file actually contain any way of
 *  updating the amplitudes.
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

    const size_t dims=2;
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
    EigenTensor<2,double> Ftilde=Htilde(p,q);//+
           //Gtilde1(p,q,i,i)*2.0-Gtilde2(p,i,i,q);

    //Singles residuals
    EigenTensor<4,double> u=T2(i,a,j,b)*2.0-T2(j,a,i,b);
    auto OmegaA1=u(k,c,i,d)*Gtilde(a,d,k,c);
    auto OmegaB1=u(k,a,l,c)*Gtilde(k,i,l,c)*-1.0;
    auto OmegaC1=u(i,a,k,c)*Ftilde(k,c);
    auto OmegaD1=Ftilde(i,a);
    EigenTensor<2,double> Omega1=OmegaA1+OmegaB1+OmegaC1+OmegaD1;

    //Doubles residuals
    EigenTensor<4,double> OmegaA2=Gtilde(i,a,j,b)+T2(i,c,j,d)*Gtilde(a,c,b,d);
    EigenTensor<4,double> OmegaB2=T2(k,a,l,b)*Gtilde(k,i,l,j)+T2(k,a,l,b)*T2(i,c,j,d)*Gtilde(k,c,l,d);
    EigenTensor<4,double> OmegaC2=-0.5*T2(k,b,j,c)*Gtilde(k,i,a,c)-
                0.5*T2(k,b,j,c)*T2(l,a,i,d)*Gtilde(k,d,l,c)-
                T2(k,b,i,c)*Gtilde(k,j,a,c)+
                T2(k,b,i,c)*T2(l,a,j,d)*Gtilde(k,d,l,c);
    EigenTensor<4,double> OmegaD2=0.5*u(j,b,k,c)*Ltilde(a,i,k,c)+
                             0.25*u(j,b,k,c)*u(i,a,l,d)*Ltilde(l,d,k,c);
    EigenTensor<4,double> OmegaE2=T2(i,a,j,c)*Ftilde(b,c)-
                                  T2(i,a,j,c)*u(k,b,l,d)*Gtilde(l,d,k,c)-
                                  T2(i,a,k,b)*Ftilde(k,j)-
                                  T2(i,a,k,b)*u(l,c,j,d)*Gtilde(k,d,l,c);

    EigenTensor<4,double> Omega2=OmegaA2(i,a,j,b)+OmegaB2(i,a,j,b)+
                                 OmegaC2(i,a,j,b)+OmegaC2(j,b,i,a)+
                                 OmegaD2(i,a,j,b)+OmegaD2(j,b,i,a)+
                                 OmegaE2(i,a,j,b)+OmegaE2(j,b,i,a);

    //EigenTensor<4,double> L=G(p,q,r,s)*2.0-G(p,s,r,q);
    //auto CCSD_corr_egy=T2(i,a,j,b)*L(i,a,j,b)+T1(i,a)*L(i,a,j,b)*T1(j,b);

    return 0;
}
