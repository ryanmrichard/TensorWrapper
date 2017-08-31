#pragma once
namespace TWrapper {
namespace detail_{

template<typename LHS_Idx, typename RHS_Idx,size_t lrank, size_t rrank>
struct ContractionTraits{};

//Note: all results assume vectors are columns, negate the results for rows


//i*i or i*j
template<typename LHS_Idx, typename RHS_Idx>
struct ContractionTraits<LHS_Idx,RHS_Idx,1,1>
{
    constexpr static bool rows_equal=(LHS_Idx::template get<0>()==
                                      RHS_Idx::template get<0>());
    constexpr static size_t nfree=2*(!rows_equal);
    constexpr static size_t ndummy=1-nfree/2;
    constexpr static bool ltranspose=(nfree==0);
    constexpr static bool rtranspose=(nfree==2);
};

//i * i,j or j * i,j
template<typename LHS_Idx, typename RHS_Idx>
struct ContractionTraits<LHS_Idx,RHS_Idx,1,2>
{
    constexpr static bool rows_equal=(LHS_Idx::template get<0>()==
                                      RHS_Idx::template get<0>());
    constexpr static size_t nfree=1;
    constexpr static size_t ndummy=1;
    constexpr static bool ltranspose=true;
    constexpr static bool rtranspose=!rows_equal;
};


//i,j * j or i,j * i
template<typename LHS_Idx, typename RHS_Idx>
struct ContractionTraits<LHS_Idx,RHS_Idx,2,1>
{
    constexpr static bool rows_equal=(LHS_Idx::template get<0>()==
                                          RHS_Idx::template get<0>());
    constexpr static size_t nfree=1;
    constexpr static size_t ndummy=1;
    constexpr static bool ltranspose=rows_equal;
    constexpr static bool rtranspose=false;
};

//i,j * i,j or i,j*j,i or j,i*i,j or j,i*j,i
//i,j * j,k or i,j*k,j or j,i*j,k or j,i*k,j
template<typename LHS_Idx, typename RHS_Idx>
struct ContractionTraits<LHS_Idx,RHS_Idx,2,2>
{
    constexpr static bool row_equal=LHS_Idx::template get<0>()==
                                    RHS_Idx::template get<0>();
    constexpr static bool row_col=LHS_Idx::template get<0>()==
                                  RHS_Idx::template get<1>();
    constexpr static bool col_equal=LHS_Idx::template get<1>()==
                                    RHS_Idx::template get<1>();
    constexpr static bool col_row=LHS_Idx::template get<1>()==
                                  RHS_Idx::template get<0>();

    constexpr static size_t nfree=4-2*(row_equal+row_col+col_equal+col_row);
    constexpr static size_t ndummy=2-nfree/2;
    constexpr static bool ltranspose=(!nfree? !row_equal : row_equal || row_col);
    constexpr static bool rtranspose=(!nfree? false : col_equal || row_col);
};

///This struct basically maps matrix and vector multiplications for us
template<size_t NFree, size_t NDummy, bool rows_same, bool cols_same>
struct ContractionHelper{};

#define CHelperSpecial(NFree,NDummy,ltranspose,rtranspose,guts)\
template<>\
struct ContractionHelper<NFree,NDummy,ltranspose,rtranspose>{\
template<typename LHS_t,typename RHS_t>\
auto contract(const LHS_t& lhs, const RHS_t& rhs){\
   return guts;\
}}

//i,j * i,j or j,i * j,i
CHelperSpecial(0,2,false,false,lhs.cwiseProduct(rhs).sum());
//i,j * j,i or j,i * i,j
CHelperSpecial(0,2,false,true,lhs.cwiseProduct(rhs.transpose()).sum());
//i,j * j,i or j,i * i,j
CHelperSpecial(0,2,true,false,lhs.transpose().cwiseProduct(rhs).sum());
//i,j * k,j
CHelperSpecial(2,1,false,true,lhs*rhs.transpose());
//j,i * j,k
CHelperSpecial(2,1,true,false,lhs.transpose()*rhs);
//i,j * j,k
CHelperSpecial(2,1,false,false,lhs*rhs);
//j,i * k,j
CHelperSpecial(2,1,true,true,lhs.transpose()*rhs.transpose());
//j,i* i
CHelperSpecial(1,1,false,false,lhs*rhs);
//i*i,j or i,j*i
CHelperSpecial(1,1,true,false,lhs.transpose()*rhs);
//i*j,i
CHelperSpecial(1,1,true,true,lhs.transpose()*rhs.transpose());

//i * i
CHelperSpecial(0,1,true,false,lhs.transpose()*rhs);
//i * j or j*i
CHelperSpecial(2,0,false,true,lhs*rhs.transpose());


#undef CHelperSpecial



}}
