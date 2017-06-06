#pragma once
#include<iostream>
#include<string>
#include<vector>
#include<sstream>
#include<set>
#include<map>

namespace TWrapper {
namespace detail_ {

class C_String{
private:
    const char* str_;
    size_t end_;

    /** \brief Returns the position of the i-th occurence of a character
     *
     *   \param i which occurance? 1st occurence is i=1, 2nd is i=2, etc.
     *   \param j the current index (used for recursion)
     *   \param k how many instances we found (used for recursion)
     *
     *  \return the index of the i-th instance, if available, otherwise size()
     */
    constexpr size_t find(char x, size_t i, size_t j, size_t k)const
    {
        if(is_char(j,x)){
            return k==i? j : find(x,i,j+1,k+1);
        }
        else if(j+1==size())
        {
            return size();
        }
        else{
            return find(x,i,j+1,k);
        }

    }


    ///For splitting only
    constexpr C_String(const char* str, size_t N)
        :str_(str),end_(N)
    {}

public:
    constexpr C_String():str_(nullptr),end_(0){}

    ///Makes a new c string
    template<size_t N>
    constexpr C_String(const char (&arr)[N])
        :str_(arr),end_(N-1)
    {}

    ///The size of the c string
    constexpr size_t size()const
    {
        return end_;
    }


    ///Returns the n-th character of the string
    constexpr char operator[](size_t n)const
    {
        return str_[n];
    }

    ///True if the n-th character of the string is an x
    constexpr bool is_char(size_t n,char x)const
    {
        return (*this)[n]==x;
    }

    ///Returns the position of the i-th occurence of x
    constexpr size_t find(char x, size_t i)const{
        return find(x,i,0,0);
    }

    ///Returns the i-th substring based on splitting on character
    constexpr C_String split(char x, size_t i)const
    {
        const size_t start = i>0? find(x,i-1)+1: 0;
        const size_t end = find(x,i);
        C_String rv(str_+start,end-start);
        return rv;
    }

    ///Returns true if this C_String equals other C_String
    constexpr bool operator==(const C_String& other)const
    {
        if(size()!=other.size())return false;
        for(size_t i=0;i<size();++i)
            if((*this)[i]!=other[i])return false;
        return true;
    }

    ///Returns tru if these strings are not equal
    constexpr bool operator!=(const C_String& other)const
    {
        return !((*this)==other);
    }
};

template<size_t rank, size_t N> constexpr
std::array<C_String,rank> parse_idx(const char(&arr)[N])
{
    std::array<C_String,rank> rv{};
    C_String temp(arr);
    for(size_t i=0;i<rank;++i)
        rv[i]=temp.split(',',i);
    return rv;
}

///Splits a string into indices based on commas
template<size_t N>
std::array<std::string,N> split_comma(const std::string idx)
{
    std::array<std::string,N> rv=std::array<std::string,N>();
    std::stringstream ss(idx);
    char val;
    size_t i=0;
    while (ss >> val)
    {
        rv[i]+=val;
        while(ss.peek() == ' ')
            ss.ignore();
        if (ss.peek() == ',')
        {
            ++i;
            ss.ignore();
        }

    }
    return rv;

}


/** \brief Updates our contraction information given the new index list
 *
 *  Basically we loop over the indices, if they're in idx2int we've seen it
 *  before and that means it's a repeated index and will be contracted over.
 *  If it's not in that list it's a new index and needs to be recorded.  This
 *  logic is sound even for repeated indices on the same tensor.
 */
template<size_t rank>
void add_contraction(const std::array<std::string,rank>& idx,
                     std::map<std::string,size_t>& idx2int,
                     std::set<std::string>& idx2contract)
{
    for(const auto& x: idx)
    {
        if(idx2int.count(x))//Seen it so it's a contraction index
            idx2contract.insert(x);
        else//Not seen it so it needs added
            idx2int.emplace(x,idx2int.size());
    }
}

}//End namespace detail

template <size_t rank>
struct Indices{
    const std::array<detail_::C_String,rank> idx_;
    template<size_t N>
    constexpr Indices(const char(&arr)[N]):
        idx_(detail_::parse_idx<rank>(arr))
    {}
};


template<size_t rank,typename Tensor_t>
struct IndexedTensor{
    const Tensor_t& tensor_;
    const std::array<std::string,rank> idx_;
    IndexedTensor(const Tensor_t& tensor,const char* idx):
        tensor_(tensor),idx_(detail_::split_comma<rank>(idx))
    {}
};

template<typename LHS_t, typename RHS_t>
struct Contraction{
    const LHS_t& lhs_;
    const RHS_t& rhs_;
    std::map<std::string,size_t> idx2int_;
    std::set<std::string> idx2contract_;

    ///Our first pair of indexed tensors
    template<size_t rank_lhs, size_t rank_rhs,typename _LHS_t,typename _RHS_t>
    Contraction(const IndexedTensor<rank_lhs,_LHS_t>& lhs,
                const IndexedTensor<rank_rhs,_RHS_t>& rhs):
        lhs_(lhs),rhs_(rhs)
    {
        detail_::add_contraction<rank_lhs>(lhs.idx_,idx2int_,idx2contract_);
        detail_::add_contraction<rank_rhs>(rhs.idx_,idx2int_,idx2contract_);
    }

    ///LHS_t is a contraction, take it's information
    template<size_t rank,typename _RHS_t,typename T1, typename T2>
    Contraction(const Contraction<T1,T2>& lhs,
                const IndexedTensor<rank,_RHS_t>& rhs):
        lhs_(lhs),rhs_(rhs),
        idx2int_(std::move(lhs.idx2int_)),
        idx2contract_(std::move(lhs.idx2contract_))
    {
        detail_::add_contraction<rank>(rhs.idx_,idx2int_,idx2contract_);
    }

    ///RHS_t is a contraction, take it's information
    template<size_t rank, typename _LHS_t,typename T1, typename T2>
    Contraction(const IndexedTensor<rank,_LHS_t>& lhs,
                const Contraction<T1,T2>& rhs):
        lhs_(lhs),rhs_(rhs),
        idx2int_(std::move(rhs.idx2int_)),
        idx2contract_(std::move(rhs.idx2contract_))
    {
        detail_::add_contraction<rank>(lhs.idx_,idx2int_,idx2contract_);
    }
};

}

template<size_t rank_lhs, size_t rank_rhs, typename LHS_t, typename RHS_t>
TWrapper::Contraction<TWrapper::IndexedTensor<rank_lhs,LHS_t>,
                      TWrapper::IndexedTensor<rank_rhs,RHS_t>>
operator*(const TWrapper::IndexedTensor<rank_lhs,LHS_t>& lhs,
          const TWrapper::IndexedTensor<rank_rhs,RHS_t>& rhs)
{
    return TWrapper::Contraction<
            TWrapper::IndexedTensor<rank_lhs,LHS_t>,
            TWrapper::IndexedTensor<rank_rhs,RHS_t>>(lhs,rhs);
}

template<size_t rank, typename LHS_t, typename RHS_t>
TWrapper::Contraction<LHS_t,TWrapper::IndexedTensor<rank,RHS_t>>
operator*(const LHS_t& lhs,
          const TWrapper::IndexedTensor<rank,RHS_t>& rhs)
{
    return TWrapper::Contraction<LHS_t,TWrapper::IndexedTensor<rank,RHS_t>>(lhs,rhs);
}

template<size_t rank, typename LHS_t, typename RHS_t>
TWrapper::Contraction< TWrapper::IndexedTensor<rank,LHS_t>,RHS_t>
operator*(const TWrapper::IndexedTensor<rank,LHS_t>& lhs,
          const RHS_t& rhs)
{
    return TWrapper::Contraction<TWrapper::IndexedTensor<rank,LHS_t>,RHS_t>(lhs,rhs);
}
