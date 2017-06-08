#pragma once
#include<iostream>
#include<string>
#include<vector>
#include<sstream>
#include<set>
#include<map>
#include <tuple>
#include <utility>
namespace TWrapper {

namespace detail_ {

///A class for wrapping a string literal and giving it a more C++-like feel
class C_String{
private:
    ///A pointer to the beginning of the wrapped string
    const char* str_;

    ///The pointer offset just past the end of our actual string (i.e. the
    ///effective '\0' character)
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
    ///Makes an unusable string for initializing containers
    constexpr C_String():str_(nullptr),end_(0){}

    ///Makes a new C_String from a string literal
    template<size_t N>
    constexpr C_String(const char (&arr)[N])
        :str_(arr),end_(N-1)
    {}

    ///The number of char's in our string
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

    ///Returns true if these strings are not equal
    constexpr bool operator!=(const C_String& other)const
    {
        return !((*this)==other);
    }

    ///Returns true if this is less than other, lexographically speaking
    constexpr bool  operator<(const C_String& other)const
    {
        if((*this)==other)return false;
        const size_t min= size()<other.size()?size():other.size();
        for(size_t i=0;i<min;++i)
            if((*this)[i]>other[i])return false;
        return true;
    }

    constexpr bool operator<=(const C_String& other)const
    {
        return (*this)==other || (*this)< other;
    }

    constexpr bool operator>(const C_String& other)const
    {
        return other<(*this);
    }

    constexpr bool operator>=(const C_String& other)const
    {
        return other<=(*this);
    }
};

///Helper function for splitting the indices provided by the user
template<size_t rank, size_t N> constexpr
std::array<C_String,rank> parse_idx(const char(&arr)[N])
{
    std::array<C_String,rank> rv{};
    C_String temp(arr);
    for(size_t i=0;i<rank;++i)
        rv[i]=temp.split(',',i);
    return rv;
}


template<size_t i,typename...Args> struct ContractionTraits;

template<size_t i,typename T,typename...Args>
struct ContractionTraits<i,T,Args...>: public ContractionTraits<i+1,Args...>{
    static const size_t tensor_number=i;
    static const size_t my_rank=T::my_rank;
    static const size_t sum_of_ranks=T::my_rank+ContractionTraits<i+1,Args...>::sum_of_ranks;
    using idx_array_t=std::array<C_String,sum_of_ranks>;
    using tensor_t= typename T::Tensor_Type;
};

template<size_t i>
struct ContractionTraits<i>{
    static const size_t tensor_number=i;
    static const size_t my_rank=0;
    static const size_t sum_of_ranks=0;
};


///Recursion to get rank of tensor j
template<size_t n,typename...Args> constexpr
typename std::enable_if<n == sizeof...(Args), size_t>::type
get_rank(size_t,const std::tuple<Args...>&)
{
   return 0;
}

template<size_t n,typename...Args> constexpr
typename std::enable_if<n < sizeof...(Args), size_t>::type
get_rank(size_t j,const std::tuple<Args...>& tuple)
{
    if(n==j)
        return ContractionTraits<n,Args...>::my_rank;
    else
        return get_rank<n+1>(j,tuple);
}

///Recursion to get index i of tensor j
template<size_t n,typename...Args> constexpr
typename std::enable_if<n == sizeof...(Args), C_String>::type
get_index(size_t,size_t,const std::tuple<Args...>&)
{
   return C_String{};
}

template<size_t n,typename...Args> constexpr
typename std::enable_if<n < sizeof...(Args), C_String>::type
get_index(size_t i,size_t j,const std::tuple<Args...>& tuple)
{
    if(n==j)
        return std::get<n>(tuple).idx_.idx_[i];
    else
        return get_index<n+1>(i,j,tuple);
}

template<size_t n,typename...Args> constexpr
typename std::enable_if<n == sizeof...(Args), size_t>::type
get_position(C_String,size_t,const std::tuple<Args...>&)
{
    return 0;
}

template<size_t n,typename...Args> constexpr
typename std::enable_if<n < sizeof...(Args), size_t>::type
get_position(C_String idx,size_t j,const std::tuple<Args...>& tuple)
{
    if(n==j){
        const auto& tensor = std::get<n>(tuple);
        for(size_t i=0;i<tensor.get_rank();++i)
            if(tensor.idx_.idx_[i]==idx)
                return i;
    }
    return get_position<n+1>(idx,j,tuple);
}

template<typename...Args> constexpr
size_t sum_of_ranks(const std::tuple<Args...>&)
{
    return ContractionTraits<0,Args...>::sum_of_ranks;
}

template<typename...Args> constexpr
typename ContractionTraits<0,Args...>::idx_array_t
get_indices(const std::tuple<Args...>& tuple)
{
    typename ContractionTraits<0,Args...>::idx_array_t idxs_{};
    for(size_t i=0,counter=0;i<sizeof...(Args);++i)
        for(size_t j=0; j<get_rank<0>(i,tuple);++j,++counter)
            idxs_[counter]=get_index<0>(j,i,tuple);
    return idxs_;
}


}//End namespace detail

///A compile-time class for wrapping the index string the user provided
template <size_t rank>
struct Indices{
    using String_t=detail_::C_String;
    const std::array<String_t,rank> idx_;


    ///Makes a new Indices object from a string literal
    template<size_t N>
    constexpr Indices(const char(&arr)[N]):
        idx_(detail_::parse_idx<rank>(arr))
    {}
};

///A wrapper around the tensor and its indices
template<size_t rank,typename Tensor_t>
struct IndexedTensor{
    const Tensor_t& tensor_;
    const Indices<rank> idx_;
    static const size_t my_rank=rank;
    using Tensor_Type=Tensor_t;
    constexpr size_t get_rank()const
    {
        return rank;
    }

    template<size_t N> constexpr
    IndexedTensor(const Tensor_t& tensor, const char(&idx)[N]):
        tensor_(tensor),idx_(idx)
    {}

    constexpr size_t get_index(const detail_::C_String& stridx)const{
        for(size_t i=0;i<idx_.idx_.size();++i)
            if(idx_.idx_[i]==stridx)return i;
        return idx_.idx_.size();
    }
};

///Description of a contraction
template<typename...Args>
struct Contraction{
private:
    //There's a bit of code duplication between this and the next function

    ///Returns the number of free indices if want_free==true otherwise the
    ///number of indices in the contraction
    constexpr size_t index_count_kernel(bool want_free)const
    {
        auto indices=detail_::get_indices(tensors_);
        std::set<detail_::C_String> free;
        std::set<detail_::C_String> contract;
        for(size_t i=0;i<indices.size();++i)
        {
            const auto& x=indices[i];
            if(contract.count(x))continue;
            for(size_t j=0;j<indices.size();++j)
            {
                if(i==j)continue;
                if(x==indices[j])
                {
                    contract.insert(x);
                    break;
                }
            }
            if(!contract.count(x))
                free.insert(x);
         }
        return want_free?free.size():contract.size();
    }

    ///Similar to above, except returns the k-th index of the desired type
    constexpr detail_::C_String index_get_kernel(size_t k,bool want_free)const
    {
        auto indices=detail_::get_indices(tensors_);
        std::set<detail_::C_String> free;
        std::set<detail_::C_String> contract;
        for(size_t i=0;i<indices.size();++i)
        {
            const auto& x=indices[i];
            if(contract.count(x))continue;
            for(size_t j=0;j<indices.size();++j)
            {
                if(i==j)continue;
                if(x==indices[j])
                {
                    if(contract.size()==k && !want_free)
                        return x;
                    contract.insert(x);
                    break;
                }
            }
            if(!contract.count(x))
            {
                if(free.size()==k && want_free)
                    return x;
                free.insert(x);
            }
         }
        return detail_::C_String{};
    }

public:
    ///The actual tensors
    const std::tuple<Args...> tensors_;

    using idx_array_t=typename detail_::ContractionTraits<0,Args...>::idx_array_t;
    using idx_count_array_t=typename std::array<size_t,detail_::ContractionTraits<0,Args...>::sum_of_ranks>;

    constexpr Contraction(const std::tuple<Args...>& tensors):
        tensors_(tensors)
    {}

    constexpr idx_array_t get_indices()const
    {
        return detail_::get_indices(tensors_);
    }

    ///Returns the number of tensors in the contraction
    constexpr size_t n_tensors()const{
        return std::tuple_size<std::tuple<Args...>>::value;
    }

    ///Returns the rank of tensor i
    constexpr size_t rank(const size_t i)const
    {
        return detail_::get_rank<0>(i,tensors_);
    }

    ///Returns the i-th index of the j-th tensor
    constexpr detail_::C_String get_index(size_t i,size_t j)const{
        return detail_::get_index<0>(i,j,tensors_);
    }

    ///Returns the number of free indices
    constexpr size_t n_free_indices()const
    {
       return index_count_kernel(true);
    }

    ///Returns the number of indices involved in contractions
    constexpr size_t n_contraction_indices()const
    {
       return index_count_kernel(false);
    }

    ///Returns the i-th free index
    constexpr detail_::C_String free_index(size_t i)const
    {
        return index_get_kernel(i,true);
    }

    ///Returns the i-th contraction index
    constexpr detail_::C_String contraction_index(size_t i)const
    {
        return index_get_kernel(i,false);
    }

    ///Returns the position of index in the i-th tensor
    constexpr size_t get_position(const detail_::C_String& idx, size_t i)const
    {
        return detail_::get_position<0>(idx,i,tensors_);
    }

};

///Given a contraction returns an array of the indices to be contracted over
template<typename...Args>
std::vector<detail_::C_String> get_contraction_list(const Contraction<Args...>& ct)
{
    const size_t ncontract=ct.n_contraction_indices();
    std::vector<detail_::C_String> idx(ncontract);
    for(size_t i=0;i<ncontract;++i)
        idx[i]=ct.contraction_index(i);
    return idx;
}

///Given a contraction returns an array of the indices remaining after contraction
template<typename...Args>
std::vector<detail_::C_String> get_free_list(const Contraction<Args...>& ct)
{
    const size_t n_free=ct.n_free_indices();
    std::vector<detail_::C_String> idx_(n_free);
    for(size_t i=0;i<n_free;++i)
        idx_[i]=ct.free_index(i);
    return idx_;
}



}//End namespace TWrapper

///Makes the first contraction in a series
template<size_t lhs_rank,typename LHS_t,size_t rhs_rank,typename RHS_t>
constexpr TWrapper::Contraction<TWrapper::IndexedTensor<lhs_rank,LHS_t>,
                      TWrapper::IndexedTensor<rhs_rank,RHS_t>>
operator*(const TWrapper::IndexedTensor<lhs_rank,LHS_t>& lhs,
          const TWrapper::IndexedTensor<rhs_rank,RHS_t>& rhs)
{
    return TWrapper::Contraction<TWrapper::IndexedTensor<lhs_rank,LHS_t>,
            TWrapper::IndexedTensor<rhs_rank,RHS_t>>(std::make_tuple(lhs,rhs));
}

///Makes second and on up contractions
template<size_t rank,typename Tensor_t,typename...Args>constexpr
TWrapper::Contraction<Args...,TWrapper::IndexedTensor<rank,Tensor_t>>
operator*(const TWrapper::Contraction<Args...>& lhs,
          const TWrapper::IndexedTensor<rank,Tensor_t>& rhs)
{
    return TWrapper::Contraction<Args...,TWrapper::IndexedTensor<rank,Tensor_t>>
                (std::tuple_cat(lhs.tensors_,std::make_tuple(rhs)));
}

///Allows printing of C_String
std::ostream& operator<<(std::ostream& os,const TWrapper::detail_::C_String& str)
{
    for(size_t i=0;i<str.size();++i)
        os<<str[i];
    return os;
}
