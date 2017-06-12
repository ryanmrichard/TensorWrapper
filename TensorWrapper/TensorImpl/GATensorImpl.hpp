//Warning!!! This file only meant for inclusion in GATensor.hpp
//This file provides the implementations of GATensor's non-trivial functions

///Namespace for classes/functions which the user is never meant to see.
namespace detail_ {

/** \brief A simple meta type to convert the GATensor's T template type
 *         parameter to the equivalent GA type.
 *
 *  Basically we define this template class (called the primary template) with
 *  no members.  We then specialize it for each C++ type that maps to a GA type.
 *  In those specializations we declare a member variable \p value which will
 *  be the integer corresponding to that type.
 *
 *  As far as I can tell, the recognized types are:
 *  - int which maps to C_INT
 *  - long which maps to C_LONG
 *  - long long which maps to C_LONGLONG
 *  - float which maps to C_FLOAT
 *  - double which maps to C_DBL
 *  - std::complex<float> which maps to C_SCPL
 *  - std::complex<double> which maps to C_DCPL
 *
 *  \note If a user tries to use an unsupported type the reliance on this
 *  template class will cause a compile time error something along the lines of
 *  struct CType2GAType does have a member "type"
 */
template<typename T>
struct CType2GAType{
    //No member value, primary template
};

///Macro for declaring and intstantiating specializations
#define CTYPE2GATYPE_DECL(x,y)\
    template<>\
    struct CType2GAType<x>{\
       static const int value=y;\
    };
CTYPE2GATYPE_DECL(int,C_INT)
CTYPE2GATYPE_DECL(long,C_LONG)
CTYPE2GATYPE_DECL(long long,C_LONGLONG)
CTYPE2GATYPE_DECL(float,C_FLOAT)
CTYPE2GATYPE_DECL(double,C_DBL)
CTYPE2GATYPE_DECL(std::complex<float>,C_SCPL)
CTYPE2GATYPE_DECL(std::complex<double>,C_DCPL)
#undef CTYPE2GATYPE_DECL

//This is a little function for converting an arbitrary container to an array
//of size_t's (unfortunately array doesn't support a common API with other
//containers for filling)
template<size_t rank,typename T,typename Cont_t>
std::array<T,rank> to_array(const Cont_t& con)
{
    std::array<T,rank> rv;
    std::copy(con.begin(),con.end(),rv.begin());
    return rv;
}

//This function subtracts one from each element of an array converting a C++
//end point to a GA end point
template<typename T, typename Cont_t>
Cont_t subtract_1(const Cont_t& con)
{
    Cont_t rv(con);
    std::transform(rv.begin(),rv.end(),rv.begin(),[](T val){return --val;});
    return rv;
}

template<size_t rank>
std::array<int,rank> unit_stride()
{
    std::array<int,rank> rv;
    std::fill(rv.begin(),rv.end(),1);
    return rv;
}

}//End namespace detail

template<size_t rank, typename T>
template<typename Cont_t1, typename Cont_t2>
GATensor<rank,T>::GATensor(const Cont_t1& dims,
                           const Cont_t2& chunks,
                           const char * name):
    handle_(std::make_unique<int>(GA_Create_handle())),
    dims_(detail_::to_array<rank,size_t>(dims)),
    chunks_(detail_::to_array<rank,size_t>(chunks)),
    scale_(1.0),
    transposed_(false)

{
    //This is just a buffer for converted containers
    auto temp=detail_::to_array<rank,int>(dims);
    GA_Set_data(*handle_,rank,temp.data(), detail_::CType2GAType<T>::value);
    if(chunks.size())
    {
        temp=detail_::to_array<rank,int>(chunks);
        GA_Set_chunk(*handle_,temp.data());
    }
    if(name)
        GA_Set_array_name(*handle_,const_cast<char *>(name));
    GA_Allocate(*handle_);
}

template<size_t rank, typename T>
GATensor<rank,T>::GATensor(const GATensor<rank,T>& other):
    GATensor<rank,T>(other.dims_,other.chunks_,nullptr)
{
    scale_=other.scale_;
    if(!other.transposed_)
        GA_Copy(*other.handle_,*handle_);
    else
        GA_Transpose(*other.handle_,*handle_);
}

template<size_t rank,typename T>
GATensor<rank,T>::GATensor(GATensor<rank,T>&& other):
    handle_(std::move(other.handle_)),
    dims_(std::move(other.dims_)),
    chunks_(std::move(other.chunks_)),
    scale_(1.0),
    transposed_(std::move(other.transposed_))
{
    handle_=nullptr;
}

template<size_t rank,typename T>
GATensor<rank,T>& GATensor<rank,T>::operator=(GATensor<rank,T>&& other){
    if(handle_)
        GA_Destroy(*handle_);
    handle_=std::move(other.handle_);
    dims_=std::move(other.dims_);
    chunks_=(std::move(other.chunks_));
    scale_=(std::move(other.scale_));
    transposed_=(std::move(other.transposed_));
    return *this;
}

template<size_t rank,typename T>
GATensor<rank,T>::~GATensor(){
    if(handle_)
        GA_Destroy(*handle_);
}


template<size_t rank,typename T>
template<typename Cont_t1,typename Cont_t2>
void GATensor<rank,T>::set_values(const Cont_t1& low, const Cont_t2& high, const T* values)
{
    if(scale_!=1.0)
    {
        GA_Scale(*handle_,&scale_);
        scale_=1.0;
    }
    if(transposed_)
    {
        *this=std::move(transpose_());
        transposed_=false;
    }
    auto ga_high=detail_::to_array<rank,int>(detail_::subtract_1<T>(high));
    auto ga_low=detail_::to_array<rank,int>(low);
    auto strides=detail_::unit_stride<rank>();
    NGA_Put(*handle_,
            ga_low.data(),ga_high.data(),
            const_cast<T*>(values),
            strides.data());
}

template<size_t rank,typename T>
template<typename Cont_t1,typename Cont_t2>
std::vector<T> GATensor<rank,T>::get_values(const Cont_t1& low, const Cont_t2& high)const
{
    const size_t total_size=std::inner_product(high.begin(),high.end(),
                                         low.begin(),(size_t)1,
                                         std::multiplies<size_t>(),
                                         std::minus<size_t>());
    std::vector<T> rv(total_size);
    auto ga_high=detail_::to_array<rank,int>(detail_::subtract_1<T>(high));
    auto ga_low=detail_::to_array<rank,int>(low);
    auto strides=detail_::unit_stride<rank>();
    NGA_Get(!transposed_?*handle_:*transpose_().handle_,
            ga_low.data(),ga_high.data(),
            rv.data(),
            strides.data());
    if(scale_!=1.0)
        std::transform(rv.begin(), rv.end(), rv.begin(),
                       std::bind1st(std::multiplies<T>(),scale_));
    return rv;
}

template<size_t rank,typename T>
void GATensor<rank,T>::fill(T value)const
{
    GA_Fill(*handle_,&value);
}

template<size_t rank,typename T>
GATensor<rank,T> GATensor<rank,T>::operator+(const GATensor<rank,T>& rhs)const
{
    GATensor C(dims_,0.0);
    GA_Add(const_cast<T*>(&scale_),
           !transposed_?*handle_:*(transpose_().handle_),
           const_cast<T*>(&rhs.scale_),
           !rhs.transposed_?*rhs.handle_:*rhs.transpose_().handle_,
           *C.handle_);
    return C;
}

template<size_t rank,typename T>
GATensor<rank,T>& GATensor<rank,T>::operator+=(const GATensor<rank,T>& rhs)
{
    if(transposed_)
    {
        *this=std::move(transpose_());
        transposed_=false;
    }
    GA_Add(const_cast<T*>(&scale_),*handle_,
           const_cast<T*>(&rhs.scale_),
           !rhs.transposed_?*rhs.handle_:*rhs.transpose_().handle_,
           *handle_);
    scale_=1.0;
    return *this;
}

template<size_t rank,typename T>
GATensor<rank,T> GATensor<rank,T>::operator-(const GATensor<rank,T>& rhs)const
{
    GATensor C(dims_,0.0);
    T temp=-1*rhs.scale_;
    GA_Add(const_cast<T*>(&scale_),
           !transposed_?*handle_:*transpose_().handle_,
           const_cast<T*>(&temp),
           !rhs.transposed_?*rhs.handle_:*rhs.transpose_().handle_,
           *C.handle_);
    return C;
}

template<size_t rank,typename T>
GATensor<rank,T>& GATensor<rank,T>::operator-=(const GATensor<rank,T>& rhs)
{
    T temp=-1*rhs.scale_;
    if(transposed_)
    {
        *this=std::move(transpose_());
        transposed_=false;
    }
    GA_Add(const_cast<T*>(&scale_),*handle_,
           const_cast<T*>(&temp),
           !rhs.transposed_?*rhs.handle_:*rhs.transpose_().handle_,
           *handle_);
    scale_=1.0;
    return *this;
}

template<size_t rank, typename T>
GATensor<rank,T> GATensor<rank,T>::transpose_()const
{
    GATensor rv(*this);
    GA_Transpose(*handle_,*rv.handle_);
    return rv;
}

template<size_t rank,typename T>
void GATensor<rank,T>::print_out()const
{
    GA_Print(*handle_);
}

template<size_t rank, typename T>
 typename std::enable_if<rank==2,GATensor<2, T>>::type  GATensor<rank,T>::operator*(const GATensor<2,T>& rhs)const
{
    char ta=transposed_?'T':'N';
    char tb=rhs.transposed_?'T':'N';
    size_t m=dims()[ta=='N'?0:1];
    size_t n=rhs.dims()[tb=='T'?0:1];
    int k=(int)dims()[ta=='T'?0:1];
    GATensor C(std::array<size_t,2>{m,n});
    T scale=scale_*rhs.scale_;
    T zero=0;
    GA_Dgemm(ta,tb,(int)m,(int)n,k,scale,*handle_,*rhs.handle_,zero,*C.handle_);
    return C;
}
