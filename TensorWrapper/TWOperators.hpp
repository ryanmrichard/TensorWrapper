//This file not meant for inclusion outside TensorWrapper.hpp

/** \brief A macro for generating the code for our operator overloading.
 *
 *  For each operator we need several scenarios:
 *
 *  1. TensorWrapper::operator(RHS_t)
 *  2. TensorWrapper::operator(TensorWrapper)
 *  3. TensorWrapper::operator(TensorWrapper')
 *  4. LHS_t::operator(TensorWrapper)
 *
 *  Where LHS_t/RHS_t are any class recognized by the backend (or fundamental
 *  type of the C++ language) and TensorWrapper' is another TensorWrapper
 *  instance.  At least for the moment, each symbol that is overloaded really
 *  only makes sense if the tensors have the same rank.  Although, I've
 *  written the list as if these operators were class members we actually just
 *  overload the free functions in an effort to keep the class cleaner.
 *
 *  \param[in] rv  The return value of the operator (can be auto)
 *  \param[in] sym The operator's symbol (+,-, etc.)
 *  \param[in] name The function to call in TensorWrapperImpl
 *
 */
#define TWOPERATOR(rv,sym,name)\
    template<size_t rank, typename T,\
             TWrapper::detail_::TensorTypes T1,typename RHS_t>\
    rv operator sym(const TWrapper::TensorWrapper<rank,T,T1>& lhs,\
                             const RHS_t& other)\
    {\
        TWrapper::detail_::TensorWrapperImpl<rank,T,T1> impl;\
        return name(lhs.tensor(),other);\
    }\
    template<size_t rank, typename T,TWrapper::detail_::TensorTypes T1>\
    rv\
    operator sym(const TWrapper::TensorWrapper<rank,T,T1>& lhs,\
                 const TWrapper::TensorWrapper<rank,T,T1>& other)\
    {\
        TWrapper::detail_::TensorWrapperImpl<rank,T,T1> impl;\
        return name(lhs.tensor(),other.tensor());\
    }\
    template<size_t rank, typename T, TWrapper::detail_::TensorTypes T1,\
             TWrapper::detail_::TensorTypes T2>\
    rv\
    operator sym(const TWrapper::TensorWrapper<rank,T,T1>& lhs,\
                 const TWrapper::TensorWrapper<rank,T,T2>& other)\
    {\
        TWrapper::detail_::TensorWrapperImpl<rank,T,T1> impl;\
        return name(lhs.tensor(),\
            TWrapper::detail_::TensorConverter<rank,T,T1,T2>::convert(\
                other.tensor()));\
    }\
    template<typename LHS_t,size_t rank, typename T, TWrapper::detail_::TensorTypes T1>\
    rv\
    operator sym(const LHS_t& lhs,\
                 const TWrapper::TensorWrapper<rank,T,T1>& rhs)\
    {\
        TWrapper::detail_::TensorWrapperImpl<rank,T,T1> impl;\
        return name(lhs,rhs.tensor());\
    }

TWOPERATOR(auto,+,impl.add)
TWOPERATOR(auto,-,impl.subtract)
TWOPERATOR(bool,==,impl.are_equal)
TWOPERATOR(bool,!=,!impl.are_equal)
#undef TWOPERATOR

//Some operators we miss with the above macro

///Left multipy by a T
template<size_t rank, typename T, TWrapper::detail_::TensorTypes T1>
decltype(auto)
operator*(T lhs,const TWrapper::TensorWrapper<rank,T,T1>& rhs)
{
    return rhs*lhs;
}

///Right multipy by a T
//template<size_t rank, typename T, TWrapper::detail_::TensorTypes T1>
//decltype(auto)
//operator*(T lhs,const TWrapper::TensorWrapper<rank,T,T1>& rhs)
//{
//    return rhs*lhs;
//}
