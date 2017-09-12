#pragma once
namespace TWrapper {
namespace detail_{

/** \brief A struct for parsing the indices of a matrix contraction.
 *
 *  In order to determine which path to take we need to map the indices to
 *  transposes.
 */
template<typename LHS_Idx, typename RHS_Idx,size_t lrank, size_t rrank>
struct ContractionTraits{};

//Note: all results assume vectors are columns, negate the results for rows

/** \brief Specilaization to vector vector product
 *
 *   We have two ways to contract two vectors the dot and outer products.
 *   The former looks like:
 *   \f[
 *      T_i*S_i
 *   \f]
 *   and the latter like:
 *   \f[
 *      T_i*S_j
 *   \f]
     Hence if the indices are equal we have 2 dummy and 0 free
 *   indices, whereas if they differ we have 0 dummy and 2 free.  If we assume
 *   our vectors are column vectors, the left must be transposed when we have a
 *   dot product and the right when we have the outter product.
 *
 */
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

/** \brief Specialization to vector matrix product
 *
 *   Here our left side is a vector (assuming column vectors it must always be
 *   transposed to line up with the matrix) and our right side is a matrix.
 *   There are two possiblities:
 *   \f[
 *      V_i*T_{ij}
 *   \f]
 *   or
 *   \f[
 *      V_j*T_{ij}=V_j*T_{ji}^T
 *   \f]
 *   Either way we always have 1 free and 2 dummy index.  It also is clear that
 *   we need to transpose when the index of the vector is not the same as the
 *   row index of the matrix.
 */
template<typename LHS_Idx, typename RHS_Idx>
struct ContractionTraits<LHS_Idx,RHS_Idx,1,2>
{
    constexpr static bool rows_equal=(LHS_Idx::template get<0>()==
                                      RHS_Idx::template get<0>());
    constexpr static size_t nfree=1;
    constexpr static size_t ndummy=2;
    constexpr static bool ltranspose=true;
    constexpr static bool rtranspose=!rows_equal;
};


/** \brief Specialization to matrix vector product
 *
 *   Here our right side is a vector (assuming column vectors it is never
 *   transposed) and our left side is a matrix.
 *   There are two possiblities:
 *   \f[
 *      T_{ij}*V_j
 *   \f]
 *   or
 *   \f[
 *      T_{ij}*V_i=T_{ji}^T*V_i
 *   \f]
 *   Either way we always have 1 free and 2 dummy index.  It also is clear that
 *   we need to transpose when the index of the vector is the same as the
 *   row index of the matrix.
 */
template<typename LHS_Idx, typename RHS_Idx>
struct ContractionTraits<LHS_Idx,RHS_Idx,2,1>
{
    constexpr static bool rows_equal=(LHS_Idx::template get<0>()==
                                      RHS_Idx::template get<0>());
    constexpr static size_t nfree=1;
    constexpr static size_t ndummy=2;
    constexpr static bool ltranspose=rows_equal;
    constexpr static bool rtranspose=false;
};

/** \brief Specialization to matrix matrix product
 *
 *  This is the hard one as there are several possibilities.  The first two are
 *  dot product like:
 *  \f[
 *      T_{ij}*S_{ij}=T_{ji}*S_{ji}
 *  \f]
 *  or
 *  \f[
 *      T_{ij}*S_{ji}=T_{ji}*S_{ij}
 *  \f]
 *  In these cases we have no free indices and four dummy indices.  We never
 *  have to transpose the first matrix and only need to transpose the second if
 *  the the rows aren't indexed the same. The next four are typical matrix
 *  products with transposes:
 *  \f[
 *     T_{ik}*S_{kj}
 *  \f]
 *  \f[
 *      T_{ik}*S_{jk}=T_{ik}*S_{kj}^T
 *  \f]
 *  \f[
 *      T_{ki}*S_{kj}=T_{ik}^T*S_{kj}
 *  \f]
 *  and
 *  \f[
 *      T_{ki}*S_{jk}=T_{ik}^T*S_{kj}^T
 *  \f]
 *  Here the first matrix is transposed if the rows of the matrices or the row
 *  of the first and the column of the second are equal.  The second is
 *  transposed if the columns or the row of the first and the column of the
 *  second are equal.
 */
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
    constexpr static bool is_dot=(row_equal && col_equal)||
                                 (row_col   && col_row);
    constexpr static size_t nfree=is_dot?0:2;
    constexpr static size_t ndummy=is_dot?4:2;
    constexpr static bool ltranspose=(is_dot? false : row_equal || row_col);
    constexpr static bool rtranspose=(is_dot? !row_equal : col_equal || row_col);
};

}}
