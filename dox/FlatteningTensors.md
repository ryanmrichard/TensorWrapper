Flattening Tensors
==================

The purpose of this page is to explain how tensors are mapped to memory.  In
turn this can be thought of as being an explanation for how the IndexItr class
works.

Preliminaries
-------------

Memory can be thought of as a rank 1 tensor (*i.e.* that is the only thing that
matters is the offset).  This in turn means that for any tensor with rank
\f$R\f$ greater than 1 we need to "flatten" the tensor to fit it in memory.
This amounts to establishing a unique mapping between the general rank \f$R\f$
index and its offset in memory.  There's two ways of doing this: row-major and
column-major format.  Note the names stem from how one flattens a matrix.
Row-major format keeps rows together and column-major format keeps columns
together.

By means of notation.  Given our rank \f$R\f$ tensor any element can be
expressed as an \f$R\f$ element vector, \f$\vec{I}\f$, where \f$I_i\f$ is the
offset along the \f$i\f$-th dimension of the tensor.  We further assume that
the length of the dimensions of the tensor are given by an \f$R\f$ element
vector \f$\vec{L}\f$, such that \f$L_i\f$ is the length of the \f$i\f$-th
dimension.

Row Major
---------

In row-major ordering a matrix would be laid out in memory like:

\f[
(0,0), (0,1), ..., (0,L_1-1), (1,0), (1,1), ..., (1,L_1-1), (2,0), ...,
(L_0-1,L_1-1)
\f]

It is straightforward to show that the \f$i,j\f$-th index is mapped to
\f$n\f$-th position in memory as:

\f[
n=i*L_1+j
\f]

A rank 3 tensor would be laid out as:

\f[
(0,0,0), (0,0,1), ..., (0,0,L_2-1), (0,1,0), ..., (0,L_1-1,L_2-1), (1,0,0), ...,
(L_0-1,L_1-1,L_2-1)
\f]

and the \f$i,j,k\f$-th index is mapped to the \f$n\f$-th position as:

\f[
n=i*L_1*L_2+j*L_2+k
\f]

Generalizing, to a rank \f$R\f$ tensor indices are laid out in memory in the
order given by the following algorithm:

1. Starting with \f$I_{R-1}\f$ find the lowest value of \f$i\f$ such that
   \f$I_{R-1-i}+1<L_{R-1-i}\f$.
2. Increment index \f$I_{R-1-i}\f$.
3. Reset indices \f$I_{R-i}\f$ through \f$I_{R-1}\f$ to their initial values.

In turn, the \f$\vec{I}\f$-th index maps to the \f$n\f$-th position as:
\f[
n=\sum_{i=0}^{R-1} I_i\prod_{j=i+1}^{R-1} L_j
\f]

Rewriting the above as:
\f[
n=\sum_{k=0}^{i-1} I_k\prod_{j=k+1}^{R-1}L_j+
  \sum_{k=i+1}^{R-1} I_k\prod_{j=k+1}^{R-1}L_j+
   I_i\prod_{j=i+1}^{R-1} L_j
\f]
where we have isolated \f$I_i\f$. We now aim to divide both sides by
\f$P=\prod_{j=i+1}^{R-1} L_j\f$.  In order to do this note that the products
appearing in the first summation all contain \f$P\f$, thus:
\f[
\frac{\sum_{k=0}^{i-1} I_k\prod_{j=k+1}^{R-1}L_j}{\prod_{j=i+1}^{R-1} L_j}=
      \sum_{k=0}^{i-1} I_k\prod_{j=k+1}^{i}L_j=
      L_i\left(\sum_{k=0}^{i-1} I_k\prod_{j=k+1}^{i-1}L_j\right)=
      Z_iL_i
\f]
where the second equality follows by cancelling terms common to the numerator
and the denominator, the third follows by pulling the \f$L_i\f$ term out of the
summation (it is common to all products), and the final line follows by calling
the summation some integer \f$Z_i\f$, the exact value of which is irrelevant.
Focusing on the second summation we have:
\f[
\frac{\sum_{k=i+1}^{R-1} I_k\prod_{j=k+1}^{R-1}L_j}{\prod_{j=i+1}^{R-1} L_j}=
\sum_{k=i+1}^{R-1}\frac{I_k}{\prod_{j=i+1}^{k} L_j}\lt
\sum_{k=i+1}^{R-1}\frac{L_k}{\prod_{j=i+1}^{k} L_j}=
1+\sum_{k=i+2}^{R-1}\frac{1}{\prod_{j=i+1}^{k-1} L_j}
\f]
The first equality follows from cancelling terms common to the numerator and the
denominator.  The inequality follows from substituting the maximum possible
value for each index in the numerator (*i.e.* using the fact that
\f$I_k\le L_k-1\f$ in the numerator).  The third equality follows from
cancelling the \f$L_k\f$ common to both the numerator and denominator and
pulling the resulting \f$k=i+1\f$ term out of the summation.  Somehow we go
from this to:
\f[
\frac{n}{\prod_{j=i+1}^{R-1} L_j}=Z_iL_i+I_i
\f]
which tells us \f$I_i\f$ is the remainder after dividing the left side by
\f$L_i\f$ thus we can inver the relationship between the offset and the
\f$I_i\f$ by:
\f[
I_i=\left\lfloor\frac{n}{\prod_{j=i+1}^{R-1}L_j}\right\rfloor\mod L_i
\f]



Column Major
------------

In column-major ordering a matrix would be laid out in memory like:

\f[
(0,0), (1,0), ..., (L_0-1,0), (0,1), (1,1), ..., (L_0-1,1), (0,2), ...,
(L_0-1,L_1-1)
\f]

It is straightforward to show that the \f$i,j\f$-th index is mapped to
\f$n\f$-th position in memory as:

\f[
n=i+j*L_0
\f]

A rank 3 tensor would be laid out as:

\f[
(0,0,0), (1,0,0), ..., (L_0-1,0,0), (0,1,0), ..., (L_0-1,L_1-1,0), (0,0,1), ...,
(L_0-1,L_1-1,L_2-1)
\f]

and the \f$i,j,k\f$-th index is mapped to the \f$n\f$-th position as:

\f[
n=i+j*L_0+k*L_0*L_1
\f]

Generalizing, to a rank \f$R\f$ tensor indices are laid out in memory in the
order given by the following algorithm:

1. Starting with \f$I_{0}\f$ find the lowest value of \f$i\f$ such that
   \f$I_{i}+1<L_{i}\f$.
2. Increment index \f$I_{i}\f$.
3. Reset indices \f$I_{0}\f$ through \f$I_{i-1}\f$ to their initial values.

In turn, the \f$\vec{I}\f$-th index maps to the \f$n\f$-th position as:
\f[
n=\sum_{i=0}^{R-1} I_i*\prod_{j=0}^{i-1} L_j
\f]

Using the same logic as the row-major case the inverse relationship is:
\f[
I_i=\left\lfloor\frac{n}{\prod_{j=0}^{i-1}L_j}\right\rfloor\text{mod} L_i
\f]

Cyclic Distributions
--------------------

The above examples use what is called period 1, meaning the next index is
obtained by adding 1 to the previous index's value.  There is no reason the
increment needs to be 1.  Instead let us define another vector \f$\vec{P}\f$
such that \f$P_i\f$ is the period of the \f$i\f$-th dimension.  In row-major
form our matrix is then laid out:

\f[
(0,0), (0,P_1), (0,2P_1), (P_0,0), ...
\f]

and in column-major format the indices are laid out:


\f[
(0,0), (P_0,0), (2P_0,0), (0,P_1), ...
\f]

As should be apparant given a single starting index (here simply the zero
vector) we will not iterate over all indices, but rather only indices for
which \f$I_i\f$ is an integer multiple of \f$P_i\f$.  Admittedly, the cyclic
distributions depicted above are not quite what is usually used in practice.
Instead of cyclically distributing the \f$R\f$ element index one instead
typically cyclically distributes the offsets.  This means one does not typically
end up with nice starting/stopping points like depected, but rather you get
something like (assuming a 3x3 matrix with period 2):

\f[
0,2,4,6,8
\f]

in row-major this means we have elements:
\f[
(0,0), (0,2), (1,1), (2,0), (2,2)
\f]

and in column-major we have elements:
\f[
(0,0), (2,0), (1,1), (0,2), (2,2)
\f]

Note that for this particular period the first row will have two elements, the
second row has one element, and the third row has two elements.  Hence it no
longer

