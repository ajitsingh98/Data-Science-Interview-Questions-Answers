# Data Science Interview Questions And Answers

## Maths Questions

## Contents
- [Vector](#vector)
- [Matrices](#matrices)

## Vector

1. Dot product
    1. What’s the geometric interpretation of the dot product of two vectors?
    1. Given a vector $u$ , find vector $v$  of unit length such that the dot product of $u$  and $v$  is maximum.
1. Outer product
    1. Given two vectors $a=[3,2,1]$  and $b=[−1,0,1]$. Calculate the outer product $a^Tb$ ?
    1. Give an example of how the outer product can be useful in ML.
1. What does it mean for two vectors to be linearly independent?
1. Given two sets of vectors $A=a_1,a_2,a_3,...,a_n$  and $B=b_1,b_2,b_3,...,b_m$. How do you check that they share the same basis?
1. Given $n$  vectors, each of $d$  dimensions. What is the dimension of their span?
1. Norms and metrics
    1. What's the norm? What is  $L_0,L_1,L_2,L_{norm}$?
    1. How do norms and metrics differ? Given a norm, make a metric. Given a metric, can we make a norm?

## Matrices

1. Why do we say that matrices are linear transformations?
2. What’s the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?
3. What does the determinant of a matrix represent?
4. What happens to the determinant of a matrix if we multiply one of its rows by a scalar  $t×R$ ?
5. A $4×4$  matrix has four eigenvalues $3,3,2,−1$. What can we say about the trace and the determinant of this matrix?
6. Given the following matrix:
```math
\begin{bmatrix}
1 & 4 & -2\\
-1 & 3 & 2 \\
3 & 5 & -6
\end{bmatrix}
```
Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?

7. What’s the difference between the covariance matrix $A^TA$  and the Gram matrix $AA^T$ ?
8. Given $A∈R^{n×m}$  and $b∈R^n$ 
    1. Find $x$ such that: $Ax=b$.
    1. When does this have a unique solution?
    1. Why is it when $A$ has more columns than rows, $Ax=b$ has multiple solutions?
    1. Given a matrix $A$ with no inverse. How would you solve the equation  $Ax=b$? What is the pseudoinverse and how to calculate it?
9. Derivative is the backbone of gradient descent.
    1. What does derivative represent?
    1. What’s the difference between derivative, gradient, and Jacobian?
10. Say we have the weights $w∈R^{d×m}$  and a mini-batch $x$  of $n$  elements, each element is of the shape $1×d$  so that $x∈R^{n×d}$. We have the output $y=f(x;w)=xw$. What’s the dimension of the Jacobian $\frac{δy}{δx}$?
11. Given a very large symmetric matrix $A$ that doesn’t fit in memory, say $A∈R^{1M×1M}$  and a function $f$ that can quickly compute $f(x)=Ax$ for $x∈R1M$. Find the unit vector $x$ so that $x^TAx$  is minimal.
