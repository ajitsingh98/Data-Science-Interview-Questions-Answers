# Data Science Interview Questions And Answers

## Vector and Matrices

Contents
---
- [Vector](#vector)
- [Matrices](#matrices)

---

## Vector

Q. Define following terms
  1. Scalers
  2. Vectors
  3. Matrices
  4. Tensors

<details><summary><b>Answer</b></summary>

1. Scalers : A scaler is just a single number. example - $1, 2, 3$ etc
2. Vectors : A vector is an array of numbers. It is like identifying points in the space, with each element giving the coordinate along a different axis.

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
$$

3. Matrices : A matrix is 2D array of numbers. 

Let $\mathbf{A}$ be a matrix defined as:

$$
\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}
$$

4. Tensors: An array of numbers arranged on a regular grid with a variable number of axes is known as a tensor.

Let $\mathcal{T}$ be a 3-dimensional tensor defined as:

$$
\mathcal{T} = \begin{bmatrix}
\begin{bmatrix} t_{111} & t_{112} \\ t_{121} & t_{122} \end{bmatrix}, &
\begin{bmatrix} t_{211} & t_{212} \\ t_{221} & t_{222} \end{bmatrix}
\end{bmatrix}
$$

</details>

---

Q. What is broadcasting in matrices in context of deep learning?

<details><summary><b>Answer</b></summary>

In deep learning we allow the addition of vector and matrix, yielding another matrix: $C = A+b$, where $C_{i, j} = A_{i, j}+b_j$. The vector $b$ is being added to each row of the matrix. The implicit copying of $b$ to many location is called broadcasting.

</details>

---

Q. Dot product
  1. What’s the geometric interpretation of the dot product of two vectors?
  1. Given a vector $u$ , find vector $v$  of unit length such that the dot product of $u$  and $v$  is maximum.

<details><summary><b>Answer</b></summary>

1. Let $\vec{A}= ⟨a_1....a_k⟩$ and $\vec{B}= ⟨b_1....b_k⟩$  be k-dimensional vectors. The dot product of $\vec{A}$ and $\vec{B}$ , denoted $\vec{A} \cdot \vec{B}$ is a number, defined as follows

$$\vec{A} \cdot \vec{B} = a_1b_1+a_2b_2+....+a_kb_k$$

The dot product has the following geometric interpretation: Let $\alpha$ be the angle between $\vec{A}$ and $\vec{B}$. Then 

$$\vec{A} \cdot \vec{B} = |\vec{A}| \cdot |\vec{B}| \cdot \cos(\alpha)$$

2. To find a vector $ v $ of unit length such that the dot product of $ u $ and $ v $ is maximum, we want to maximize the expression for the dot product $ u \cdot v $

According to the formula for the dot product:

$$u \cdot v = |u| |v| \cos(\theta)$$

where:
- $ |u| $ is the magnitude of $ u $,
- $ |v| $ is the magnitude of $ v $ (which is 1 in this case because $ v $ is a unit vector),
- $\theta$ is the angle between $ u $ and $ v $.

To maximize $ u \cdot v $, the $\cos(\theta)$ part must be maximized. The cosine of an angle reaches its maximum value of 1 when the angle $\theta$ is 0 degrees, meaning that the vectors $ u $ and $ v $ must be pointing in the same direction.

Thus, vector $ v $ should be a unit vector in the direction of $ u $. This can be achieved by normalizing $ u $. The normalization of $ u $ is done by dividing $ u $ by its magnitude. If $ u $ is represented as $u = (u_1, u_2, \ldots, u_n)$ and its magnitude $ |u| $ is given by:

$$|u| = \sqrt{u_1^2 + u_2^2 + \ldots + u_n^2}$$

Then, the unit vector $ v $ in the direction of $ u $ is:

$$v = \frac{u}{|u|} = \left(\frac{u_1}{|u|}, \frac{u_2}{|u|}, \ldots, \frac{u_n}{|u|}\right)$$

This vector $ v $ will have a unit length and the dot product $ u \cdot v $ will be maximum, equal to the magnitude of $ u $ (since $ u \cdot v = |u| \cdot 1 \cdot \cos(0^\circ) = |u| $).

</details>

---

Q. Outer product
  1. Given two vectors $a=[3,2,1]$  and $b=[−1,0,1]$. Calculate the outer product $a^Tb$ ?
  1. Give an example of how the outer product can be useful in ML.

<details><summary><b>Answer</b></summary>

1. resultant product will be a $3 \times 3$ matrix, which can be given as follows:

$$
\left[\begin{matrix}
    -3 & 0 & 3 \\\
    -2 & 0 & 2 \\\
    -1 & 0 & 1
\end{matrix}\right]
$$

2. One useful application of the outer product in machine learning is in the computation of covariance matrices, where the outer product is used to calculate the covariance of different feature vectors. For instance, the covariance matrix can be estimated as the average outer product of the centered data vectors (i.e., data vectors from which the mean has been subtracted). This is crucial for algorithms that rely on data distribution, such as Principal Component Analysis (PCA) and many types of clustering algorithms.


</details>

---

Q. What does it mean for two vectors to be linearly independent?

<details><summary><b>Answer</b></summary>

Two vectors are said to be **linearly independent** if no vector in the set can be written as a linear combination of the others. In simpler terms, neither of the vectors can be expressed as a scalar multiple or a combination involving scalar multiples of the other vector.

For two vectors $\vec{a}$ and $\vec{b}$, they are linearly independent if the only solution to the equation

$$c_1 \vec{a} + c_2 \vec{b} = \vec{0}$$

is $c_1 = 0$ and $c_2 = 0$, where $c_1$ and $c_2$ are scalars and $\vec{0}$ is the zero vector. 

</details>

---

Q. Given two sets of vectors $A=a_1,a_2,a_3,...,a_n$  and $B=b_1,b_2,b_3,...,b_m$. How do you check that they share the same basis?

<details><summary><b>Answer</b></summary>

</details>

---

Q. How can we inspect if two vectors are orthogonal?

<details><summary><b>Answer</b></summary>

Two vectors are orthogonal to each other if $\vec{a} \cdot \vec{b} = 0$. Note that they both should have non-zero norm i.e any of the two should not be a zero vector.

</details>

---

Q. How to check if two vectors are orthonormal?

<details><summary><b>Answer</b></summary>

Two vectors are orthonormal to each other if they are orthogonal and both have unit norm.

</details>

---

Q. Given $n$  vectors, each of $d$  dimensions. What is the dimension of their span?


<details><summary><b>Answer</b></summary>

$$
\text{Dimension of the span} = \min(n, d)
$$

</details>

---

Q. Norms and metrics
1. What's the norm? What is  $L_0,L_1,L_2,L_{norm}$?
1. How do norms and metrics differ? Given a norm, make a metric. Given a metric, can we make a norm?

<details><summary><b>Answer</b></summary>

1. A **norm** on a vector space is a function that assigns a non-negative length or size to vectors, except for the zero vector, which is assigned a length of zero. Norms are denoted by $\|\cdot\|$ and must satisfy the following properties for any vectors $x, y$ and any scalar $a$:
  - **Non-negativity**: $\|x\| \geq 0$ and $\|x\| = 0$ if and only if $x = 0$.
  - **Scalar multiplication**: $\|a \cdot x\| = |a| \cdot \|x\|$.
  - **Triangle inequality**: $\|x + y\| \leq \|x\| + \|y\|$.

  Different types of norms can be defined on vector spaces:

- **$L_0$ norm** (not a true norm): It counts the number of non-zero entries in a vector. It does not satisfy the triangle inequality or the homogeneity property (scalar multiplication), which is why it's technically not a norm.
- **$L_1$ norm**: It is defined as $\|x\|_1 = \sum |x_i|$, summing the absolute values of the entries of the vector.
- **$L_2$ norm** (Euclidean norm): It is defined as $\|x\|_2 = \sqrt{\sum x_i^2}$, which corresponds to the usual geometric length of a vector.
- **$L_p$ norm**: It generalizes the $L_1$ and $L_2$ norms and is defined as $\|x\|_p = (\sum |x_i|^p)^{1/p}$ for $1 \leq p < \infty$.

2. Norm vs Metric
- A **norm** provides a way to measure the length of vectors in vector spaces.
- A **metric** is a more general function that defines a distance between any two elements in a set, satisfying:
  1. **Non-negativity**: $d(x, y) \geq 0$ and $d(x, y) = 0$ if and only if $x = y$.
  2. **Symmetry**: $d(x, y) = d(y, x)$.
  3. **Triangle inequality**: $d(x, z) \leq d(x, y) + d(y, z)$.

**Given a norm, make a metric**

If you have a norm $\|\cdot\|$ on a vector space, you can define a metric $d$ by $d(x, y) = \|x - y\|$. This metric satisfies all the metric properties, derived from the properties of the norm.

**Given a metric, can we make a norm?**

Not all metrics come from norms. To derive a norm from a metric $d$, the metric must satisfy additional properties:
1. **Translation invariance**: $d(x+z, y+z) = d(x, y)$ for all $x, y, z$.
2. **Homogeneity**: $d(\alpha x, \alpha y) = |\alpha| d(x, y)$ for all scalars $\alpha$.

If a metric satisfies these conditions, it can be associated with a norm, where the norm $\|x\|$ can be defined as $d(x, 0)$. However, many metrics (like the discrete metric) do not satisfy these properties and thus cannot be associated with a norm.

</details>

## Matrices

---

Q. Explain transpose operation on matrices?

<details><summary><b>Answer</b></summary>

The transpose of a matrix is mirror image of the matrix across a diagonal line, called as main diagonal. We denote the transpose of a matrix $\mathbf{A}$ as $\mathbf{A^T}$. 

It is defined as:

$$
\mathbf{A^T}_{i, j} = \mathbf{A_{j, i}}
$$

</details>

---

Q. Define the condition under which we can multiply two matrices?

<details><summary><b>Answer</b></summary>

Suppose we have two matrices $\mathbf{A}$ and $\mathbf{B}$, In order to define the product of $\mathbf{A}$ and $\mathbf{B}$, $\mathbf{A}$ must have same number of columns as $\mathbf{B}$ has rows. 

If shape of $\mathbf{A}$ is $m x n$ and shape of $\mathbf{B}$ is $n x p$, then the resultant matrix will have shape of $m x p$.

$$
\mathbf{C_{m \times p}} = \mathbf{A_{m \times n}}\mathbf{B_{n \times p}}
$$

Where, 

$$
C_{i, j} = \sum_k A_{i, k}B_{k, j}
$$


</details>

---

Q. What is the Hadamard product?

<details><summary><b>Answer</b></summary>

The Hadamard product, also known as the element-wise product, is an operation that takes two matrices of the same dimensions and produces another matrix of the same dimensions, where each element is the product of the corresponding elements of the input matrices. 

Mathematically, if $\mathbf{A} = [a_{ij}]$ and $\mathbf{B} = [b_{ij}]$ are two matrices of the same size, then the Hadamard product $\mathbf{C} = \mathbf{A} \circ \mathbf{B}$ is defined as:

$$
\mathbf{C} = \begin{bmatrix} a_{11} \cdot b_{11} & a_{12} \cdot b_{12} & \cdots \\ a_{21} \cdot b_{21} & a_{22} \cdot b_{22} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}
$$

where $ c_{ij} = a_{ij} \cdot b_{ij} $.

</details>

--- 

Q. Where do we use Hadamard product?

<details><summary><b>Answer</b></summary>

The Hadamard product is commonly used in various applications such as signal processing, neural networks, and other fields where element-wise operations are needed.

</details>

--- 

Q. How is the Hadamard product different from the dot product?

<details><summary><b>Answer</b></summary>

The Hadamard product and dot product are distinct operations:

1. **Operation**:
   - **Hadamard Product**: An element-wise multiplication of two matrices of the same size, resulting in a matrix of the same dimensions.
   - **Dot Product**: Involves multiplying corresponding elements of vectors or matrices and summing the results; for matrices, it refers to matrix multiplication.

2. **Output Dimensions**:
   - **Hadamard Product**: Output has the same dimensions as the input matrices.
   - **Dot Product**: The output matrix dimensions depend on the inner dimensions of the inputs (e.g., multiplying an $m \times n$ matrix by an $n \times p$ matrix results in an $m \times p$ matrix).

3. **Applications**:
   - **Hadamard Product**: Used in element-wise operations in deep learning and image processing.
   - **Dot Product**: Used in vector projections, transformations, and solving linear systems.

Example:  
For matrices $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$:

- **Hadamard Product**:

$$
  \mathbf{A} \circ \mathbf{B} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
$$

- **Dot Product** (Matrix Multiplication):  

$$
  \mathbf{A} \cdot \mathbf{B} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

</details>

--- 

Q. Write the properties of matrix product operations?

<details><summary><b>Answer</b></summary>

- Matrix multiplication is distributive 

$$
\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{A}\mathbf{B} +  \mathbf{A}\mathbf{C}
$$

- Matrix multiplication is associative

$$
\mathbf{A}(\mathbf{B}\mathbf{C}) = (\mathbf{A}\mathbf{B})\mathbf{C}
$$

- Matrix multiplication is not commutative

$$
\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}
$$

</details>

---

Q. Is dot product between two vectors are commutative?

<details><summary><b>Answer</b></summary>

Yes

$$x^{T}y = y^{T}x$$

</details>

---

Q. What is an identity matrix?

<details><summary><b>Answer</b></summary>

An identity matrix is a matrix that does not change any vector when we multiply that vector by that matrix.

$$
\forall \mathbf{x} \in \mathbb{R}^n, \; I_n \mathbf{x} = \mathbf{x}
$$

The structure of identity matrix is simple, all the entries along with main diagonal is $1$, while all the other entries are zero.

</details>

---

Q. Define matrix inverse?

<details><summary><b>Answer</b></summary>

The matrix inverse of $\mathbf{A}$ is denoted as $\mathbf{A^{-1}}$, and is defined as the matrix such that

$$
\mathbf{A^{-1}}\mathbf{A} = \mathbf{I_n}
$$

</details>

---

Q. How to check if a matrix is a singular matrix?

<details><summary><b>Answer</b></summary>

Key characteristics of a singular matrix:

-  $\mathbf{A}$ should be a square matrix
- **Determinant**: $\det(\mathbf{A}) = 0$.
- **Linear Dependence**: At least one row or column is redundant.
- **Non-Invertibility**: The matrix cannot be inverted, meaning there is no matrix $\mathbf{A}^{-1}$ such that $\mathbf{A} \mathbf{A}^{-1} = I$, where $I$ is the identity matrix.

</details>

---

Q. Under what conditions inverse of a matrix exists?

<details><summary><b>Answer</b></summary>

The inverse of a matrix $\mathbf{A}$ exists if:

- **$\det(\mathbf{A}) \neq 0$**: The determinant is non-zero.
- **Square Matrix**: $\mathbf{A}$ is $n \times n$.
- **Full Rank**: All rows/columns are linearly independent.
- **No Zero Eigenvalues**: No eigenvalues are zero.

</details>

---

Q. What is rank of matrix and how to determine it?

<details><summary><b>Answer</b></summary>

The rank of a matrix is defined as the maximum number of linearly independent rows or columns in the matrix. It represents the dimension of the row space or column space of the matrix.

We can determine rank via performing row operations to transform the matrix into row echelon form (REF). The rank is the number of non-zero rows in this form.

</details>

---

Q. What is a full rank matrix??

<details><summary><b>Answer</b></summary>

If the rank is equal to the smallest dimension of the matrix (i.e., the number of rows or columns), the matrix is said to have full rank.

</details>

---

Q. How to check if matrix is full rank?

<details><summary><b>Answer</b></summary>

For square matrices, compute the determinant. If that is non-zero, the matrix is of full rank. 

If the matrix $\mathbf{A}$ is $n$ by $m$, assume that $m≤n$ and compute all determinants of $m$ by $m$ sub-matrices. If one of them is non-zero, the matrix has full rank.

</details>

---


Q. Why do we say that matrices are linear transformations?

<details><summary><b>Answer</b></summary>

Matrices are considered linear transformations because they map vectors from one space to another while preserving the operations of vector addition and scalar multiplication, which are the core properties of linearity.

**Here is the proof for the same:**

Every matrix transformation is a linear transformation

Suppose that $\mathbf{T}$ is a matrix transformation such that $𝑇(𝑥⃗)=𝐴𝑥⃗$ for some matrix $𝐴$ and that the vectors $𝑢⃗$ and $𝑣⃗$ are in the domain. Then for arbitrary scalars $c$ and $d$:

$$
T(c \mathbf{u} + d \mathbf{v}) = A(c \mathbf{u} + d \mathbf{v})
$$

$$
= c A \mathbf{u} + d A \mathbf{v}
$$

$$
= c T
$$

$$
\text{As } T(c \mathbf{u} + d \mathbf{v}) = c T(\mathbf{u}) + d T(\mathbf{v}), \text{ the transformation } T \text{ must be linear.}
$$

</details>

---


Q. Do all matrices have an inverse? Is the inverse of a matrix always unique?

<details><summary><b>Answer</b></summary>

Not all matrices have an inverse; a matrix must be square and non-singular to have its inverse.

If a matrix $\mathbf{A}$ has an inverse, that inverse is always **unique**. If $\mathbf{A}^{-1}$ is an inverse of $\mathbf{A}$, then no other matrix can serve as the inverse. This is because if there were two different inverses, say $\mathbf{B}$ and $\mathbf{C}$, then:

$$
     \mathbf{A} \mathbf{B} = \mathbf{I} \quad \text{and} \quad \mathbf{A} \mathbf{C} = \mathbf{I}
$$

Multiplying both sides of the first equation by $\mathbf{C}$ yields:

$$
     \mathbf{B} = \mathbf{C}
$$

Thus, the inverse is unique.

</details>

---

Q. What does norm of a vector represents?

<details><summary><b>Answer</b></summary>

The norm of a vector $x$ measures the distance from the origin to the point $x$. 

</details>

---

Q. Explain Euclidean norm?

<details><summary><b>Answer</b></summary>

It is the $L^2$ norm, with $p=2$, which is simply the euclidean distance from the origin to the $x$. It is denoted by $||x||_2$ or just $||x||$.

It is also a common to measure the size of a vector using the squared $L^2$ norm, which is equal to $x^{T}x$

$$
||x|| = (\sum_{i} |x_i|^{2})^{\frac{1}{2}}
$$

</details>

---

Q. When we should use $L^1$ norm instead of $L^2$ norm?

<details><summary><b>Answer</b></summary>


$L^1$ norm is commonly used in machine learning when the difference between zero and non-zero elements is important. 

</details>

---

Q. What does max norm and unit norm depicts?

<details><summary><b>Answer</b></summary>

*Max Norm*

The max norm simplifies to the absolute value of the element with the largest magnitude in the vector.

It is denoted by $L^{\infty}$:

$$
\| \mathbf{x} \|_{\infty} = \max_i |x_i|
$$

*Unit Norm*

A vector with unit norm

$$
||x||_{2} = 1
$$

</details>

---

Q. How to measure size of a matrix?

<details><summary><b>Answer</b></summary>

We can use Frobenius norm, which is like $L^2$ norm of a vector.

$$
\| \mathbf{A} \|_{F} = \sqrt{\sum_{i,j} A_{i,j}^2}
$$

</details>

---

Q. Can you write the dot product of two vectors in terms of their norms?

<details><summary><b>Answer</b></summary>

Yes, the dot product of two vectors $\mathbf{x}$ and $\mathbf{y}$ can be expressed in terms of their norms. If $\mathbf{x}$ and $\mathbf{y}$ are vectors, their dot product $\mathbf{x} \cdot \mathbf{y}$ is given by:

$$
\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T \mathbf{y}
$$

This can also be expressed as:

$$
\mathbf{x}^T \mathbf{y} = \|\mathbf{x}\|_2 \|\mathbf{y}\|_2 \cos{\theta}
$$

where $\|\mathbf{x}\|_2$ and $\|\mathbf{y}\|_2$ are the Euclidean norms (or magnitudes) of $\mathbf{x}$ and $\mathbf{y}$, and $\theta$ is the angle between the two vectors.

</details>

---

Q. Define following type of matrices:

- Diagonal Matrix
- Symmetric Matrix
- Orthogonal Matrix

<details><summary><b>Answer</b></summary>

*Diagonal Matrix*

A matrix $\mathbf{D}$ is a diagonal if and only if $\mathbf{D}_{i, j} = 0$ for all $i \neq j$. 

*Symmetric Matrix*

A symmetric matrix is any matrix that is equal to its own transpose.

$$
\mathbf{A} = \mathbf{A^T}
$$

*Orthogonal Matrix*

An orthogonal matrix is a square matrix whose rows are mutually orthonormal and whose columns are also mutually orthonormal.

$$
\mathbf{A^T}\mathbf{A} = \mathbf{A}\mathbf{A^T} = \mathbf{I}
$$

Which implies that,

$$
\mathbf{A^{-1}} = \mathbf{A^T}
$$

</details>

---

Q. What is Eigen decomposition?

<details><summary><b>Answer</b></summary>

Eigen decomposition is a matrix factorization method where a matrix is decomposed into its eigenvalues and corresponding eigenvectors.

</details>

---

Q. Define eigenvector and eigenvalues?

<details><summary><b>Answer</b></summary>

An eigenvector of a square matrix $\mathbf{A}$ is a nonzero vector $\mathbf{v}$ that remains invariant in direction when transformed by $\mathbf{A}$—only its magnitude changes. This can be mathematically expressed as:

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
$$

In this equation, $\lambda$ is the eigenvalue associated with the eigenvector $\mathbf{v}$. The eigenvalue represents the scale factor by which the eigenvector is stretched or compressed. Thus, eigenvectors are the directions that remain unchanged by the transformation, and eigenvalues measure the extent of this scaling.

</details>

---

Q. Express the eigen decomposition of square matrix $\mathbf{A}$ ?

<details><summary><b>Answer</b></summary>

Suppose the matrix $\mathbf{A}$ has $n$ linearly independent eigenvectors ${v^{(1)},..,v^{(n)}}$ with corresponding eigenvalues ${\lambda_1,..,\lambda_n}$

Lets define a matrix $\mathbf{V}$ and a vector $\mathbf{\lambda}$

$$
\mathbf{V} = [v^{(1)},..,v^{(n)}]
$$

$$
\mathbf{\lambda} = [\lambda_1,..,\lambda_n]^T
$$

Eigen decomposition of $\mathbf{A}$ can be expressed as:

$$
\mathbf{A} = \mathbf{V}\text{diag}(\mathbf{\lambda})\mathbf{V}^{-1}
$$

</details>

---

Q. What is the significant to eigen-decomposition?

<details><summary><b>Answer</b></summary>

Eigendecomposition of a matrix tells use many useful facts about the matrix.

- The matrix is singular if and only if any of the eigenvalues are zero.
- The determinant of the matrix $\mathbf{A}$ equals the product of its eigenvalues.
- The trace of the matrix $\mathbf{A}$ equals the summation of its eigenvalues.
- If the eigenvalues of $\mathbf{A}$ are $\lambda_{i}$ and $\mathbf{A}$ is non-singular, then the eigenvalues of $\mathbf{A^{-1}}$ are simply $\lambda{1}{\lambda_{i}}$.
- The eigenvectors of $\mathbf{A^{-1}}$ are the same as eigenvectors of $\mathbf{A}$.

</details>

---

Q. Do non-square matrices have eigenvalues?

<details><summary><b>Answer</b></summary>

Eigenvalues and eigenvectors of a matrix $\mathbf{A}$ help identify subspaces that remain invariant under the linear transformation represented by $\mathbf{A}$. However, if $\mathbf{A}$ is non-square, meaning $\mathbf{A} : \mathbb{R}^m \to \mathbb{R}^n$ with $m \neq n$, the equation $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ is not applicable because $\mathbf{A}\mathbf{v}$ will not necessarily lie in $\mathbb{R}^m$.

</details>

---

Q. What is the benefit of using Singular Value Decomposition(SVD) over eigenvalue decomposition?

<details><summary><b>Answer</b></summary>

Every real matrix has a singular value decomposition, but same is not true for eigenvalue decomposition.

</details>

---

Q. What is Singular Value Decomposition (SVD)?

<details><summary><b>Answer</b></summary>

Singular Value Decomposition (SVD) is a technique used to factorize a matrix into its constituent singular vectors and singular values.

For a given real matrix $\mathbf{A}$ of dimensions $m \times n$, the decomposition is expressed as:

$$
\mathbf{A} = \mathbf{U} \mathbf{D} \mathbf{V}^T
$$

where:

- $\mathbf{U}$ is an $m \times m$ orthogonal matrix whose columns are the left singular vectors.
- $\mathbf{D}$ is an $m \times n$ diagonal matrix containing the singular values on its diagonal. It may not be square.
- $\mathbf{V}$ is an $n \times n$ orthogonal matrix whose columns are the right singular vectors.

SVD provides a way to decompose a matrix into components that reveal important properties and facilitate various applications in data analysis, signal processing, and more.

</details>

---

Q. What is trace of a matrix? 

<details><summary><b>Answer</b></summary>

The trace operator gives the sum of all the diagonal entries of a matrix.

$$
Tr(A) = \sum_i{A_{i, i}}
$$


</details>

---

Q. Write the main properties of trace operator? 

<details><summary><b>Answer</b></summary>

- Frobenius norm of a matrix:

$$
||A||_{F} = \sqrt{Tr(AA^T)}
$$

- Trace operator is invariant to the transpose operator

$$
Tr(A) = Tr(A^T)
$$

- Invariance to cyclic permutations

$$
Tr(AB) = Tr(BA)
$$

</details>

---

Q. What does the determinant of a matrix represent?

<details><summary><b>Answer</b></summary>

The determinant of a square matrix denoted by $\text{det}(\mathbf{A})$, is a function that maps matrices to real scalers. The determinant is equal to the product of all the eigenvalues of the matrix. 

</details>

---

Q. What does the absolute value of the determinant depicts?

<details><summary><b>Answer</b></summary>

The absolute value of the determinant of a matrix provides a measure of the scale factor by which the matrix expands or contracts space.

-  $|\text{det}(\mathbf{A})| = 0$ : Space is contracted completely and transformation will result in loss of dimensionality.
-  $|\text{det}(\mathbf{A})| = 1$ : Volume remains unchanged 
- $|\text{det}(\mathbf{A})| > 1$ : Volume gets enlarged
- $|\text{det}(\mathbf{A})| < 1$ : Volume gets compressed

</details>

---


Q. What happens to the determinant of a matrix if we multiply one of its rows by a scalar  $t×R$ ?

<details><summary><b>Answer</b></summary>

If you multiply one of the rows of a matrix $\mathbf{A}$ by a scalar $t$, the determinant of the matrix is scaled by the same factor $t$. Specifically:

- Let $\mathbf{A}$ be an $n \times n$ matrix.
- If you multiply one row of $\mathbf{A}$ by a scalar $t$, the new matrix $\mathbf{A'}$ will have a determinant given by:

$$
\text{det}(\mathbf{A}') = t \cdot \text{det}(\mathbf{A})
$$

This property reflects that the determinant is a multilinear function of the rows of the matrix. Hence, multiplying a row by a scalar scales the determinant by that scalar.

</details>

---


Q. A $4×4$  matrix has four eigenvalues $3, 3, 2, −1$. What can we say about the trace and the determinant of this matrix?

<details><summary><b>Answer</b></summary>

1. **Trace**: The trace of a matrix is the sum of its eigenvalues. Therefore, for this matrix:

$$
   \text{Trace} = 3 + 3 + 2 + (-1) = 7
$$

2. **Determinant**: The determinant of a matrix is the product of its eigenvalues. Thus, for this matrix:

$$
   \text{Determinant} = 3 \times 3 \times 2 \times (-1) = -18
$$

</details>

---


Q. Given the following matrix:

$$
\begin{bmatrix}
1 & 4 & -2\\
-1 & 3 & 2 \\
3 & 5 & -6
\end{bmatrix}
$$

Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?

<details><summary><b>Answer</b></summary>

We can write the above matrix into its row echelon form:

- Add the first row to the second row to make the entry in the second row, first column zero:

$$
R2 \rightarrow R2 + R1
$$

$$
\begin{bmatrix}
1 & 4 & -2 \\
0 & 7 & 0 \\
3 & 5 & -6
\end{bmatrix}
$$

- Subtract $3$ times the first row from the third row to make the entry in the third row, first column zero:

$$
R3 \rightarrow R3 - 3 \times R1
$$

$$
\begin{bmatrix}
1 & 4 & -2 \\
0 & 7 & 0 \\
0 & -7 & 0
\end{bmatrix}
$$

- Divide the second row by $7$ to normalize the leading coefficient:

$$
R2 \rightarrow \frac{1}{7} \times R2
$$

$$
\begin{bmatrix}
1 & 4 & -2 \\
0 & 1 & 0 \\
0 & -7 & 0
\end{bmatrix}
$$

- Add $7$ times the second row to the third row to make the entry in the third row, second column zero:

$$
     R3 \rightarrow R3 + 7 \times R2
$$

$$
     \begin{bmatrix}
     1 & 4 & -2 \\
     0 & 1 & 0 \\
     0 & 0 & 0
     \end{bmatrix}
$$

- Subtract $4$ times the second row from the first row to make the entry in the first row, second column zero:

$$
     R1 \rightarrow R1 - 4 \times R2
$$

$$
     \begin{bmatrix}
     1 & 0 & -2 \\
     0 & 1 & 0 \\
     0 & 0 & 0
     \end{bmatrix}
$$

The presence of a row of zeros in the row echelon form indicates that the matrix is singular, meaning its determinant is zero.

</details>

---


Q. What’s the difference between the covariance matrix $A^TA$  and the Gram matrix $AA^T$?

<details><summary><b>Answer</b></summary>

- **Dimensions**:
  -  $A^T A$ is $n \times n$ (columns of $A$).
  - $A A^T$ is $m \times m$ (rows of $A$).

- **Focus**:
  - $A^T A$ focuses on the relationships between columns.
  - $A A^T$ focuses on the relationships between rows.

- **Applications**:
  - $A^T A$ is used to understand the variance and covariance of data columns.
  - $A A^T$ is used to understand the similarity and inner product of data rows.


</details>

---

Q. Given $A∈R^{n×m}$  and $b∈R^n$ 
1. Find $x$ such that: $Ax=b$.
1. When does this have a unique solution?
1. Why is it when $A$ has more columns than rows, $Ax=b$ has multiple solutions?
1. Given a matrix $A$ with no inverse. How would you solve the equation  $Ax=b$? What is the pseudo inverse and how to calculate it?

<details><summary><b>Answer</b></summary>

1. To find $x$ such that $A x = b$ where $A \in \mathbb{R}^{n \times m}$ and $b \in \mathbb{R}^n$, you generally need to solve a linear system. The method used depends on the properties of $A$:

- **If $A$ is square (i.e., $n = m$) and invertible**, you can find $x$ directly using:

$$
  x = A^{-1} b
$$

- **If $A$ is not square or not invertible**, we may use other methods such as:
  - **Gaussian Elimination**: Useful for finding solutions and performing row reductions.
  - **Least Squares Solution**: If $A$ has more rows than columns ($n > m$) and does not have an exact solution, find the least squares solution.
  - **Pseudo-Inverse**: When $A$ is not invertible or not square, the Moore-Penrose pseudo-inverse is used.

2. The linear system $A x = b$ has a unique solution if:

- **The matrix $A$ is square ($n = m$) and invertible**, meaning $\text{det}(A) \neq 0$. In this case, the matrix $A$ has full rank, and the solution is given by $x = A^{-1} b$.

- **For non-square matrices**, a unique solution occurs when the system is consistent and has a unique solution if:
  - The matrix $A$ has full column rank (if $m \leq n$) and $b$ is in the column space of $A$.
  - In the case of an over-determined system (more rows than columns), $A$ should have full column rank.

3. The system $A x = b$ has infinitely many solutions because there are free variables associated with the null space. This leads to a solution space that forms an affine subspace in \(\mathbb{R}^m\), where each solution can be expressed as $x = x_0 + \text{null}(A)$, where $x_0$ is a particular solution and \(\text{null}(A)\) represents the null space of $A$.

4. 

</details>


---


Q. Given a very large symmetric matrix $A$ that doesn’t fit in memory, say $A∈R^{1M×1M}$  and a function $f$ that can quickly compute $f(x)=Ax$ for $x∈R1M$. Find the unit vector $x$ so that $x^TAx$  is minimal.

<details><summary><b>Answer</b></summary>



</details>

---