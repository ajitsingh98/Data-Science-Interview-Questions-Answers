# Data Science Interview Questions And Answers

## Maths Questions

## Contents
- [Vector](#vector)
- [Matrices](#matrices)

## Vector

1. Dot product
    1. What’s the geometric interpretation of the dot product of two vectors?
    1. Given a vector $u$ , find vector $v$  of unit length such that the dot product of $u$  and $v$  is maximum.

<details><summary><b>Answer</b></summary>

1. Let $\vec{A}= ⟨a_1....a_k⟩$ and $\vec{B}= ⟨b_1....b_k⟩$  be k-dimensional vectors. The dot product of $\vec{A}$ and $\vec{B}$ , denoted $\vec{A} \cdot \vec{B}$ is a number, defined as follows

$$\vec{A} \cdot \vec{B} = a_1b_1+a_2b_2+....+a_kb_k$$

The dot product has the following geometric interpretation: Let $\alpha$ be the angle between $\vec{A}$ and $\vec{B}$. Then 

$$\vec{A} \cdot \vec{B} = |\vec{A}| \cdot |\vec{B}| \cdot \cos(\alpha)$$

2. To find a vector \( v \) of unit length such that the dot product of \( u \) and \( v \) is maximum, we want to maximize the expression for the dot product $\( u \cdot v \)$.
According to the formula for the dot product:

$$u \cdot v = |u| |v| \cos(\theta)$$

where:
- \( |u| \) is the magnitude of \( u \),
- \( |v| \) is the magnitude of \( v \) (which is 1 in this case because \( v \) is a unit vector),
- $\( \theta \)$ is the angle between \( u \) and \( v \).

To maximize $ u \cdot v $, the $\cos(\theta)$ part must be maximized. The cosine of an angle reaches its maximum value of 1 when the angle $\theta$ is 0 degrees, meaning that the vectors \( u \) and \( v \) must be pointing in the same direction.

Thus, vector \( v \) should be a unit vector in the direction of \( u \). This can be achieved by normalizing \( u \). The normalization of \( u \) is done by dividing \( u \) by its magnitude. If \( u \) is represented as \( u = (u_1, u_2, \ldots, u_n) \) and its magnitude \( |u| \) is given by:

\[ |u| = \sqrt{u_1^2 + u_2^2 + \ldots + u_n^2} \]

Then, the unit vector \( v \) in the direction of \( u \) is:

\[ v = \frac{u}{|u|} = \left(\frac{u_1}{|u|}, \frac{u_2}{|u|}, \ldots, \frac{u_n}{|u|}\right) \]

This vector \( v \) will have a unit length and the dot product \( u \cdot v \) will be maximum, equal to the magnitude of \( u \) (since \( u \cdot v = |u| \cdot 1 \cdot \cos(0^\circ) = |u| \)).

</details>


1. Outer product
    1. Given two vectors $a=[3,2,1]$  and $b=[−1,0,1]$. Calculate the outer product $a^Tb$ ?
    1. Give an example of how the outer product can be useful in ML.

<details><summary><b>Answer</b></summary>

1. resultant product will be a $3X3$ matrix, which can be given as follows:

$$
\begin{bmatrix}
    -3 & 0 & 3\\
    -2 & 0 & 2 \\
    -1 & 0 & 1
\end{bmatrix}
$$

2. One useful application of the outer product in machine learning is in the computation of covariance matrices, where the outer product is used to calculate the covariance of different feature vectors. For instance, the covariance matrix can be estimated as the average outer product of the centered data vectors (i.e., data vectors from which the mean has been subtracted). This is crucial for algorithms that rely on data distribution, such as Principal Component Analysis (PCA) and many types of clustering algorithms.


</details>

1. What does it mean for two vectors to be linearly independent?

<details><summary><b>Answer</b></summary>

Two vectors are said to be **linearly independent** if no vector in the set can be written as a linear combination of the others. In simpler terms, neither of the vectors can be expressed as a scalar multiple or a combination involving scalar multiples of the other vector.

For two vectors $\vec{a}$ and $\vec{b}$, they are linearly independent if the only solution to the equation

$$c_1 \vec{a} + c_2 \vec{b} = \vec{0}$$

is $c_1 = 0$ and $c_2 = 0$, where $c_1$ and $c_2$ are scalars and $\vec{0}$ is the zero vector. 

</details>

1. Given two sets of vectors $A=a_1,a_2,a_3,...,a_n$  and $B=b_1,b_2,b_3,...,b_m$. How do you check that they share the same basis?

<details><summary><b>Answer</b></summary>

</details>
1. Given $n$  vectors, each of $d$  dimensions. What is the dimension of their span?
1. Norms and metrics
    1. What's the norm? What is  $L_0,L_1,L_2,L_{norm}$?
    1. How do norms and metrics differ? Given a norm, make a metric. Given a metric, can we make a norm?
<details><summary><b>Answer</b></summary>

1. A **norm** on a vector space is a function that assigns a non-negative length or size to vectors, except for the zero vector, which is assigned a length of zero. Norms are denoted by \(\|\cdot\|\) and must satisfy the following properties for any vectors \(x, y\) and any scalar \(a\):

1. **Non-negativity**: \(\|x\| \geq 0\) and \(\|x\| = 0\) if and only if \(x = 0\).
2. **Scalar multiplication**: \(\|a \cdot x\| = |a| \cdot \|x\|\).
3. **Triangle inequality**: \(\|x + y\| \leq \|x\| + \|y\|\).

Different types of norms can be defined on vector spaces:

- **\(L_0\) norm** (not a true norm): It counts the number of non-zero entries in a vector. It does not satisfy the triangle inequality or the homogeneity property (scalar multiplication), which is why it's technically not a norm.
- **\(L_1\) norm**: It is defined as \(\|x\|_1 = \sum |x_i|\), summing the absolute values of the entries of the vector.
- **\(L_2\) norm** (Euclidean norm): It is defined as \(\|x\|_2 = \sqrt{\sum x_i^2}\), which corresponds to the usual geometric length of a vector.
- **\(L_p\) norm**: It generalizes the \(L_1\) and \(L_2\) norms and is defined as \(\|x\|_p = (\sum |x_i|^p)^{1/p}\) for \(1 \leq p < \infty\).

2.
- A **norm** provides a way to measure the length of vectors in vector spaces.
- A **metric** is a more general function that defines a distance between any two elements in a set, satisfying:
  1. **Non-negativity**: \(d(x, y) \geq 0\) and \(d(x, y) = 0\) if and only if \(x = y\).
  2. **Symmetry**: \(d(x, y) = d(y, x)\).
  3. **Triangle inequality**: \(d(x, z) \leq d(x, y) + d(y, z)\).

**Given a norm, make a metric.**

If you have a norm \(\|\cdot\|\) on a vector space, you can define a metric \(d\) by \(d(x, y) = \|x - y\|\). This metric satisfies all the metric properties, derived from the properties of the norm.

**Given a metric, can we make a norm?**

Not all metrics come from norms. To derive a norm from a metric \(d\), the metric must satisfy additional properties:
1. **Translation invariance**: \(d(x+z, y+z) = d(x, y)\) for all \(x, y, z\).
2. **Homogeneity**: \(d(\alpha x, \alpha y) = |\alpha| d(x, y)\) for all scalars \(\alpha\).

If a metric satisfies these conditions, it can be associated with a norm, where the norm \(\|x\|\) can be defined as \(d(x, 0)\). However, many metrics (like the discrete metric) do not satisfy these properties and thus cannot be associated with a norm.
</details>

## Matrices

1. Why do we say that matrices are linear transformations?

<details><summary><b>Answer</b></summary>



</details>

---


2. What’s the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?

<details><summary><b>Answer</b></summary>



</details>

---


3. What does the determinant of a matrix represent?

<details><summary><b>Answer</b></summary>



</details>

---


4. What happens to the determinant of a matrix if we multiply one of its rows by a scalar  $t×R$ ?

<details><summary><b>Answer</b></summary>



</details>

---


5. A $4×4$  matrix has four eigenvalues $3,3,2,−1$. What can we say about the trace and the determinant of this matrix?

<details><summary><b>Answer</b></summary>



</details>

---


6. Given the following matrix:

    $$
    \begin{bmatrix}
    1 & 4 & -2\\
    -1 & 3 & 2 \\
    3 & 5 & -6
    \end{bmatrix}
    $$

    Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?

<details><summary><b>Answer</b></summary>


</details>

---


7. What’s the difference between the covariance matrix $A^TA$  and the Gram matrix $AA^T$ ?

<details><summary><b>Answer</b></summary>



</details>

---


8. Given $A∈R^{n×m}$  and $b∈R^n$ 
    1. Find $x$ such that: $Ax=b$.
    1. When does this have a unique solution?
    1. Why is it when $A$ has more columns than rows, $Ax=b$ has multiple solutions?
    1. Given a matrix $A$ with no inverse. How would you solve the equation  $Ax=b$? What is the pseudo inverse and how to calculate it?

<details><summary><b>Answer</b></summary>



</details>

---


9. Derivative is the backbone of gradient descent.
    1. What does derivative represent?
    1. What’s the difference between derivative, gradient, and Jacobian?

<details><summary><b>Answer</b></summary>



</details>

---


10. Say we have the weights $w∈R^{d×m}$  and a mini-batch $x$  of $n$  elements, each element is of the shape $1×d$  so that $x∈R^{n×d}$. We have the output $y=f(x;w)=xw$. What’s the dimension of the Jacobian $\frac{δy}{δx}$?

<details><summary><b>Answer</b></summary>



</details>

---


11. Given a very large symmetric matrix $A$ that doesn’t fit in memory, say $A∈R^{1M×1M}$  and a function $f$ that can quickly compute $f(x)=Ax$ for $x∈R1M$. Find the unit vector $x$ so that $x^TAx$  is minimal.

<details><summary><b>Answer</b></summary>



</details>

---

