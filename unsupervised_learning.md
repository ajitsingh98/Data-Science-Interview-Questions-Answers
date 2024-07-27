# Data Science Interview Questions And Answers

Topics
---

- [Unsupervised Learning](#unsupervised-learning)

Contents
---
- [General Concepts](#general-concepts)
- [Association Mining](#association-mining)
- [Clustering](#clustering)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Recommendation Engines](#recommendation-engines)


## General Concepts

1. What is unsupervised learning?

<details><summary><b>Answer</b></summary>



</details>

---

2. Name some scenarios where we can use unsupervised learning algorithms?


<details><summary><b>Answer</b></summary>



</details>

---



## Association Mining

## Clustering

1. What is the role of the "K" in K-means?

<details><summary><b>Answer</b></summary>



</details>

---


2. What are the advantages of K-means clustering?

<details><summary><b>Answer</b></summary>



</details>

---


5. k-means clustering.
    1. How would you choose the value of k?
    1. If the labels are known, how would you evaluate the performance of your k-means clustering algorithm?
    1. How would you do it if the labels aren’t known?
    1. Given the following dataset, can you predict how K-means clustering works on it? Explain.
    ![image](img/k_means.png)

<details><summary><b>Answer</b></summary>



</details>

---


3. What are the limitations of K-means clustering?

<details><summary><b>Answer</b></summary>



</details>

---


4. How do you initialize the centroids in K-means?

<details><summary><b>Answer</b></summary>



</details>

---


5. What is the convergence criteria in K-means?

<details><summary><b>Answer</b></summary>



</details>

---


6. What are some applications of K-means clustering?

<details><summary><b>Answer</b></summary>



</details>

---


8. Can K-means handle categorical data?

<details><summary><b>Answer</b></summary>



</details>

---


10. How do you evaluate the quality of K-means clusters?

<details><summary><b>Answer</b></summary>



</details>

---


7. k-means and GMM are both powerful clustering algorithms.
    1. Compare the two.
    1. When would you choose one over another?


<details><summary><b>Answer</b></summary>



</details>

---

## Dimensionality Reduction

1. Why do we need dimensionality reduction?

<details><summary><b>Answer</b></summary>



</details>

---


2. Eigendecomposition is a common factorization technique used for dimensionality reduction. Is the eigendecomposition of a matrix always unique?

<details><summary><b>Answer</b></summary>



</details>

---


3. Name some applications of eigenvalues and eigenvectors.

<details><summary><b>Answer</b></summary>



</details>

---


4. We want to do PCA on a dataset of multiple features in different ranges. For example, one is in the range $0-1$ and one is in the range $10 - 1000$. Will PCA work on this dataset?

<details><summary><b>Answer</b></summary>



</details>

---


5. Under what conditions can one apply eigendecomposition? What about SVD?
    1. What is the relationship between SVD and eigendecomposition?
    1. What’s the relationship between PCA and SVD?

<details><summary><b>Answer</b></summary>



</details>

---


6. How does $t-SNE$ (T-distributed Stochastic Neighbor Embedding) work? Why do we need it?

<details><summary><b>Answer</b></summary>



</details>

---


7. Is it good to use PCA as a feature selection method?

<details><summary><b>Answer</b></summary>



</details>

---


8. Is PCA a linear model or non-linear model?

<details><summary><b>Answer</b></summary>



</details>

---


9. What is the importance of eigenvalues and eigenvectors in PCA?

<details><summary><b>Answer</b></summary>



</details>

---


10. How do you decide the number of principal components to retain in PCA?

<details><summary><b>Answer</b></summary>



</details>

---


11. What is the difference between PCA and Linear Discriminant Analysis (LDA)?

<details><summary><b>Answer</b></summary>



</details>

---


12. What are the limitations of PCA?

<details><summary><b>Answer</b></summary>



</details>

---


13. Can PCA be used for feature selection?

<details><summary><b>Answer</b></summary>



</details>

---


14. Explain the concept of whitening in PCA.

<details><summary><b>Answer</b></summary>



</details>

---


## Recommendation Engines

9. Given this directed graph.
    ![image](assets/dag.png)
    1. Construct its adjacency matrix.
    1. How would this matrix change if the graph is now undirected?
    1. What can you say about the adjacency matrices of two isomorphic graphs?

<details><summary><b>Answer</b></summary>



</details>

---


10. Imagine we build a user-item collaborative filtering system to recommend to each user items similar to the items they’ve bought before.
    1. You can build either a user-item matrix or an item-item matrix. What are the pros and cons of each approach?
    1. How would you handle a new user who hasn’t made any purchases in the past?

<details><summary><b>Answer</b></summary>



</details>

---


11. What are the key parameters in DBSCAN, and what do they represent?

<details><summary><b>Answer</b></summary>



</details>

---


12. What is the difference between core points, border points, and noise points in DBSCAN?

<details><summary><b>Answer</b></summary>



</details>

---


13. How does DBSCAN handle clusters of different shapes?

<details><summary><b>Answer</b></summary>



</details>

---


14. What are the advantages of using DBSCAN over other clustering algorithms, such as K-means?

<details><summary><b>Answer</b></summary>



</details>

---

