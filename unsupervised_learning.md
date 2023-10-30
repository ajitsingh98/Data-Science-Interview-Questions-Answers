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
2. Name some scenarios where we can use unsupervised learning algorithms?
 




## Association Mining

## Cluserting

5. k-means clustering.
    1. How would you choose the value of k?
    1. If the labels are known, how would you evaluate the performance of your k-means clustering algorithm?
    1. How would you do it if the labels aren’t known?
    1. Given the following dataset, can you predict how K-means clustering works on it? Explain.
    ![image](img/k_means.png)

7. k-means and GMM are both powerful clustering algorithms.
    1. Compare the two.
    1. When would you choose one over another?




## Dimensionality Reduction

1. Why do we need dimensionality reduction?
2. Eigendecomposition is a common factorization technique used for dimensionality reduction. Is the eigendecomposition of a matrix always unique?
3. Name some applications of eigenvalues and eigenvectors.
4. We want to do PCA on a dataset of multiple features in different ranges. For example, one is in the range $0-1$ and one is in the range $10 - 1000$. Will PCA work on this dataset?
5. Under what conditions can one apply eigendecomposition? What about SVD?
    1. What is the relationship between SVD and eigendecomposition?
    1. What’s the relationship between PCA and SVD?
6. How does $t-SNE$ (T-distributed Stochastic Neighbor Embedding) work? Why do we need it?
7. Is it good to use PCA as a feature selection method?
8. Is PCA a linear model or non-linear model?



## Recommendation Engines

9. Given this directed graph.
    ![image](assets/dag.png)
    1. Construct its adjacency matrix.
    1. How would this matrix change if the graph is now undirected?
    1. What can you say about the adjacency matrices of two isomorphic graphs?
10. Imagine we build a user-item collaborative filtering system to recommend to each user items similar to the items they’ve bought before.
    1. You can build either a user-item matrix or an item-item matrix. What are the pros and cons of each approach?
    1. How would you handle a new user who hasn’t made any purchases in the past?

