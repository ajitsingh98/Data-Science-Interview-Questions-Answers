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

It is a set of statistical tools intended for the setting in which we have only a set of features $X1, X2, . . . , Xp$ measured on $n$ observations. We are not interested in prediction, because we do not have an associated response variable $Y$. Rather, the goal is to discover interesting patterns in the measurements on $X1, X2, . . . , X_p$

</details>

---

2. Name some scenarios where we can use unsupervised learning algorithms?


<details><summary><b>Answer</b></summary>

We can use unsupervised learning in following use-cases:

- Anomaly detections
- Dimensionality reduction
- Clustering

</details>

---



## Association Mining

## Clustering

1. What do you mean by clustering?

<details><summary><b>Answer</b></summary>

Clustering refers to a very broad set of techniques for finding subgroups, or clusters, in a data set. When we cluster the observations of a data set, we seek to partition them into distinct groups so that the observations within each group are quite similar to each other, while observations in different groups are quite different from each other.

</details>

---

1. What is the main difference between clustering and PCA?

<details><summary><b>Answer</b></summary>

Although both clustering and PCA seeks to simplify the data via a small number of summaries, but their mechanism is different:

- PCA looks to find a low dimensional representation of the observations that explain a good fraction of the variance.
- Clustering looks to find homogenous subgroups among the observations.

</details>

---

1. What is the role of the $K$ in K-means?

<details><summary><b>Answer</b></summary>


$K$ is a hyperparameter that decides number of clusters we want from the model to find in the underlying data.

</details>

---

2. What are the advantages of K-means clustering?

<details><summary><b>Answer</b></summary>

K-means clustering offers several key advantages:

1. **Simplicity**: Easy to understand and implement.
2. **Efficiency**: Fast and computationally efficient, even for large datasets.
3. **Scalability**: Works well with large datasets.
4. **Quick Convergence**: Typically converges quickly in a few iterations.
5. **Interpretability**: Results are easy to interpret, with clear centroids representing clusters.

</details>

---


5. k-means clustering.
    1. How would you choose the value of k?
    1. If the labels are known, how would you evaluate the performance of your k-means clustering algorithm?
    1. How would you do it if the labels aren’t known?
    1. Given the following dataset, can you predict how K-means clustering works on it? Explain.
    ![image](img/k_means.png)

<details><summary><b>Answer</b></summary>

1. We can use following methods to get the optimal value $K$

    - **Elbow Method**:
        - Plot the sum of squared distances from each point to its assigned cluster centroid (known as the Within-Cluster Sum of Squares or WCSS) against the number of clusters $K$.
        - As $K$ increases, WCSS decreases. The idea is to choose the value of $K$ at the "elbow" point where the rate of decrease sharply slows down. This indicates diminishing returns and is a good trade-off between reducing WCSS and keeping $K$ manageable.

    - **Silhouette Score**:
        - Calculate the silhouette score for different values of $K$. The silhouette score measures how similar a data point is to its own cluster compared to other clusters.
        - The value of $K$ that maximizes the average silhouette score across all data points is typically considered optimal.
        
    - **Cross-Validation**:
        - Split the data into training and validation sets, perform K-means clustering on the training set for different values of $K$, and evaluate the clustering performance on the validation set.
        - Choose the $K$ that gives the best performance on the validation set.

2. 



</details>

---


3. What are the limitations of K-means clustering?

<details><summary><b>Answer</b></summary>

K-means clustering suffers from following limitations:

1. **Assumes Spherical Clusters**: K-means works well for data with spherical-like cluster shapes but performs poorly when clusters have complex geometric shapes.

2. **Difficulty with Distant Data Points**: K-means doesn't allow distant data points to belong to the same cluster, even if they naturally should, leading to incorrect clustering when data points are spread across different regions.

3. **Bias Toward Larger Clusters**: K-means tends to favor larger clusters when minimizing within-cluster variation, which can result in smaller clusters being inaccurately represented or misclassified.

4. **Poor Performance with Non-Linear Boundaries**: K-means struggles with data that has non-linear boundaries or complicated shapes, such as moons or concentric circles, failing to cluster such data correctly.

5. **K is a hyperparameter**: It doesn’t learn the number of clusters from the data and requires it to be pre-defined. 

6. **Feature Scaling**: The features should be scaled to have mean zero and standard deviation one.

</details>

---


4. How do you initialize the centroids in K-means?

<details><summary><b>Answer</b></summary>

Initializing centroid plays crucial role in model's performance and can impact quality of final clusters.

- Random Initialization

    - Select $k$ data points randomly from the dataset as initial centroid
    - It may lead to suboptimal clustering or slow convergence 

- K-means++ Initialization

    - This is improvement over random initialization, here we select first centroid randomly and then chooses subsequent centroids probabilistically based on their distance from the nearest existing centroid. 
    - Points further away from existing centroids are more likely to be selected.
    - Computationally expensive  

</details>

---


5. What is the convergence criteria in K-means?

<details><summary><b>Answer</b></summary>

In K-means clustering, convergence is typically achieved when one of the following criteria is met:

1. **Centroid Stabilization**: The centroids stop changing (or change very little) between iterations.
2. **No Change in Cluster Assignments**: No data points change clusters between iterations.
3. **Maximum Iterations**: The algorithm reaches a pre-set maximum number of iterations.
4. **Minimal WCSS Improvement**: The decrease in within-cluster sum of squares (WCSS) falls below a set threshold.

These criteria ensure the algorithm stops when clusters are stable and further iterations provide minimal benefit.

</details>

---

6. What is Silhouette score, and How do we calculate it?

<details><summary><b>Answer</b></summary>

The Silhouette score is a measure used to evaluate the quality of clustering by quantifying how similar each data point is to its own cluster compared to other clusters.

We can calculate it using following expressions:

1. Find average inter-cluster distance:

    - For a given data point $i$, compute the average distance of all other points within the same cluster.
    $$a(i) = \frac{1}{|C_i| - 1}\sum_{j\epsilon C,j!=i}d(i, j)$$
    where, $C_i$ is the cluster to which point $i$ belongs, and $d(i, j)$ is the distance between $i$ and $j$.

2. Find average nearest cluster distance 
    - For the same data point $i$, compute the average distance to all points in the nearest neighboring cluster.

    $$b(i) = \min_{C \ne C_i}(\frac{1}{|C|}\sum_{j \epsilon C}d(i, j))$$

    where, $C$ is a cluster different from $C_i$ and $d(i, j)$ is the distance between points $i$ and $j$ in the nearest cluster.

3. Compute Silhouette score for each data point

    - Calculate the Silhouette score for each data point $i$ using values of $a(i)$ and $b(i)$
    $$s(i) = \frac{b(i) - a(i)}{max(a(i),b(i))}$$
    - Note the score ranges from -1 to 1
        
        - $-1$ means data point is misclassified 
        - $0$ means data point is on or very close to the decision boundary between two neighboring clusters
        - $1$ means data point is well clustered   

4. Calculate average Silhouette score

    - Calculate average silhouette score across all data points to get measure of overall cluster quality 
    $$ \text{Average Silhouette Score} = \frac{1}{n} \sum_{i=1}^{n}s(i)$$
    where $n$ is the number of data points.


</details>

---

6. What are some applications of K-means clustering?

<details><summary><b>Answer</b></summary>

Here are some application of K-Means clustering:

- Customer segmentation
- Fraud detection
- Predicting account attrition
- Image compression


</details>

---


8. Can K-means handle categorical data?

<details><summary><b>Answer</b></summary>

The standard K-means algorithm is not directly applicable to categorical data because it relies on Euclidean distance, which is not meaningful for discrete, non-numeric categories. Unlike continuous data, categorical data lacks a natural origin and does not support Euclidean distance measurements. As a result, traditional K-means, which computes distances based on mean values, is not suitable for such data.

We can use a variation of K-means called K-modes. It uses *Hamming distance* as a distance metric and update centroids by mode instead of mean, making it more subtle for discrete attributes.




</details>

---


10. How do you evaluate the quality of K-means clusters?

<details><summary><b>Answer</b></summary>

We can use following metrics to evaluate the K-means clusters:

- Elbow Method
- Silhouette Score

</details>

---


7. k-means and GMM are both powerful clustering algorithms.
    1. Compare the two.
    1. When would you choose one over another?


<details><summary><b>Answer</b></summary>



</details>

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

## Dimensionality Reduction

1. Why do we need dimensionality reduction?

<details><summary><b>Answer</b></summary>

We may need dimensionality reduction for following reasons:

- **Mitigate Curse of Dimensionality**: Improve model performance by simplifying the data space.
- **Enhance Visualization**: Facilitate the visualization of high-dimensional data in 2D or 3D.
- **Improve Model Performance**: Reduce overfitting and improve generalization by removing noisy or irrelevant features.
- **Simplify Models**: Make models more interpretable by focusing on key features.


</details>

---


2. Eigen decomposition is a common factorization technique used for dimensionality reduction. Is the eigen decomposition of a matrix always unique?

<details><summary><b>Answer</b></summary>




</details>

---


3. Name some applications of eigenvalues and eigenvectors.

<details><summary><b>Answer</b></summary>



</details>

---


4. We want to do PCA on a dataset of multiple features in different ranges. For example, one is in the range $0-1$ and one is in the range $10 - 1000$. Will PCA work on this dataset?

<details><summary><b>Answer</b></summary>

Yeah, PCA will work in this scenario but it may not provide optimal principal components.

It is important to scale the data points such that they are centered to have mean zero and standard deviation of $1$. Because PCA is basically variance maximizing exercise. It projects the original data onto directions which maximize the variance. 

*Impact of scaling on PCA*

<table align='center'>
<tr>
<td align="center">
    <img src="img/scaling_impact_pca.png" alt= "Scaling impact on PCA" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> Impact of Scaling the features on PCA </td>
</tr>
</table>

</details>

---


5. Under what conditions can one apply eigen decomposition? What about SVD?
    1. What is the relationship between SVD and eigen decomposition?
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

No, PCA is not a good way to do feature selection as it does not consider response while calculating the principal components. A feature having less variance does not mean it is providing less or no information.

</details>

---


8. Is PCA a linear model or non-linear model?

<details><summary><b>Answer</b></summary>

PCA is a linear model. Because, PCA works by finding new axes (principal components) that are linear combinations of the original features. These components are created by finding the directions in the data that maximize variance while ensuring that these directions are orthogonal (uncorrelated) to each other.

</details>

---


9. What is the importance of eigenvalues and eigenvectors in PCA?

<details><summary><b>Answer</b></summary>



</details>

---


10. How do you decide the number of principal components to retain in PCA?

<details><summary><b>Answer</b></summary>

We use different approaches to get optimal number of principal components in PCA.

Certainly! Here are the main points summarized:

1. **Explained Variance (Cumulative Variance):** Retain enough components to explain a desired percentage (e.g., 90-95%) of the total variance.

<table align='center'>
<tr>
<td align="center">
    <img src="img/cummulative_var_pca.png" alt= "cumulative proportion of variance explained by the principal components" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> cumulative proportion of variance explained by the principal components </td>
</tr>
</table>

2. **Kaiser Criterion (Eigenvalue > 1):** Keep components with eigenvalues greater than 1, as they explain more variance than individual original variables.

3. **Scree Plot:** Identify the "elbow" in the plot of eigenvalues to decide the number of components to retain.

<table align='center'>
<tr>
<td align="center">
    <img src="img/scree_plot_pca.png" alt= "A scree plot depicting the proportion of variance explained by each of the principal components" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> A scree plot depicting the proportion of variance explained by each of the principal components  </td>
</tr>
</table>


4. **Cross-Validation:** Select components that optimize the performance of a predictive model through cross-validation if we are using PCA as a feature extraction tool.


</details>

---


11. What is the difference between PCA and Linear Discriminant Analysis (LDA)?

<details><summary><b>Answer</b></summary>

- PCA is unsupervised while LDA is supervised in training
- PCA focus on capturing maximum variance in the data without considering class labels, while LDA focus on maximizing class separability, taking class labels into the account.
- PCA outputs the principal components that capture variance while LDA output linear discriminants that enhance class separability.

</details>

---


12. What are the limitations of PCA?

<details><summary><b>Answer</b></summary>

Here are main limitations of PCA:

1. **Linearity Assumption:** PCA assumes linear relationships between variables, making it less effective for capturing non-linear patterns in the data.

2. **Loss of Interpretability:** The principal components are linear combinations of the original features, which can make them difficult to interpret.

3. **Variance-Based Selection:** PCA prioritizes components based on variance, not necessarily on their relevance to the target variable, which may not always lead to better predictive performance.

4. **Sensitivity to Scaling:** PCA is sensitive to the scale of the features, requiring careful standardization or normalization of data before applying the method.


</details>

---


14. Explain the concept of whitening in PCA.

<details><summary><b>Answer</b></summary>

PCA Whitening is a processing step for mainly image based data that makes input less redundant.

The goal of whitening is:

- the features are less correlated with each other
- the features all have the same variance.

Whitening has two simple steps:

- Project the dataset onto the eigenvectors. This rotates the dataset so that there is no correlation between the components.

- Normalize the the dataset to have a variance of 1 for all components. This is done by simply dividing each component by the square root of its eigenvalue.

</details>

---


## Recommendation Engines

9. Given this directed graph.
    ![image](img/dag.png)
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


