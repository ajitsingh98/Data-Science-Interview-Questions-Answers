# Data Science Interview Questions And Answers


Topics
---

- [Maximal Margin Classifier](#maximal-margin-classifier)
- [Support Vector Classifier](#support-vector-classifier)
- [Support Vector Machines](#support-vector-machines)
- [Support Vector Regression](#support-vector-regression)

---

## Maximal Margin Classifier

1. Is support vector machine(SVM) a generalization of maximal margin classifier?

<details><summary><b>Answer</b></summary>

True

</details>

---

1. What is a hyperplane?

<details><summary><b>Answer</b></summary>

In a p-dimensional space, a hyperplane is a flat affine subspace of dimension $p-1$. For example, in two dimensions, a hyperplane is a flat one-dimensional subspace i.e a line.

</details>

---

1. Write generic expression of a p-dimensional hyperplane?

<details><summary><b>Answer</b></summary>

$$\beta_0 + \beta_1X_1 + \beta_2X_2 + .... + \beta_pX_p  = 0$$

</details>

---

1. How to determine whether a point or a vector lies on a hyperplane?

<details><summary><b>Answer</b></summary>

We can put the given point in the hyperplane equation and check the sign. 

</details>

---

1. How does maximal margin classifier work?

<details><summary><b>Answer</b></summary>

Maximal margin classifier is a linear classifier that attempts to separate two classes by finding the hyperplane that maximizes the margin between the classes.

It works as follows:

Suppose we are working on a binary classification problem with $n$ training examples $\{(x_i, y_i)\}^n_{i=1}$ where
- $x_i \in \mathbb{R}^n$ is a $p$-dimensional feature vector
- $y_i \in \{-1, +1\}$ 

With the above setup, maximal margin classifier tries to find a hyperplane which can be represented as:
$$f(x) = w^Tx + b$$
where:
- $w \in \mathbb{R}^p$ is the weight vector (normal to the hyperplane)
- $b \in \mathbb{R}$ is the offset of the hyperplane from the origin

Decision boundary can be defined by:
$$f(x) = 0$$
$$ w^Tx + b  = 0$$

Now, we want to maximize the margin which is basically the distance between the decision boundary(hyperplane) and the closest points in the dataset(support vectors).  

Assume $x_i$ is a support vector, the distance(d) from the hyperplane can be expressed as

$$d = \frac{|w^Tx_i + b|}{||w||}$$

If we assume data is perfectly separable:

$$y_i(w^Tx_i + b) \ge 1$$

Using above two equations, margin can be derived as 

$$Margin = \frac{2}{||w||}$$

Note that here $2$ comes from the distance between the two support vectors(one from each class), each being at the distance of $\frac{1}{||w||}$ from the hyperplane.

So, maximal margin classifier tends to optimize the margin subject to the constraints that all points are classified correctly:
$$\min_{\mathbf{w}, b}\frac{1}{2}||w||^2$$
subject to:
$$y_i(w^Tx_i + b) \ge 1 \forall i$$

- Here we have used $||w||^2$ instead of $||w||$ to make the objective function convex and differential
- The constraints ensures that all points are on the correct side of the margin 

</details>

---


1. What is the main limitation of maximal margin classifier?

<details><summary><b>Answer</b></summary>

Maximal margin classifier has following limitations:

- It works well when the data is perfectly linearly separable, in case of non linearly separable cases it fails to find the optimal hyperplane.
- It is highly sensitive to outliers if they are close to decision boundary or wrong side of it.
- The maximal margin hyperplane is extremely sensitive to a change in a single observation suggests that it may have overfit the training data.

</details>

---

1. How can we overcome limitation of maximal margin classifier?

<details><summary><b>Answer</b></summary>

We can use soft margin i.e a hyperplane that almost separates the classes. It is also called as support vector classifier.

</details>

---

1. How can we overcome limitation of maximal margin classifier?

<details><summary><b>Answer</b></summary>

We can use soft margin i.e a hyperplane that almost separates the classes. It is also called as support vector classifier.

</details>

---

1. What is difference between Maximal margin classifier(MMC) and Support vector classifier(SVC)?

<details><summary><b>Answer</b></summary>

MMC seeks the largest possible margin so that every observation is on correct side of hyperplane as well as on the correct side of margin.  

But support vector machine, sometimes called as soft margin classifier allows some observations to be on incorrect side of the margin or even the incorrect side of hyperplane(The margin is soft because it can be violated by some training observations.)

</details>

---

1. How does Support vector classifier(SVC) works?

<details><summary><b>Answer</b></summary>

The support vector classifier classifies a test observations depending on which side of hyperplane it lies. The hyperplane is chosen to correctly separate most of the training observations into two classes, but may classify few observations.

It is the solution to the optimization problem:

$$\max_{\mathbf{\beta_0,..,\beta_p}, \epsilon_1,..,\epsilon_n, M} M$$
subjected to
$$\sum_{j=1}^p\beta_j^2 = 1$$
$$y_i(\beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + ... + \beta_px_{ip}) \ge M(1-\epsilon_i)$$
$$\epsilon_i \ge 0, \sum_{i=1}^{n}\epsilon_i \le C$$

Where $C$ is a nonnegative tuning parameter. $M$ is the width of the margin and $\epsilon_1,....\epsilon_n$ are slack variables that allows individual observations on the wrong side of the margin or the hyperplane.

Once we have hyperplane after solving above set of equations, we classify a test observations $x^*$, by simply determining on which side of hyperplane it lies.
$$Sign(f(x^*) = \beta_0 + \beta_1x_1^{*}+...+\beta_Px_p^*)$$

</details>

---

1. What kind of information we get from slack variables $\epsilon$ in SVC?

<details><summary><b>Answer</b></summary>

The slack variable $\epsilon_i$ tells us where the $i$th observations is located, relative to the hyperplane and relative to the margin.

- If $\epsilon_i > 0$ : $i$th observation is on the wrong side of the margin
- If $\epsilon_i = 0$ : $i$th observation is on the correct side of the margin
- If $\epsilon_i > 1$ : $i$th observation is on the wrong side of the hyperplane

</details>

---

1. How to interpret regularization parameter $C$ in SVC?

<details><summary><b>Answer</b></summary>

We can think of $C$ as a budget for the amount that margin can be violated by $n$ observations. If $C=0$ then there is no budget for violations to the margin i.e maximal margin classifier. As the budget $C$ increases, the model becomes more tolerant of violations to the margin, and so the margin will widen. Conversely, as C decreases, we become less tolerant of violations to the margin and so the margin narrows.

</details>

---

1. What are support vectors in SVC?

<details><summary><b>Answer</b></summary>

Observations that lie directly on the margin or on the wrong side of the margin for their class, are known as support vectors. 

</details>

---

1. Explain bias-variance tradeoff in SVC?

<details><summary><b>Answer</b></summary>

The regularization parameter $C$ controls the bias-variance trade-off. 

- C is small : Low bias and high variance 
- C is large : High bias and low variance 

</details>

---

1. Explain bias-variance tradeoff in SVC?

<details><summary><b>Answer</b></summary>

The regularization parameter $C$ controls the bias-variance trade-off. 

- C is small : Low bias and high variance 
- C is large : High bias and low variance 

</details>

---

1. Is SVC robust to outliers?

<details><summary><b>Answer</b></summary>

Yeah mostly, since the decision boundary is influenced only by support vectors, outliers that are far from the hyperplane(on the correct side) have little or no impact on the decision boundary.

</details>

---

1. What is the main limitation of SVC?

<details><summary><b>Answer</b></summary>

It struggles with the cases having non-linear decision boundary.

</details>

---

1. What is the main limitation of SVC?

<details><summary><b>Answer</b></summary>

It struggles with the cases having non-linear decision boundary.

</details>

---

1. Can we use feature space enlarging technique to solve non linear decision boundary problem with SVC?

<details><summary><b>Answer</b></summary>

Yeah we can address the problem of non-linear boundaries between classes by enlarging the feature space using quadratic, cubic or even higher order polynomial functions of the predictors. But there is an issue because there are many possible ways to enlarge the feature space and we may end up with huge number of features.  Then computation would become unmanageable. 

</details>

---

1. What is support vector machine(SVM)?

<details><summary><b>Answer</b></summary>

The support vector machine(SVM) is an extension of the support vector classifier that results from the enlarging the feature in a certain way using kernels.
</details>

---

1. What do you mean by kernel function in SVM?

<details><summary><b>Answer</b></summary>

Kernel function quantifies the similarity of two observations.

- Linear kernel
$$K(x_i, x_{i'}) = \sum_{j-1}^p{x_{ij}{x_{i'j}}}$$
- Polynomial kernel
$$K(x_i, x_i') = (1 + \sum_{j=1}^p{x_{ij}{x_{i'j}}})^d$$
where, $d > 1$

- Radial kernel
$$K(x_i,x_{i'}) = exp(-\gamma\sum_{j=1}^{p}(x_{ij} - x_{i'j})^2)$$

where, $\gamma$ is positive constant.


</details>

---

1. What is advantage of using kernel trick over simply enlarging feature space using functions of the original features?

<details><summary><b>Answer</b></summary>

Kernel trick is more computationally effective technique. We only need to compute $K(x_i, x'{_{i}})$ for all $\binom{n}{2}$ distinct pairs of $i, i'$. This can be done without explicitly working in the enlarged feature space. This is important because in many applications of SVMs, the enlarged feature space is so large that computations are intractable.

</details>

---

1. What is advantage of using kernel trick over simply enlarging feature space using functions of the original features?

<details><summary><b>Answer</b></summary>

Kernel trick is more computationally effective technique. We only need to compute $K(x_i, x'{_{i}})$ for all $\binom{n}{2}$ distinct pairs of $i, i'$. This can be done without explicitly working in the enlarged feature space. This is important because in many applications of SVMs, the enlarged feature space is so large that computations are intractable.

</details>

---

1. Is SVM a linear model or non-linear model?

<details><summary><b>Answer</b></summary>

SVM can be linear or non-linear depending on the kernels we are using. When we are using linear kernel then the decision boundary will be linear, but if we using non-linear kernels like radial basis function(RBF) or sigmoid kernel.

</details>

---

1. What is the kernel trick in Support Vector Machines (SVM), and how does it work?

<details><summary><b>Answer</b></summary>

The kernel trick in Support Vector Machines (SVM) is a technique used to handle non-linearly separable data. It allows SVM to find a hyperplane that can separate data points in higher-dimensional space without explicitly computing their coordinates in that space.

Instead of transforming the data explicitly:

- The SVM algorithm computes the kernel function directly between pairs of data points.
- This computation effectively simulates the inner products in the higher-dimensional space.
- The optimization problem remains in the original space, but the decision boundary can capture complex, non-linear relationships.

</details>

---

1. How can we set up SVMs to work with multi class classification problems?

<details><summary><b>Answer</b></summary>

We can use following approaches to accomplish it:

*One-versus-one classification*

To classify data with SVMs when there are $( K > 2 )$ classes, the *one-versus-one* (or *all-pairs*) approach is used. This method involves constructing $\binom{K}{2}$ SVM classifiers, each trained to distinguish between a pair of classes. For example, one SVM might compare the $k$th class with the $k'$th class. To classify a new observation, each of these classifiers is used to predict the class, and the observation is assigned to the class that receives the most votes from the pairwise classifiers.

*One-Versus-All classification*

In the one-versus-all (or one-versus-rest) approach for classifying with SVMs when there are $K > 2$ classes, $K$ separate SVMs are trained. Each SVM is designed to distinguish one class from the rest. For each SVM, the class of interest is coded as $+1$, while the remaining classes are coded as $-1$. 

To classify a new observation $x^*$, we compute the decision function values for all $K$ SVMs. The observation is assigned to the class for which this decision function value is the highest, indicating the strongest confidence that the observation belongs to that class.

</details>

---


1. Is the SVM unique in its use of kernels to enlarge the feature space to accommodate non-linear class boundaries?

<details><summary><b>Answer</b></summary>

No, we could use it for other classification methods as well like with logistic regression. For historical reasons, the use of non-linear kernel is attached with SVMs.

</details>

---


1. When should we use SVMs over logistic regression?

<details><summary><b>Answer</b></summary>

When the classes are well separated, SVMs tend to behave better than logistic regression; in more overlapping regimes, logistic regression is often preferred.

</details>

---

1. When should we use SVMs over logistic regression?

<details><summary><b>Answer</b></summary>

When the classes are well separated, SVMs tend to behave better than logistic regression; in more overlapping regimes, logistic regression is often preferred.

</details>

---

1. Given a true label $y = +1$ and a predicted score $\hat{y} = 0.5$, calculate the hinge loss for this classification example?

<details><summary><b>Answer</b></summary>

To calculate the hinge loss for the given true label and predicted score, use the hinge loss formula:

$$L(X, y, \beta) = \sum_{i=1}^{n}max[0, 1 - y_i(\beta_0 + \beta_1x_{i1} + ... + \beta_{p}x_{ip})]$$

On putting the given values:

$$Loss = max[0, 1 - 0.5] = 0.5$$

</details>



## Support Vector Machines

1. SVM.
    1. What’s linear separation? Why is it desirable when we use SVM?
    ![image](img/svm1.png)
    1. How well would vanilla SVM work on this dataset?
    ![image](img/svm2.png)
    1. How well would vanilla SVM work on this dataset?
    ![image](img/svm3.png)
    1. How well would vanilla SVM work on this dataset?
    ![image](img/svm4.png)

<details><summary><b>Answer</b></summary>



</details>

---

2. What is the main difference between Support Vector Machines (SVM) and Support Vector Regression (SVR)?

<details><summary><b>Answer</b></summary>



</details>

---

3. What are the key components of SVR?

<details><summary><b>Answer</b></summary>



</details>

---

4. What is the role of the ε-tube in SVR?

<details><summary><b>Answer</b></summary>



</details>

---

5. What are the different types of SVR Kernels, and when should they be used?

<details><summary><b>Answer</b></summary>



</details>

---

6. How do you tune the hyperparameters in SVR?

<details><summary><b>Answer</b></summary>



</details>

---

7. What is the significance of the regularization parameter (C) in SVR?

<details><summary><b>Answer</b></summary>



</details>

---

8. How does SVR handle outliers in the data?

<details><summary><b>Answer</b></summary>



</details>

---

9. What are the evaluation metrics used to assess the performance of SVR models?

<details><summary><b>Answer</b></summary>



</details>

---

10. What are the advantages and disadvantages of SVR compared to other regression techniques?

<details><summary><b>Answer</b></summary>



</details>

---

11. Can SVR be used for time series forecasting?


<details><summary><b>Answer</b></summary>



</details>

---

