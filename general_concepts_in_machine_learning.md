# Data Science Interview Questions And Answers


Topics
---

- [General Concepts In Machine Learning](#general-concepts-in-machine-learning)

## General Concepts In Machine Learning

Contents
----
- [Basics](#basics)
- [Cross Validation](#cross-validation)
- [Similarity Measures](#Similarity-Measures)
- [Sampling Techniques and Creating Training Data](#sampling-techniques-and-creating-training-data)
- [Objective Functions and Performance Matrices](#objective-functions-and-performance-matrices)
- [Feature Engineering](#feature-engineering)
- [Bias and Variance](#bias-and-variance)

### Basics

1. Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.

<details><summary><b>Answer</b></summary>
    

*Supervised Learning*

It uses labeled data to train a model. The model learns to predict outputs from inputs based on examples with known outcomes. Common tasks include classification and regression.

*Unsupervised Learning*

It works with unlabeled data to find hidden patterns or structures. It identifies clusters or relationships within the data without predefined labels. Examples include clustering and dimensionality reduction.

*Weakly Supervised Learning*

It uses data with noisy, incomplete, or inaccurate labels. The model is trained on this imperfect data to make predictions, often incorporating techniques to handle label uncertainty.

*Semi-Supervised Learning*

It combines a small amount of labeled data with a large amount of unlabeled data. The model leverages the labeled examples to better understand the structure of the unlabeled data and improve learning performance.

*Active Learning*

It involves an iterative process where the model actively selects the most informative examples to be labeled by an oracle (e.g., a human expert). This helps improve model performance efficiently by focusing on challenging or uncertain examples.


</details>

---


2. Empirical risk minimization.
    1. What’s the risk in empirical risk minimization?
    1. Why is it empirical?
    1. How do we minimize that risk?


<details><summary><b>Answer</b></summary>
    
</details>

---

3. Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?

<details><summary><b>Answer</b></summary>
    
</details>

---

5. If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?

<details><summary><b>Answer</b></summary>
    
</details>

---

6. The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?

<details><summary><b>Answer</b></summary>
    
</details>

---

7. What are saddle points and local minima? Which are thought to cause more problems for training large NNs?

<details><summary><b>Answer</b></summary>
    
</details>

---

8. Hyper-parameters.
    1. What are the differences between parameters and hyper-parameters?
    1. Why is hyperparameter tuning important?
    1. Explain algorithm for tuning hyper-parameters.


<details><summary><b>Answer</b></summary>
    
</details>

---

9. Classification vs. regression.
    1. What makes a classification problem different from a regression problem?
    1. Can a classification problem be turned into a regression problem and vice versa?

<details><summary><b>Answer</b></summary>
    
</details>

---

10. Parametric vs. non-parametric methods.
    1. What’s the difference between parametric methods and non-parametric methods? Give an example of each method.
    1. When should we use one and when should we use the other?

<details><summary><b>Answer</b></summary>
    
</details>

---

11. Why does ensembling independently trained models generally improve performance?

<details><summary><b>Answer</b></summary>
    
</details>

---

12. Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?

<details><summary><b>Answer</b></summary>
    
</details>

---

13. Why does an ML model’s performance degrade in production?

<details><summary><b>Answer</b></summary>
    
</details>

---

14. What problems might we run into when deploying large machine learning models?

<details><summary><b>Answer</b></summary>
    
</details>

---

15. Your model performs really well on the test set but poorly in production.
    1. What are your hypotheses about the causes?
    1. How do you validate whether your hypotheses are correct?
    1. Imagine your hypotheses about the causes are correct. What would you do to address them?

<details><summary><b>Answer</b></summary>
    
</details>

---

16. What are some common encoding techniques in machine learning?

<details><summary><b>Answer</b></summary>
    
</details>

---

### Cross Validation

#### Contents
- [CV approaches](#cv-approaches)
- K-Fold CV
- Stratification
- LOOCV

---

1.  Fig (8.1) depicts two different cross-validation approaches. Name them.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/cross-validation-1.png" alt= "Figure 8.1: Two CV approaches" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Figure 8.1: Two CV approaches </td>
  </tr>
</table>

<details><summary><b>Answer</b></summary>
    
</details>

---

2.  1. What is the purpose of following Python code snippet?
    ```python
    skf = StratifiedKFold(y, n_folds=5, random_state=989, shuffle=True)
    ```
    2. Explain the benefits of using the K-fold cross validation approach.
    3. Explain the benefits of using the Stratified K-fold cross validation approach.
    4. State the difference between K-fold cross validation and stratified cross validation.
    5. Explain in your own words what is meant by “We adopted a 5-fold cross-validation approach to estimate the testing error of the model”.

<details><summary><b>Answer</b></summary>
    
</details>

---

3. **True or False:** In a K-fold CV approach, the testing set is completely excluded from the process and only the training and validation sets are involved in this approach.

<details><summary><b>Answer</b></summary>
    
</details>

---

4.  **True or False:** In a K-fold CV approach, the final test error is:

$$
CV_k = 1/k\sum_{i=1}^{k}MSE
$$

<details><summary><b>Answer</b></summary>
    
</details>

---

5. Mark all the correct choices regarding a cross-validation approach:
    1. A 5-fold cross-validation approach results in 5-different model instances being fitted.
    1.  A 5-fold cross-validation approach results in 1 model instance being fitted over and over again 5 times.
    1. A 5-fold cross-validation approach results in 5-different model instances being fitted over and over again 5 times.
    1. Uses K-different data-folds.

<details><summary><b>Answer</b></summary>
    
</details>

---

6. Mark all the correct choices regarding the approach that should be taken to compute the performance of K-fold cross-validation:
    1. We compute the cross-validation performance as the arithmetic mean over the K per- formance estimates from the validation sets.
    1. We compute the cross-validation performance as the best one over the K performance estimates from the validation sets.

<details><summary><b>Answer</b></summary>
    
</details>

---

7. A data-scientist who is interested in classifying cross sections of histopathology image slices (8.2) decides to adopt a cross-validation approach he once read about in a book.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/stratification-1.png" alt= "Figure 8.2: A specific CV approach" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Figure 8.2: A specific CV approach </td>
  </tr>
</table>

Name the approach from the following options:

1. 3-fold CV
2. 3-fold CV with stratification
3. A (repeated) 3-fold CV

<details><summary><b>Answer</b></summary>
    
</details>

---

8. 1. **True or false**: The leave-one-out cross-validation (LOOCV) approach is a sub-case of k-fold cross-validation wherein K equals N , the sample size.
    1. **True or false**: It is always possible to find an optimal value n, K = n in K-fold cross-validation.

<details><summary><b>Answer</b></summary>
    
</details>

---

9. What is the main difference between RandomizedSearchCV and GridSearchCV?

<details><summary><b>Answer</b></summary>
    
</details>

---

10. When would you prefer to use RandomizedSearchCV over GridSearchCV, and vice versa?

<details><summary><b>Answer</b></summary>
    
</details>

---

11. What are the advantages of RandomizedSearchCV?

<details><summary><b>Answer</b></summary>
    
</details>

---

12. What are the advantages of GridSearchCV?

<details><summary><b>Answer</b></summary>
    
</details>

---

13. What is cross-validation in the context of hyperparameter tuning?

<details><summary><b>Answer</b></summary>
    
</details>

---

14. Can you combine RandomizedSearchCV and GridSearchCV techniques for hyperparameter tuning?

<details><summary><b>Answer</b></summary>
    
</details>

---


### Similarity Measures

- Image, text similarity
- Jcard similarity
- The Kullback-Leibler Distance
- Min Hash
---

23. A data scientist extracts a feature vector from an image using a pre-trained ResNet34 CNN as follows

```python
import torchvision.models as models
...
res_model = models.resnet34(pretrained=True)
```
He then applies the following algorithm, entitled xxx on the image (9.2).

```python
import math

def xxx(arr):
    mod = 0.0
    
    for i in arr:
        mod += i * i
    
    mag = math.sqrt(mod)
    
    for i in range(len(arr)):
        arr[i] /= mag

# Example usage:
arr = [1.0, 2.0, 3.0]
xxx(arr)
print(arr)
```
Which results in this list:

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/similarity-1.png" style="max-width:70%;" />
    </td>
  </tr>
</table>

Name the algorithm that he used and explain in detail why he used it.

<details><summary><b>Answer</b></summary>
    
</details>

---

24. Further to the above, the scientist then applies the following algorithm:

<details><summary><b>Answer</b></summary>
    
</details>

---

**Algo 1**

Data: Two vectors v1 and v2 are provided Apply algorithm xxx on the two vectors Run algorithm 2

**Algo 2**

```python
def algo2(v1, v2):
    mul = 0.0

    for i in range(len(v1)):
        mul += v1[i] * v2[i]
    if mul < 0:
        return 0
    
    return mul
```
1. Name the algorithm algo2 that he used and explain in detail what he used it for.
2. Write the mathematical formulae behind it.
3. What are the minimum and maximum values it can return?
4. An alternative similarity measures between two vectors is: $\text{simeuc}(v1, v2) = -\|v1 - v2\|$.
Name the measure.

<details><summary><b>Answer</b></summary>
    
</details>

---

25. 1. What is the formulae for the Jaccard similarity of two sets?
    2. Explain the formulae in plain words.
    3. Find the Jacard similarity given the sets depicted in (8.13)

<table align='center'>
<tr>
  <td align="center">
    <img src="img/similarity-2.png" alt= "FIGURE 8.13: Jaccard similarity." style="max-width:70%;" />
  </td>
</tr>
<tr>
  <td align="center"> FIGURE 8.13: Jaccard similarity.</td>
</tr>
</table>

4. Compute the Jaccard similarity of each pair of the following sets:
    1. 12, 14, 16, 18
    2. 11, 12, 13, 14, 15
    3. 11, 16, 17

<details><summary><b>Answer</b></summary>
    
</details>

---

26. In this problem, you have to actually read 4 different papers, so you will probably not encounter such a question during an interview, however reading academic papers is an ex- cellent skill to master for becoming a DL researcher.

The Kullback-Leibler divergence is a meas- ure of how different two probability distribution are. As noted, the KL divergence of the probability distributions P, Q on a set X is defined as shown in Equation 8.11.

$$
D_{KL}(P \| Q) = \sum_{x \in X} P(x) \log\left(\frac{P(x)}{Q(x)}\right)
$$

Note however that since KL divergence is a non-symmetric information theoretical meas- ure of distance of P from Q, then it is not strictly a distance metric. During the past years, various KL based distance measures (rather than divergence based) have been introduced in the literature generalizing this measure.
Name each of the following KL based distances:

$$
D_{KLD1}(P \| Q) = D_{KL}(P \| Q) + D_{KL}(Q \| P)
$$

$$
D_{KLD2}(P \| Q) = \sum_{x \in X} (P(x) - Q(x)) \log(P(x))
$$

$$
D_{KLD3}(P \| Q) = \frac{1}{2} [D_{KL}\left(Q\|\right(\frac{P+Q}{2})) +  D_{KL}\left(P\|\right(\frac{P+Q}{2})) ]
$$

$$
D_{KLD4}(P \| Q) = max(D_{KL}\left(Q\|\right(P)) +  D_{KL}\left(P\|\right(Q)))
$$

<details><summary><b>Answer</b></summary>
    
</details>

---


### Sampling Techniques and Creating Training Data

1. If you have 6 shirts and 4 pairs of pants, how many ways are there to choose 2 shirts and 1 pair of pants?

<details><summary><b>Answer</b></summary>
    
</details>

---

2. What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?

<details><summary><b>Answer</b></summary>
    
</details>

---

3. Explain Markov chain Monte Carlo sampling.

<details><summary><b>Answer</b></summary>
    
</details>

---

4. If you need to sample from high-dimensional data, which sampling method would you choose?

<details><summary><b>Answer</b></summary>
    
</details>

---

5. Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it’ll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms.

<details><summary><b>Answer</b></summary>
    
</details>

---

6. Suppose you want to build a model to classify whether a Reddit comment violates the website’s rule. You have $10$ million unlabeled comments from $10K$ users over the last $24$ months and you want to label $100K$ of them.
    1. How would you sample $100K$ comments to label?
    1. Suppose you get back $100K$ labeled comments from $20$ annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them?

<details><summary><b>Answer</b></summary>
    
</details>

---

7. Suppose you work for a news site that historically has translated only $1%$ of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?

<details><summary><b>Answer</b></summary>
    
</details>

---

8. How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?

<details><summary><b>Answer</b></summary>
    
</details>

---

9. How do you know you’ve collected enough samples to train your ML model?

<details><summary><b>Answer</b></summary>
    
</details>

---

10. How to determine outliers in your data samples? What to do with them?

<details><summary><b>Answer</b></summary>
    
</details>

---

11. Sample duplication
    1. When should you remove duplicate training samples? When shouldn’t you?
    1. What happens if we accidentally duplicate every data point in your train set or in your test set?

<details><summary><b>Answer</b></summary>
    
</details>

---


12. Missing data
    1. In your dataset, two out of 20 variables have more than 30% missing values. What would you do?
    1. How might techniques that handle missing data make selection bias worse? How do you handle this bias?

<details><summary><b>Answer</b></summary>
    
</details>

---


13. Why is randomization important when designing experiments (experimental design)?

<details><summary><b>Answer</b></summary>
    
</details>

---


14. Class imbalance.
    1. How would class imbalance affect your model?
    1. Why is it hard for ML models to perform well on data with class imbalance?
    1. Imagine you want to build a model to detect skin legions from images. In your training dataset, only $1%$ of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?

<details><summary><b>Answer</b></summary>
    
</details>

---


15. Training data leakage.
    1. Imagine you're working with a binary task where the positive class accounts for only 1% of your data. You decide to oversample the rare class then split your data into train and test splits. Your model performs well on the test split but poorly in production. What might have happened?
    1. You want to build a model to classify whether a comment is spam or not spam. You have a dataset of a million comments over the period of 7 days. You decide to randomly split all your data into the train and test splits. Your co-worker points out that this can lead to data leakage. How?

<details><summary><b>Answer</b></summary>
    
</details>

---


16. How does data sparsity affect your models?

<details><summary><b>Answer</b></summary>
    
</details>

---


17. Feature leakage
    1. What are some causes of feature leakage?
    1. Why does normalization help prevent feature leakage?
    1. How do you detect feature leakage?

<details><summary><b>Answer</b></summary>
    
</details>

---


18. Suppose you want to build a model to classify whether a tweet spreads misinformation. You have 100K labeled tweets over the last 24 months. You decide to randomly shuffle on your data and pick 80% to be the train split, 10% to be the valid split, and 10% to be the test split. What might be the problem with this way of partitioning?

<details><summary><b>Answer</b></summary>
    
</details>

---


19. You’re building a neural network and you want to use both numerical and textual features. How would you process those different features?

<details><summary><b>Answer</b></summary>
    
</details>

---


20. Your model has been performing fairly well using just a subset of features available in your data. Your boss decided that you should use all the features available instead. What might happen to the training error? What might happen to the test error?

<details><summary><b>Answer</b></summary>
    
</details>

---



### Objective Functions and Performance Metrices

1. Convergence.
    1. When we say an algorithm converges, what does convergence mean?
    1. How do we know when a model has converged?

<details><summary><b>Answer</b></summary>
    
</details>

---


2. Draw the loss curves for overfitting and underfitting.

<details><summary><b>Answer</b></summary>
    
</details>

---


3. Bias-variance trade-off
    1.  What’s the bias-variance trade-off?
    1. How’s this tradeoff related to overfitting and underfitting?
    1. How do you know that your model is high variance, low bias? What would you do in this case?
    1. How do you know that your model is low variance, high bias? What would you do in this case?

<details><summary><b>Answer</b></summary>
    
</details>

---


4. Cross-validation.
    1. Explain different methods for cross-validation.
    1. Why don’t we see more cross-validation in deep learning?

<details><summary><b>Answer</b></summary>
    
</details>

---


5. Train, valid, test splits.
    1. What’s wrong with training and testing a model on the same data?
    1. Why do we need a validation set on top of a train set and a test set?
    1. Your model’s loss curves on the train, valid, and test sets look like this. What might have been the cause of this? What would you do?
    ![image](assets/loss_training.png)

<details><summary><b>Answer</b></summary>
    
</details>

---


6. Your team is building a system to aid doctors in predicting whether a patient has cancer or not from their X-ray scan. Your colleague announces that the problem is solved now that they’ve built a system that can predict with 99.99% accuracy. How would you respond to that claim?

<details><summary><b>Answer</b></summary>
    
</details>

---


7. F1 score.
    1. What’s the benefit of F1 over the accuracy?
    1. Can we still use F1 for a problem with more than two classes. How?

<details><summary><b>Answer</b></summary>
    
</details>

---


8. Given a binary classifier that outputs the following confusion matrix.

```math
\begin{bmatrix} 
	"" & Predicted True & Predicted False \\
	Actual True & 30 & 20\\
	Actual False & 5 & 40 \\
	\end{bmatrix}
```
    1. Calculate the model’s precision, recall, and F1.
    1. What can we do to improve the model’s performance?

<details><summary><b>Answer</b></summary>
    
</details>

---


9. Consider a classification where $99%$ of data belongs to class A and $1%$ of data belongs to class B.
    1. If your model predicts A 100% of the time, what would the F1 score be? Hint: The F1 score when A is mapped to 0 and B to 1 is different from the F1 score when A is mapped to 1 and B to 0.
    1. If we have a model that predicts A and B at a random (uniformly), what would the expected $F_1$ be?

<details><summary><b>Answer</b></summary>
    
</details>

---


10. For logistic regression, why is log loss recommended over MSE (mean squared error)?

<details><summary><b>Answer</b></summary>
    
</details>

---


11. When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?

<details><summary><b>Answer</b></summary>
    
</details>

---


12. Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.

<details><summary><b>Answer</b></summary>
    
</details>

---


13. For classification tasks with more than two labels (e.g. MNIST with $10$ labels), why is cross-entropy a better loss function than MSE?

<details><summary><b>Answer</b></summary>
    
</details>

---


14. Consider a language with an alphabet of $27$ characters. What would be the maximal entropy of this language?

<details><summary><b>Answer</b></summary>
    
</details>

---


15. A lot of machine learning models aim to approximate probability distributions. Let’s say P is the distribution of the data and Q is the distribution learned by our model. How do measure how close Q is to P?

<details><summary><b>Answer</b></summary>
    
</details>

---


16. MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori)
    1. How do MPE and MAP differ?
    1. Give an example of when they would produce different results.

<details><summary><b>Answer</b></summary>
    
</details>

---


17. Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than $10%$ from the actual price. Which metric would you use?

<details><summary><b>Answer</b></summary>
    
</details>

---



### Feature Engineering

4. Feature selection.
    1. Why do we use feature selection?
    1. What are some of the algorithms for feature selection? Pros and cons of each.

<details><summary><b>Answer</b></summary>
    
</details>

---


11. Is feature scaling necessary for kernel methods?

<details><summary><b>Answer</b></summary>
    
</details>

---


12. What are the different types of feature selection techniques?

<details><summary><b>Answer</b></summary>
    
</details>

---



### Bias and Variance
1. Explain Bias and Variance?

<details><summary><b>Answer</b></summary>
    
</details>

---


1. Why is the bias-variance tradeoff important in machine learning?

<details><summary><b>Answer</b></summary>
    
</details>

---


2. How can you tell if your model has a high bias or high variance problem?

<details><summary><b>Answer</b></summary>
    
</details>

---


3. What are some techniques to reduce bias in a model?

<details><summary><b>Answer</b></summary>
    
</details>

---


4. What are some techniques to reduce variance in a model?

<details><summary><b>Answer</b></summary>
    
</details>

---


5. Can you explain cross-validation's role in addressing the bias-variance tradeoff?

<details><summary><b>Answer</b></summary>
    
</details>

---


6. Is it always better to reduce bias and variance simultaneously?

<details><summary><b>Answer</b></summary>
    
</details>

---

