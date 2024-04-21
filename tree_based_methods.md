# Data Science Interview Questions And Answers


Topics
---

## Tree Based Methods in Machine Learning

Contents
----

- Decison Trees
- Bagging, Boosting, Stacking and Blending
- Approaches for Combining Predictors
- Monolithic and Heterogeneous Ensembling
- Ensemble Learning
- Snapshot Ensembling 
- Multi-model Ensembling
- Learning-rate Schedules in Ensembling
- Random Forest
- Boosting based Algorithms

---

1. What is a decision tree?

<details><summary><b>Answer</b></summary>
Decision Trees(DTs) are non-parametric supervised learning method which can be employed for Classification and Regressions tasks.
</details>

---
2. What is the purpose of decision trees in machine learning?

<details><summary><b>Answer</b></summary>
The main purpose of DTs in machine learning is to model data by learning simple decision rules inferred from the attributes of the datasets. A decision tree can be seen as piecewise constant approximation.
</details>

---
3. How is a decision tree built?

<details><summary><b>Answer</b></summary>
The main purpose of DTs in machine learning is to model data by learning simple decision rules inferred from the attributes of the datasets. A decision tree can be seen as piecewise constant approximation.
</details>

---
4. What is over-fitting in decision trees, and how can it be prevented?
5. What are some common impurity measures used in decision tree algorithms?
6. What is pruning in decision trees?
7. Can decision trees handle categorical data, and how is it done?

---

8. What are some advantages of decision trees in machine learning?

<details><summary><b>Answer</b></summary>
1. Simple to understand, interpret and visualize.
2. Requires little data preparations like normalization. Some tree based methods even provide support missing values.
3. The cost of predictions is logarithmic in the number of data points used in training the model.
4. Able to handle multi-output problems.
</details>
---

9. What are some limitations of decision trees?

<details><summary><b>Answer</b></summary>
1. More prone to overfitting the underlying pattern if they grow in uncontrollable manner.
2. DTs can be unstable sometimes like small deviation in data might result in completely different tree all together.
3. Not good at extrapolation since the decision boundaries are piecewise constant approximation.
4. Does not perform well incase of data imbalance scenarios.
</details>

---

11. What is ID3, and how does it work?

<details><summary><b>Answer</b></summary>
ID3 stands for Iterative Dichotomiser 3 and It means the model iteratively(repeatedly) dichotomizes(divides) features into two or more groups at each step.

It uses top down greedy approach ti build a decision tree and was invented by Ross Quinlan. Top-down means we start building the tree from the top and greedy approach means that at each iteration we select the best feature at the preset moment to create a node.

</details>

---


12. What is information gain in ID3?

<details><summary><b>Answer</b></summary>

Information Gain calculates the reduction in the entropy and measures how good a given attribute split the target classes. The feature with highest information gain is the best one.

Information Gain for a feature column A is calculated as:

$$
IG(S, A) = Entropy(S) - \sum{((|S_v|/|S|)*Entropy(S_v))}
$$

where $S_v$ is the set of rows in $S$ for which the feature column $A$ has value $v$, $|Sᵥ|$ is the number of rows in $S_v$ and likewise $|S|$ is the number of rows in $S$.

</details>

---

18. What are the steps involved in building a decision tree with ID3?

<details><summary><b>Answer</b></summary>

The following steps involve in building a ID3 trees:
1. Calculate the Information Gain for each attribute.
2. Split the dataset $S$ into subsets with the attribute having highest IG.
3. Make a decision tree node using the feature with maximum Information Gain.
4. If all the rows belong to same class, make the current node as leaf node with the class as it label.
5. Repeat for the remaining feature until we run out of all features or the decision tree has all leaf nodes.

</details>

---

13. What are the limitations of ID3?

<details><summary><b>Answer</b></summary>

Limitations of ID3:

- ID3 follows greedy algorithm while building the decision trees and hence can provide suboptimal solutions sometimes.
- It can overfit the train data. Smaller decision trees should be preferred over larger decision trees.
- ID3 is mainly good with nominal features so continuous features can be only used after converting them to nominal bins.

</details>

---

15. How does ID3 handle over-fitting?
16. What is the difference between ID3 and C4.5?

<details><summary><b>Answer</b></summary>

| Feature                         | ID3                                                        | C4.5                                                         |
|---------------------------------|------------------------------------------------------------|--------------------------------------------------------------|
| **Splitting Criteria**          | Uses information gain                                      | Uses gain ratio as splitting criteria                        |
| **Handling of Numeric Attributes** | Cannot handle numeric attributes directly                | Can handle numeric attributes                                |
| **Pruning Method**              | Generally does not use pruning                             | Uses error-based pruning after the growing phase             |
| **Handling of Missing Values**  | Typically does not handle missing values                   | Allows attribute values to be missing (marked as ?)          |
| **Handling of Continuous Attributes** | Does not handle continuous attributes                | Handles continuous attributes by binary splitting. Searches for the best threshold that maximizes the gain ratio. |
| **Branching and Pruning**       | Simple growth with no specific branching procedure         | Implements a pruning procedure to remove branches that do not contribute to accuracy, replacing them with leaf nodes |
| **Performance on Small Datasets** | Not specifically mentioned                               | Performs better than J48 and C5.0 on small datasets according to comparative studies |


</details>

---

17. Can you explain how the concept of entropy is used in ID3?

<details><summary><b>Answer</b></summary>

Here's how entropy is used in the ID3 algorithm:

- **Dataset Splitting**: ID3 uses entropy to decide which attribute to split the data on at each step in the tree. The goal is to find the attribute that results in the highest gain in information or the largest decrease in entropy. This is done by calculating the entropy before and after the dataset is split on each attribute.

- **Information Gain**: The information gain for an attribute is calculated as the difference between the entropy of the parent dataset and the weighted sum of the entropies of the subsets that result from splitting the dataset on the attribute. The formula for information gain $\( IG \)$ is:
  $$
  \[
  IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
  \]
  $$
  Here, \( A \) is the attribute being considered for splitting, \( Values(A) \) are the different values that \( A \) can take, \( S_v \) is the subset of \( S \) for which \( A \) has value \( v \), and \( |S_v|/|S| \) is the proportion of the number of elements in \( S_v \) to the number of elements in \( S \).

- **Selecting the Best Attribute**: The attribute with the highest information gain is chosen for the split because it provides the most significant reduction in entropy, indicating a more definitive classification rule at that node of the tree.

- **Recursive Splitting**: This process is repeated recursively for each new subset, choosing the attribute that yields the highest information gain at each stage until a stopping criterion is met (like when no more significant information gain is possible or the subset at a node all belongs to the same class).


</details>

19. What are different criteria along which the implementation of DTs varies ?

<details><summary><b>Answer</b></summary>

- Criteria for node splitting (e.g., methods for calculating "variance")
- Capability to develop both regression models (for continuous variables like scores) and classification models (for discrete variables like class labels)
- Strategies to prevent or minimize overfitting
- Ability to process datasets with missing values

</details>

19. What is the difference between CART and ID3/C4.5?

<details><summary><b>Answer</b></summary>

| **Aspect**           | **CART (Classification and Regression Trees)**                        | **ID3 (Iterative Dichotomiser 3)**                                  | **C4.5**                                       |
|----------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|------------------------------------------------|
| **Type of Trees**    | Can create both classification and regression trees.                | Primarily used for creating classification trees.                  | Used for creating classification trees.         |
| **Splitting Criteria**| Uses Gini impurity or entropy for classification; variance reduction for regression.| Uses information gain (based on entropy).                          | Uses gain ratio (a normalization of information gain). |
| **Handling Continuous and Categorical Data** | Handles both continuous and categorical variables directly.        | Primarily handles categorical variables; continuous data must be discretized prior to building the tree.| Handles both but often requires discretization of continuous variables. |
| **Pruning**          | Uses cost-complexity pruning to avoid overfitting, which is a post-pruning technique. | Does not include a pruning step, leading to potentially overfitted trees. | Uses post-pruning methods to simplify the tree after it is fully grown. |
| **Handling Missing Values** | Has mechanisms to handle missing values directly during tree construction. | Does not handle missing values inherently; preprocessing is required. | Has improved strategies to deal with missing values compared to ID3. |

</details>

20. How does CART handle over-fitting?

<details><summary><b>Answer</b></summary>

CART (Classification and Regression Trees) handles overfitting primarily through two techniques: **pruning** and **setting constraints during the tree building process**.

1. **Pruning:**
   Pruning reduces the size of a decision tree by removing parts of the tree that provide little power in classifying instances. This process helps in reducing the complexity of the final model, thereby minimizing overfitting. There are two types of pruning:
   - **Pre-pruning (early stopping rule):** This method stops the tree from growing when further splitting is statistically unlikely to add value. This could be determined by setting a minimum number on the gain of a node’s split, or restricting the depth of the tree.
   - **Post-pruning:** This involves building the tree first and then removing non-significant branches. A common approach is to use cost-complexity pruning where a penalty is applied for the number of parameters (or the depth) of the tree, aiming to find a good trade-off between the tree’s complexity and its accuracy on the training set.

2. **Setting Constraints:**
   By setting constraints during the building of the tree, you can also control overfitting:
   - **Maximum depth of the tree:** Limiting the depth prevents the model from creating highly complex trees that fit all the details and noise in the training data.
   - **Minimum samples split:** This constraint specifies the minimum number of samples a node must have before it can be split. Higher values prevent the model from learning overly fine distinctions.
   - **Minimum samples leaf:** This parameter ensures that each leaf node has a minimum number of samples. This helps in creating more generalized regions in the leaf nodes rather than very specific rules that might apply only to the training data.
   - **Maximum leaf nodes:** Setting a maximum number of leaf nodes helps in controlling the size of the tree.

</details>
---

1. Mark all the approaches which can be utilized to boost a single model performance:
    1. Majority Voting
    2. Using K-identical base-learning algorithms
    3. Using K-different base-learning algorithms 
    4. Using K-different data-folds
    5. Using K-different random number seeds 
    6. A combination of all the above approaches

<details><summary><b>Answer</b></summary>
All options are correct.
</details>
---
1. How does stacking differ from other ensemble methods like bagging and boosting?

---

2. What are the key components of a stacking ensemble?

---

3. How do you prevent over-fitting in a stacked ensemble?

---

4. Can you explain the process of creating a stacking ensemble?

---

5. What is the advantage of stacking over using a single powerful model?

---

6. What are some popular algorithms used as base models in stacking ensembles?

---

7. Are there any limitations or challenges associated with stacking ensembles?

---

8. Can you explain the difference between stacking and blending?

---

9. When should you consider using stacking in a machine learning project?

---

10. How does blending work?

---

11. What is the purpose of a meta-model in blending?

---

12. What are the advantages of blending?

---

13. What are the common algorithms used for blending?

---

14.  What precautions should you take when implementing blending?

---

15. Can you explain the difference between bagging, boosting, and blending?

---

16.  When should you consider using blending in a machine learning project?

---

17. What challenges can arise when implementing blending in practice?

---
2. An argument erupts between two senior data-scientists regarding the choice of an approach for training of a very small medical corpus. One suggest that bagging is superior while the other suggests stacking. Which technique, bagging or stacking, in your opinion is superior? Explain in detail.
    1. Stacking since each classier is trained on all of the available data.
    2. Bagging since we can combine as many classifiers as we want by training each on a different sub-set of the training corpus.

---

3.  Complete the sentence: A random forest is a type of a decision tree which utilizes `[bagging/boosting]`

---

4. The algorithm depicted in Fig. 6.1 was found in an old book about ensembling. Name the
algorithm.

```
Algorithm 1: Algo 1
```
**Data:** A set of training data, Q with N elements has been established 

while K times do

        Create a random subset of N ′ data by sampling from Q containing the N samples;
        N′ < N;
        Execute algorithm Algo 2; 
        Return all N′ back to Q

```
Algorithm 2: Algo 2
```
Choose a learner $h_m$; 

while K times do

        Pick a training set and train with $h_m$;
    
---
11. What is a Random Forest, and how does it work?
10. How do you choose between different types of decision tree algorithms (e.g., CART, ID3, C4.5, Random Forest)?
12. What is the difference between a decision tree and a Random Forest?
13. Why is it called a "Random" Forest?
14. What is the purpose of feature bagging in a Random Forest?
15. How does a Random Forest handle missing data?
16. What are the advantages of using Random Forests?
17. What is out-of-bag error, and how is it used in Random Forests?
18. Can you explain the concept of feature importance in a Random Forest?
19. What are some potential drawbacks of using Random Forests?
20. When would you choose a Random Forest over other machine learning algorithms?

5. Fig. 6.2 depicts a part of a specific ensembling approach applied to the models $x1, x2...xk$.

In your opinion, which approach is being utilized?
<table align='center'>
<tr>
<td align="center">
    <img src="img/nn_ensemble-1.png" alt= "A specific ensembling approach" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> A specific ensembling approach </td>
</tr>
</table>
(i) Bootstrap aggregation 

(ii) Snapshot ensembling

(iii) Stacking

(iv) Classical committee machines

---

6. Consider training corpus consisting of balls which are glued together as triangles, each
of which has either $1, 3, 6, 10, 15, 21, 28, 36, or 45 balls$.

1. We draw several samples from this corpus as presented in Fig.6.3 where in each sample is equiprobable. What type of sampling approach is being utilized here?
<table align='center'>
<tr>
<td align="center">
    <img src="img/nn_ensemble-2.png" alt= "Sampling approaches" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> Sampling approaches</td>
</tr>
</table>

(i) Sampling without replacement 

(ii) Sampling with replacement

2. Two samples are drawn one after the other. In which of the following cases is the covariance between the two samples equals zero?

(i) Sampling without replacement 

(ii) Sampling with replacement

3. During training, the corpus sampled with replacement and is divided into several folds as presented in Fig. 6.4.
<table align='center'>
<tr>
<td align="center">
    <img src="img/nn_ensemble-3.png" alt= "Sampling approaches" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> Sampling approaches</td>
</tr>
</table>
If 10 balls glued together is a sample event that we know is hard to correctly classify, then it is impossible that we are using:

(i) Bagging 

(ii) Boosting

---

7. There are several methods by which the outputs of base classifiers can be combined to yield a single prediction. Fig. 6.5 depicts part of a specific ensembling approach applied to several CNN model predictions for a labelled data-set. Which approach is being utilized?
    1. Majority voting for binary classification
    2. Weighted majority voting for binary classification
    3. Majority voting for class probabilities (iv) Weighted majority class probabilities
    4. An algebraic weighted average for class probabilities
    5. An adaptive weighted majority voting for combining multiple classifiers

```python

l=[]
for i,f in enumerate(filelist):
    temp = pd.read_csv(f)
    l.append(temp)
arr = np.stack(l,axis=-1)
avg_results = pd.DataFrame(arr[:,:-1,:].mean(axis=2))
avg_results['image'] = l[0]['image']
avg_results.columns = l[0].columns

```

---

8. Read the paper **Neural Network Ensembles** and then **complete the sentenc**e: If the average error rate for a specific instance in the corpus is less than [...]% and the respective classifiers in the ensemble produce independent [...], then when the number of classifiers combined approaches infinity, the expected error can be diminished to zero.

---

9. **True or False**: A perfect ensemble comprises of highly correct classifiers that differ as much as possible.

---
9. How does bagging work?
---
10. What are the advantages of bagging over decision trees?
---
11. What are some popular algorithms that use bagging?
---
12. What's the difference between bagging and boosting?
---
13. How does bagging handle imbalanced datasets?
---
14. Can bagging be used with any base model?
---
15. What are some potential drawbacks of bagging?
---
16. What is the trade-off between bagging and variance?
---
10. **True or false**: In bagging, we re-sample the training corpus with replacement and there-
fore this may lead to some instances being represented numerous times while other instances not to be represented at all.

---

8. Bagging and boosting are two popular ensembling methods. Random forest is a bagging example while XGBoost is a boosting example.
    1. What are some of the fundamental differences between bagging and boosting algorithms?
    1. How are they used in deep learning?

---
8. How does boosting work?
---
9. What are some popular boosting algorithms?
---
10. What is the key idea behind AdaBoost?
---
11. What is overfitting, and how does boosting address it?
---
12. Can boosting models handle noisy data?
---
13. What are the hyperparameters in boosting algorithms?
---
14. What is the key idea behind XGBoost?
---
15. What are some advantages of using XGBoost?
---
16. How does LightGBM differ from traditional gradient boosting algorithms?
---
17. What is the trade-off between LightGBM's speed and memory consumption?
---
18. How does CatBoost handle categorical features?
---
19. What are some benefits of using CatBoost for gradient boosting?
---
11. 1. **True or false**: Training an ensemble of a single monolithic architecture results in lower model diversity and possibly decreased model prediction accuracy.
    2. **True or false**: The generalization accuracy of an ensemble increases with the number of well-trained models it consists of.
    3. **True or false**: Bootstrap aggregation (or bagging), refers to a process wherein a CNN ensemble is being trained using a random subset of the training corpus.
    4. **True or false**: Bagging assumes that if the single predictor shave independent errors, then a majority vote of their outputs should be better than the individual predictions.

---

12. Refer to the papers: <a href='https://arxiv.org/pdf/1506.02142.pdf'>Dropout as a Bayesian Approximation</a> and <a href='https://arxiv.org/pdf/1906.02530.pdf'>Can You TrustYour Model’s Uncertainty?</a> and answer the following question: 
    1. Do deep ensembles achieve a better performance on out-of-distribution uncertainty benchmarks compared with Monte-Carlo (MC)-dropout?

---

13. 1. In a transfer-learning experiment conducted by a researcher, a number of ImageNet-pretrained CNN classifiers, selected from Table 6.1 are trained on five different folds drawn from the same corpus. Their outputs are fused together producing a composite machine. Ensembles of these convolutional neural networks architectures have been extensively studies an evaluated in various ensembling approaches. Is it likely that the composite machine will produce a prediction with higher accuracy than that of any individual classifier? Explain why.
<table align='center'>
<tr>
<td align="center">
    <img src="img/nn_ensemble-4.png" alt= "ImageNet-pretrained CNNs. Ensembles of these CNN architectures have been extensively studies and evaluated in various ensembling approaches" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> ImageNet-pretrained CNNs. Ensembles of these CNN architectures have been extensively studies and evaluated in various ensembling approaches </td>
</tr>
</table>

    2. **True or False**: In a classification task, the result of ensembling is always superior.

    3. **True or False**: In an ensemble, we want differently trained models converge to different local minima.

---

14. In committee machines, mark all the combiners that do not make direct use of the input:
    1. A mixture of experts 
    2. Bagging
    3. Ensemble averaging 
    4. Boosting

---

15. **True or False**: Considering a binary classification problem $(y = 0\ or \ y = 1)$, ensemble
averaging, wherein the outputs of individual models are linearly combined to produce a fused output is a form of a static committee machine.
<table align='center'>
<tr>
<td align="center">
    <img src="img/nn_ensemble-5.png" alt= "A typical binary classification problem" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> A typical binary classification problem </td>
</tr>
</table>

---

16. **True or false**: When using a single model, the risk of overfitting the data increases when
the number of adjustable parameters is large compared to cardinality (i.e., size of the set) of the training corpus.

---

17.  **True or false**: If we have a committee of $K$ trained models and the errors are uncorrelated,
then by averaging them the average error of a model is reduced by a factor of $K$.

---

18. 1. Define ensemble learning in the context of machine learning.
    2. Provide examples of ensemble methods in classical machine-learning.
    3. **True or false**: Ensemble methods usually have stronger generalization ability.
    4. Complete the sentence: Bagging is `variance/bias` reduction scheme while boosting reduced `variance/bias`.

---

19. Your colleague, a well-known expert in ensembling methods, writes the following pseudo
code in Python shown in Fig. 6.7 for the training of a neural network. This runs inside a standard loop in each training and validation step.

```python

import torchvision.models as models ...
models = ['resnext']
for m in models: 
    train ...
    compute VAL loss ... 
    amend LR ...
    if (val_acc > 90.0):
        saveModel()

```
1. What type of ensembling can be used with this approach? Explain in detail.
2. What is the main advantage of snapshot ensembling? What are the disadvantages, if any?

---

20. Assume further that your colleague amends the code as follows in Fig. 6.8.

```python

import torchvision.models as models
import random
import np
...
models = ['resnext']
for m in models:
    train ...
    compute loss ...
    amend LR ...
    manualSeed= draw a new random number
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if (val_acc > 90.0):
        saveModel()

```
Explain in detail what would be the possible effects of adding `lines 10-13`.

---

21. 1. Assume your colleague, a veteran in DL and an expert in ensembling methods writes the following Pseudo code shown in Fig. 6.9 for the training of several neural networks. This code snippet is executed inside a standard loop in each and every training/validation epoch.

```python

import torchvision.models as models 
...
models = ['resnext','vgg','dense']
for m in models: 
    train ...
    compute loss/acc 
    ... 
    if (val_acc > 90.0):
        saveModel()

```
What type of ensembling is being utilized in this approach? Explain in detail.

    2. Name one method by which NN models may be combined to yield a single prediction.

---

22. 1. Referring to Fig. (6.10) which depicts a specific learning rate schedule, describe the basic notion behind its mechanism.

<table align='center'>
<tr>
<td align="center">
    <img src="img/nn_ensemble-6.png" alt= "A typical binary classification problem" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> A typical binary classification problem </td>
</tr>
</table>

    2. Explain how cyclic learning rates can be effective for the training of convolutional neural networks such as the ones in the code snippet of Fig. 6.10.

    3. Explain how a cyclic cosine annealing schedule as proposed by Loshchilov [10] and [13] is used to converge to multiple local minima.

---

13. Two popular algorithms for winning Kaggle solutions are Light GBM and XGBoost. They are both gradient boosting algorithms.
    1. What is gradient boosting?
    1. What problems is gradient boosting good for?

