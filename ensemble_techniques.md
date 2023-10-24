# Deep Learning Interview Questions


Topics
---

- [Ensemble Techniques]()


## Ensemble Techniques

Contents
----

- Bagging, Boosting and Stacking 
- Approaches for Combining Predictors
- Monolithic and Heterogeneous Ensembling
- Ensemble Learning
- Snapshot Ensembling 
- Multi-model Ensembling
- Learning-rate Schedules in Ensembling

---

1. Mark all the approaches which can be utilized to boost a single model performance:
    1. Majority Voting
    2. Using K-identical base-learning algorithms
    3. Using K-different base-learning algorithms 
    4. Using K-different data-folds
    5. Using K-different random number seeds 
    6. A combination of all the above approaches

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

9. **True or False**: A perfect ensemble comprises of highly correct classifiers that differ as
much as possible.

---

10. **True or false**: In bagging, we re-sample the training corpus with replacement and there-
fore this may lead to some instances being represented numerous times while other instances not to be represented at all.

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