# Data Science Interview Questions And Answers


## Probabilistic Modeling

Contents
---
- [Maximum Likelihood Estimation(MLE)](#maximum-likelihood-estimation)
- [Maximum A Posteriori(MAP)](#maximum-a-posteriori)
- [Naive Bayes](#naive-bayes)
- [Logistic Regression](#logistic-regression)
- [Bayesian Belief Networks](#bayesian-belief-networks)

---

## Maximum Likelihood Estimation

Q. Explain frequentist vs. Bayesian statistics.

<details><summary><b>Answer</b></summary>

Frequentist and Bayesian statistics represents two different school of thoughts while building probabilistic model from the data.

We can understand both approaches from an example:

Suppose you have a coin, and you want to determine the probability of it landing heads (H) when you toss it.

<b>Frequentist Approach:</b>

In the frequentist approach, probabilities are viewed as long-term relative frequencies based on repeated, identical experiments or events. To find the probability of getting a heads (H), you perform a large number of coin flips and calculate the proportion of times it lands heads.

1. You flip the coin 100 times.
2. It lands heads (H) 53 times.
3. The frequentist probability of getting heads is calculated as the relative frequency:

Probability of H = (Number of H outcomes) / (Total number of outcomes) = $\frac{53}{100}$ = $0.53$.

In the frequentist approach, probability is objective and based on observable data from repeated experiments.

<b>Bayesian Approach:</b>

In the Bayesian approach, probability is a measure of our uncertainty or belief in an event. You start with a prior belief (prior probability) about the probability of getting heads, and you update that belief with new evidence (likelihood) from your observations.

1. You have a prior belief that the probability of getting heads is uniformly distributed between 0 and 1, i.e., a $Beta(1, 1)$ distribution.

   Prior Probability: $Beta(1, 1)$

2. You flip the coin 10 times, and it lands heads (H) 6 times and tails (T) 4 times.

   Likelihood: $Binomial(10, 0.5)$

3. You update your prior belief using Bayes' theorem:

   Posterior Probability = (Prior Probability * Likelihood) / Evidence

   Posterior Probability: $Beta(1 + 6, 1 + 4) = Beta(7, 5)$

In the Bayesian approach, you use your prior belief and update it with observed evidence to obtain a posterior probability distribution. This posterior distribution represents your updated belief in the probability of getting heads.

Key Differences:

- Frequentist approach treats probability as a relative frequency based on data.
- Bayesian approach treats probability as a measure of belief and updates it using Bayes' theorem.
- Frequentist probabilities are fixed and objective.
- Bayesian probabilities are subjective and represent your current knowledge or belief.

</details>

---

Q. How can we estimate the parameters of a given probability distribution?

<details><summary><b>Answer</b></summary>

We can use following methods to estimates parameters:

- Maximum Likelihood Estimation(MLE)
- Maximum A Posteiori(MAP)

</details>

---

Q. What is the main assumption of MAP and MLE?

<details><summary><b>Answer</b></summary>

MLE/MAP both assumes the data are independent and identically distributed(iid)

</details>

---

Q. How is likelihood different than probability?

<details><summary><b>Answer</b></summary>

In the case of discrete distributions, likelihood is a synonym for the probability mass, or joint probability mass, of the data. In the case of continuous distribution, likelihood refers to the probability density of the data distribution.

</details>

---

Q. Write the mathematical expression of likelihood?

<details><summary><b>Answer</b></summary>

It represents the probability of observing the given data as a function of the parameters of the statistical model.

For a random variable $X$ with probability density function (PDF) $f(X; \theta)$ or probability mass function (PMF) in the discrete case, where $\theta$ represents the parameters of the model, the likelihood of observing a specific dataset $\{x_1, x_2, \ldots, x_n\}$ is given by:

$$
L(\theta; x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} f(x_i; \theta)
$$

Since we assumed that each data point is independent, the likelihood of all of our data is the product of the likelihood of each data point.

</details>

---

Q. What does *Argmax* mean?

<details><summary><b>Answer</b></summary>

Argmax is short for Arguments for the maxima. The argmax of a function is the value of the domain at which the function is maximized.

</details>

---

Q. Describe how to analytically find the MLE of a likelihood function?

<details><summary><b>Answer</b></summary>

To analytically find the Maximum Likelihood Estimator (MLE) of a likelihood function, we can follow below steps:


<table align='center'>
<tr>
<td align="center">
    <img src="img/MLE_Calculations.png" alt= "MLE calculations steps" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center"> MLE Estimation Steps </td>
</tr>
</table>

*Define the likelihood function*

Suppose we have a set of independent and identical distributed observations $X_1, X_2, ...,X_n$ from a probability distribution with a parameter $\theta$.

$$
L(\theta) = \prod_{i=1}^{n} f(X_i \mid \theta)
$$

- Here $f(X_i \mid \theta)$ is the pdf or pmf of the data given the parameter $\theta$

*Take the log likelihood*

To simplify the $L(\theta)$, take natural log on boh side:

$$
l(\theta) = log(L(\theta)) = \sum_{i=1}^{n}\log f(X_i \mid \theta)
$$

*Take the derivative wrt $\theta$*

$$
\frac{d\ell(\theta)}{d\theta}
$$

*Set the Derivative Equal to Zero*

To find the critical points set:

$$
\frac{d\ell(\theta)}{d\theta} = 0
$$

*Solve $\theta$*

Find the values of $\theta$ that maximize likelihood function. These values are potential MLEs, representing the parameter estimates that maximize the likelihood of observing the given data.

*Verify the Maximum (Second Derivative Test)*

$$
\frac{d^2\ell(\theta)}{d\theta^2}
$$

- If the second derivative is negative at the critical point, it confirms a local maximum.

</details>


---

Q. What is the term used to describe the first derivative of the log-likelihood function?

<details><summary><b>Answer</b></summary>

Score function : The score function measures the sensitivity of the log-likelihood function to changes in the parameter $\theta$. It is the gradient (or derivative) of the log-likelihood with respect to the parameter.

</details>


---

Q. What is the relationship between the likelihood function and the log-likelihood function?

<details><summary><b>Answer</b></summary>

The log-likelihood function is derived by taking the natural logarithm of the likelihood function.

$$
l(\theta) = \log{L(\theta)}
$$

- Likelihood: $L(\theta)$
- Log-Likelihood: $\ell(\theta)$

</details>

---


Q. What is likelihood function of the independent identically distributed (i.i.d) random variables:
$X_1,··· ,X_n$ where $X_i ∼ binomial(n, p)$, $∀i ∈ [1,n]$, and where p is the parameter of interest?


<details><summary><b>Answer</b></summary>

Likelihood function in case of discrete random variables is jus the PMF. 

For Binomial distribution:

$$
P(X_i = x_i) = \binom{n}{x_i} p^{x_i} (1 - p)^{n - x_i} \quad \text{PMF}
$$

Since the observations are i.i.d., the likelihood function is the product of the individual PMFs:

$$
L(p) = \prod_{i=1}^{n} P(X_i = x_i) = \prod_{i=1}^{n} \binom{n}{x_i} p^{x_i} (1 - p)^{n - x_i}.
$$

</details>


---

Q. How can we derive the maximum likelihood estimator (MLE) of the i.i.d samples $X_1, · · · , X_n$ introduced in above question?

<details><summary><b>Answer</b></summary>

Likelihood function in case of binomial distribution:

$$
L(p) = \prod_{i=1}^{n} P(X_i = x_i) = \prod_{i=1}^{n} \binom{n}{x_i} p^{x_i} (1 - p)^{n - x_i}.
$$

Log likelihood:

$$
LL(p) = \log{\binom{n}{x}} + x \log{p} + (n-x) \log{1-p} \quad text{(Log Likelihood)}
$$

On taking derivative wrt $p$

$$
\frac{dL(p)}{dp} = 0 + \frac{x}{p} - \frac{(n-x)}{1-p}
$$

$$
\frac{dL(p)}{dp} = \frac{x-pn}{p(1-p)}
$$

For maximizing the likelihood:

$$
\frac{dL(p)}{dp} = 0
$$

$$
p = \frac{x}{n}
$$


</details>

---

Q. Derive the maximum likelihood estimator of an exponential distribution.

<details><summary><b>Answer</b></summary>

*PDF of exponential distribution:*

$$
f(x \mid \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0.
$$

*Likelihood Function:*

For $n$ i.i.d. observations $ X_1, X_2, \ldots, X_n $, the likelihood function is the product of the individual densities:

$$
L(\lambda) = \prod_{i=1}^{n} f(X_i \mid \lambda) = \prod_{i=1}^{n} \lambda e^{-\lambda X_i}.
$$

Simplifying this expression:

$$
L(\lambda) = \lambda^n e^{-\lambda \sum_{i=1}^{n} X_i}.
$$

*Log-Likelihood Function:*

To make the maximization easier, take the natural logarithm of the likelihood function to get the log-likelihood function:

$$
\ell(\lambda) = \log L(\lambda) = \log(\lambda^n) + \log\left(e^{-\lambda \sum_{i=1}^{n} X_i}\right).
$$

Simplify:

$$
\ell(\lambda) = n \log(\lambda) - \lambda \sum_{i=1}^{n} X_i.
$$

*Differentiate the Log-Likelihood Function:*

Differentiate $\ell(\lambda)$ with respect to $\lambda$:

$$
\frac{d\ell(\lambda)}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^{n} X_i.
$$

*Set the Derivative Equal to Zero:*

Set the first derivative to zero to find the critical points:

$$
\frac{n}{\lambda} - \sum_{i=1}^{n} X_i = 0.
$$

*Solve for $\lambda$:*

Rearrange the equation to solve for $\lambda$:

$$
\frac{n}{\lambda} = \sum_{i=1}^{n} X_i.
$$

Therefore, the MLE of $\lambda$ is:

$$
\hat{\lambda} = \frac{n}{\sum_{i=1}^{n} X_i} = \frac{1}{\bar{X}},
$$

</details>

---

Q. A lot of machine learning models aim to approximate probability distributions. Let’s say P is the distribution of the data and Q is the distribution learned by our model. How do measure how close Q is to P?

<details><summary><b>Answer</b></summary>

We can use KL Divergence formula which is a measure of how one probability distribution $Q$ diverges from a second, expected probability distribution $P$.

$$
D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} \quad \text{(for discrete distributions)}
$$

$$
D_{KL}(P \| Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx \quad \text{(for continuous distributions)}
$$

</details>

---

## Maximum A Posteriori

Q. What is MAP? How is it different than MLE?

<details><summary><b>Answer</b></summary>

MAP estimation finds the parameter values that maximize the posterior distribution of the parameters given the data, inducing prior beliefs about the parameters.

$$
 \hat{\theta}_{\text{MAP}} = \arg\max_{\theta} P(\theta | X) = \arg\max_{\theta} \frac{P(X | \theta) P(\theta)}{P(X)}.
$$

Since $P(X)$ is constant with respect to $\theta$, it simplifies to:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} P(X | \theta) P(\theta)
$$

- MAP induces priori knowledge about the parameters through a prior distribution where as MLE does not consider any prior information.
- In MLE, parameters are treated as fixed values, while in MAP, they are treated as random variables with a prior distribution, requiring an extra assumption about the prior.

</details>

---

Q. When to use MAP over MLE?

<details><summary><b>Answer</b></summary>

If prior probability is provided in the problem setup, that information should be used (i.e., apply MAP). However, if no prior information is given or assumed, MAP cannot be used, and MLE becomes a suitable approach.


</details>

---

Q. When do MAP and MLE yield similar parameter estimates?

<details><summary><b>Answer</b></summary>

MAP and MLE will yield similar parameter estimates in following situations:

- Uniform Prior : When prior assign equal probabilities to all parameter values, adding no additional information
- Non-informative Priors : Priors that are weakly informative (e.g., with very high variance) have little impact on the posterior,
- Data Size is large : With a large amount of data, the likelihood dominates the posterior, reducing the influence of the prior


</details>

---

Q. 
1. Define the term conjugate prior.
2. Define the term non-informative prior.


<details><summary><b>Answer</b></summary>

*Conjugate Prior*

A conjugate prior is a probability distribution that, when combined with the likelihood and normalized, results in a posterior distribution that belongs to the same family as the prior.

$$
p(\theta | x) = \frac{p(x|\theta)p(\theta)}{p(x)}
$$

The prior $p(\theta)$ is conjugate to the posterior $p(\theta | x)$ if both are in same family of distributions.

*Non-Informative Prior*



</details>

---

Q. MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori)
1. How do MPE and MAP differ?
1. Give an example of when they would produce different results.

<details><summary><b>Answer</b></summary>
    
</details>

---

## Naive Bayes

Q. Naive Bayes classifier.
1. How is Naive Bayes classifier naive?
1. Let’s try to construct a Naive Bayes classifier to classify whether a tweet has a positive or negative sentiment. We have four training samples:

$$
\begin{bmatrix} 
    \text{Tweet} &  \text{Label} \\\\
    \text{This makes me so upset} & \text{Negative}\\\\
    \text{This puppy makes me happy} & \text{Positive} \\\\
    \text{Look at this happy hamster} & \text{Positive} \\\\
    \text{No hamsters allowed in my house} & \text{Negative}
\end{bmatrix}
$$

According to your classifier, what's sentiment of the sentence The hamster is upset with the puppy?

<details><summary><b>Answer</b></summary>

1. The Naive Bayes classifier is considered "naive" because it makes a strong and often unrealistic assumption: it assumes that all features (or predictors) in the dataset are independent of each other given the class label.

2. 

</details>

---

Q. Is Naive bayes a discriminative model? 

<details><summary><b>Answer</b></summary>

*True*

The Naive Bayes algorithm is generative. It models $P(x|y)$ and makes explicit assumptions on its distribution (e.g. multinomial, categorical, Gaussian, ...). The parameters of this distributions are estimated with MLE or MAP. 

</details>

---

Q. How does the Naive Bayes algorithm work?

<details><summary><b>Answer</b></summary>

*Naive Bayes Assumption*

It assumes that each feature $x$ is independent of one another give $y$

*Training Phase*

In this phase we do parameter estimations. In core Naive Bayes uses Bayes theorem.

$$
 P(\text{Class} | \text{Features}) = \frac{P(\text{Features} | \text{Class}) \cdot P(\text{Class})}{P(\text{Features})}
$$

- $P(\text{Features} | \text{Class})$: Likelihood of the features given the class.
- $P(\text{Class})$: Prior probability of the class.
- $P(\text{Features})$: Evidence, the overall probability of the features.

Using Naive Bayes Assumption

$$
P(\text{Features} | \text{Class}) = P(\text{Feature}_1 | \text{Class}) \times P(\text{Feature}_2 | \text{Class}) \times \ldots \times P(\text{Feature}_n | \text{Class})
$$

Here we can calculate all the terms of Bayes theorem:

- Prior Probability: $P(\text{Class})$: This is usually estimated from the training data by calculating the frequency of each class.
- Likelihood: $P(\text{Feature}_i | \text{Class})$: Estimated from the training data by counting how often each feature value appears within each class.
- Evidence: $P(\text{Features})$: This term is often omitted during classification since it's the same for all classes and does not affect the ranking of probabilities.

*Predictions*

- For a given set of feature values, the classifier computes the posterior probability for each class.
- The class with the highest posterior probability is chosen as the predicted class.

</details>

---


Q. Why is Naive Bayes still used despite its flawed assumption of feature independence?

<details><summary><b>Answer</b></summary>

Naive Bayes is beneficial primarily because of its "naive" assumption of feature independence, which, although technically incorrect, offers some practical advantages:

- Scalability: Handles large feature spaces efficiently. It scales linearly with the number of features
- Simplicity: Easy to implement and interpret.
- High-Dimensional Performance: Performs well in high-dimensional datasets.
- Robustness: Yields good results in many practical applications.

</details>

---


Q. What is Laplace smoothing (additive smoothing) in Naive Bayes?

<details><summary><b>Answer</b></summary>

Laplace smoothing, also known as additive smoothing, is a technique used in Naive Bayes to handle zero probabilities that occur when a feature (e.g., a word in text classification) does not appear in the training data for a given class. Without smoothing, if a word never appears in a class during training, its probability would be zero, which could incorrectly influence the final prediction.



</details>

---

Q. Can Naive Bayes handle continuous and categorical features?

<details><summary><b>Answer</b></summary>

Yeah, We can handle both categorical and continuous features both using Naive Bayes

- Categorical Features : can be handled with methods like multinomial and bernoulli distributions 
- Continuous Features : Can be handled using Gaussian assumptions
- Mixed Data : We can either convert continuous values into bins(categorization) and treat it as only categorical features or, we can fit separate model on categorical and numeric data and then combine to make prediction 

</details>

---

Q. Can Naive Bayes handle missing data?

<details><summary><b>Answer</b></summary>

Naive Bayes does not directly handle missing data, but several practical strategies, such as ignoring missing features, imputing missing values, or creating indicator variables, can be employed to manage it effectively.

</details>

---

Q. What is the difference between Naive Bayes and other classification algorithms like Logistic Regression or Decision Trees?


<details><summary><b>Answer</b></summary>



</details>

---

## Logistic Regression

Q. Define logistic regression?

<details><summary><b>Answer</b></summary>

Logistic Regression is a discriminative classifier that works by trying to learn a function that approximates $P(y|x)$. 

</details>

---

Q. What is the main assumption of logistic regression?

<details><summary><b>Answer</b></summary>

The central assumption that $P(y|x)$ can be approximated as a sigmoid function function applied to a linear combination of input features.

$$
P(Y=1 | X) = \frac{1}{1+\exp(-w_0 - \sum_i w^i X^i)}
$$

- Logistic function applied to a linear function of the data.

</details>

---

Q. Write the expression of sigmoid or logistic function?

<details><summary><b>Answer</b></summary>

$$
\sigma(z) = \frac{1}{1+\exp(-z)}
$$

</details>

---

Q. Prove that logistic regression is a linear classifier?

<details><summary><b>Answer</b></summary>

At the decision boundary:

$$
P(Y=1|X) = \frac{1}{2}
$$

We can express this as:

$$
P(Y=1|X) = \frac{1}{1 + \exp(-w_0 - \sum_i w_i X_i)} = \frac{1}{2}
$$

Solving this equation gives:

$$
\exp(-w_0 - \sum_i w_i X_i) = 1
$$

This occurs only if:

$$
-w_0 - \sum_i w_i X_i = 0
$$

This equation defines the decision boundary of logistic regression. Since it represents a straight line, logistic regression is classified as a linear classifier.

</details>

---

Q. Does closed-form solution exists for logistic regression?

<details><summary><b>Answer</b></summary>

No closed-form solution exist. That's why we use gradient descent to estimate the parameters.

</details>

---

Q. How can we learn the parameters of logistic regression model? 

<details><summary><b>Answer</b></summary>

</details>

---

Q. State the difference between Naive bayes and Logistic regression model?

<details><summary><b>Answer</b></summary>


</details>

---

Q. What is the range of logistic(sigmoid function)?

<details><summary><b>Answer</b></summary>

$(0, 1)$

</details>

---

Q. What is the difference between Conditional MLE and standard MLE, and how does it relate to logistic regression?

<details><summary><b>Answer</b></summary>

Conditional Maximum Likelihood Estimation (Conditional MLE) refers to MLE applied within a conditional model, where the parameters only influence the conditional probability $P(Y|X)$ and not the marginal probability $P(X)$. In contrast, standard MLE applies when the parameters affect both $P(Y|X)$ and $P(X)$.

Logistic regression is an example of a conditional model because the parameters $\theta$ only control $P(Y|X)$, not $P(X)$. As a result, the MLE used in logistic regression is considered a Conditional MLE.


</details>

---

Q. What is the issue with using squared losses(MSE) or absolute losses(MAE) for logistic regression model?

<details><summary><b>Answer</b></summary>

Squared and absolute losses are not typically used in logistic regression because they are not well-suited to the characteristics of the logistic function and can lead to significant issues during optimization.

- Non-Convexity Issues: This non-convexity introduces multiple local minima, complicating optimization and often leading to suboptimal solutions.
- Gradient Behavior: Squared loss flattens gradients, especially when predictions are close to 0 or 1, which is common in logistic regression. The derivative of the squared loss is proportional to the error $(y_i - \hat{y}_i)$. In logistic regression, where $\hat{y}_i$ is bounded between 0 and 1, the gradients can become very small, slowing down convergence. 
- Poor Fit for Probabilistic Outputs : More fitted for regression task

</details>

---

Q. Can we use logistic regression for multiclass classification problem?

<details><summary><b>Answer</b></summary>

Yes, logistic regression can be extended to handle multiclass classification problems through approaches like One-vs-Rest (OvR) and Softmax Regression (Multinomial Logistic Regression).

</details>

---

Q. Write the expression of softmax function?

<details><summary><b>Answer</b></summary>

For a set of scores/logits $\mathbf{z} = [z_1, z_2, \ldots, z_K]$, the probability of class $j$ is given by:

$$
 P(y = j | \mathbf{x}) = \frac{\exp(z_j)}{\sum_{k=1}^{K} \exp(z_k)}
$$


</details>

---

Q. State one issue with softmax function over sigmoid?

<details><summary><b>Answer</b></summary>

Computationally more intensive compared to binary logistic regression, especially when the number of classes is large.

</details>

---

Q. How is Maximum Likelihood Estimation (MLE) used in logistic regression, and why is it preferred over other estimation methods like least squares?

<details><summary><b>Answer</b></summary>

*Logistic Regression Model*

Logistic regression models the probability that a binary outcome $y$ is 1 given an input vector $\mathbf{x}$. The model is defined as:

$$
P(y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
$$

*Likelihood Function*

For a dataset with $n$ observations, the likelihood of the observed data given the parameters $\mathbf{w}$ and $b$ is:

$$
L(\mathbf{w}, b) = \prod_{i=1}^{n} P(y_i | \mathbf{x}_i)
$$

Since logistic regression deals with binary outcomes, this can be rewritten as:

$$
L(\mathbf{w}, b) = \prod_{i=1}^{n} \left(\frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}\right)^{y_i} \left(\frac{e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}{1 + e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}\right)^{1 - y_i}
$$

*Log-Likelihood Function*

$$
\text{Log-Likelihood} = \sum_{i=1}^{n} \left( y_i \log P(y_i | \mathbf{x}_i) + (1 - y_i) \log (1 - P(y_i | \mathbf{x}_i)) \right)
$$

*Maximizing the Log-Likelihood*

MLE estimates the parameters $w$ and $b$ by finding values that maximize the log-likelihood function. This is typically done using numerical optimization techniques like gradient descent or Newton-Raphson methods.

MLE is Preferred Over Least Squares in Logistic Regression?

- *Appropriate Loss Function*
- *Convex Optimization*: The optimization problem derived from MLE is convex, meaning it has a single global minimum, which guarantees the stability and reliability of the solution.

</details>

---

Q. What is Maximum A Posteriori (MAP) Estimation in logistic regression?

<details><summary><b>Answer</b></summary>

MAP estimation in logistic regression is a Bayesian approach that estimates model parameters by maximizing the posterior probability, which combines the likelihood of the observed data with a prior distribution over the parameters. 

$$
\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_{\mathbf{w}} \, P(\mathbf{w} | \text{data}) = \arg\max_{\mathbf{w}} \, P(\text{data} | \mathbf{w}) \, P(\mathbf{w})
$$

Using Bayes' theorem, this becomes:

$$
\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_{\mathbf{w}} \, \left(\prod_{i=1}^{n} P(y_i | \mathbf{x}_i; \mathbf{w})\right) P(\mathbf{w})
$$

</details>

---

Q. How does MAP differ from Maximum Likelihood Estimation (MLE) in logistic regression?

<details><summary><b>Answer</b></summary>

MLE maximizes the likelihood of the data given the parameters, relying solely on observed data. MAP, on the other hand, maximizes the posterior probability by incorporating a prior distribution, which acts as a regularization term.

</details>

---

Q. What role do priors play in MAP estimation?

<details><summary><b>Answer</b></summary>

Priors in MAP estimation incorporate external knowledge or beliefs about the parameters, adding a regularization effect. Common priors include Gaussian (L2 regularization) and Laplace (L1 regularization), which help control model complexity and prevent overfitting.

</details>

---

Q. Why might MAP be preferred over MLE in logistic regression?

<details><summary><b>Answer</b></summary>

MAP is preferred over MLE in scenarios where there is a risk of overfitting, when data is sparse, or when domain knowledge is important. The inclusion of priors in MAP acts as regularization, making the model more robust to noise and improving generalization.

</details>

---

Q. How does MAP help in small datasets compared to MLE?

<details><summary><b>Answer</b></summary>

In small datasets, MLE may overfit because it relies only on the observed data. MAP’s use of priors helps stabilize parameter estimates, providing more reliable results when the data alone is insufficient.

</details>

---

Q. What type of priors are commonly used in MAP for logistic regression?

<details><summary><b>Answer</b></summary>

Common priors used in MAP for logistic regression include:

- Gaussian Prior (L2 Regularization): Penalizes large weights and prevents overfitting.
- Laplace Prior (L1 Regularization): Encourages sparsity, leading to simpler models by driving some coefficients to zero.

</details>

---

Q. How does MAP provide flexibility compared to MLE?

<details><summary><b>Answer</b></summary>

MAP allows the use of different priors based on the problem context, providing flexibility in how the model is regularized or adjusted. MLE lacks this capability as it does not incorporate any prior information.

</details>

---

Q. What is the main advantage of using MAP in logistic regression?

<details><summary><b>Answer</b></summary>

The main advantage of using MAP in logistic regression is its ability to combine observed data with prior information, enhancing the model’s robustness against overfitting and making it better suited for small or noisy datasets.

</details>

---


Q. Can you explain a situation where using MAP estimation could lead to worse results than MLE?

<details><summary><b>Answer</b></summary>

MAP estimation could lead to worse results if the prior is incorrect or misaligned with the actual data distribution. For example, if a strong prior incorrectly penalizes certain parameter values, the resulting estimates could be biased, leading to poor predictive performance compared to MLE, which only relies on the observed data.

</details>

---


## Bayesian Belief Networks