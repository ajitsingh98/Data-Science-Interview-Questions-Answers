# Data Science Interview Questions And Answers


## Probabilistic Modeling

Contents
---
- [Maximum Likelihood Estimation(MLE)](#maximum-likelihood-estimation)
- [Maximum A Posteriori(MAP)](#maximum-a-posteriori)
- [Naive Bayes](#naive-bayes)
- [Logistic Regression](#logistic-regression)

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

Therefore, the MLE of \(\lambda\) is:

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


Q. MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori)
1. How do MPE and MAP differ?
1. Give an example of when they would produce different results.

<details><summary><b>Answer</b></summary>
    
</details>

---

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



</details>

---

Q. How does the Naive Bayes algorithm work?

<details><summary><b>Answer</b></summary>



</details>

---


Q. What is Laplace smoothing (additive smoothing) in Naive Bayes?

<details><summary><b>Answer</b></summary>



</details>

---


Q. Can Naive Bayes handle continuous and categorical features?

<details><summary><b>Answer</b></summary>



</details>

---


Q. What are the advantages of using Naive Bayes?

<details><summary><b>Answer</b></summary>



</details>

---


Q. Can Naive Bayes handle missing data?

<details><summary><b>Answer</b></summary>



</details>

---


Q. How do you evaluate the performance of a Naive Bayes classifier?

<details><summary><b>Answer</b></summary>



</details>

---

Q. What is the difference between Naive Bayes and other classification algorithms like Logistic Regression or Decision Trees?


<details><summary><b>Answer</b></summary>



</details>

---

