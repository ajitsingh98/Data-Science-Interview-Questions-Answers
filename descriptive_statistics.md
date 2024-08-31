# Data Science Interview Questions And Answers

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

Q. Given the array $[1,5,3,2,4,4]$, find its mean, median, variance, and standard deviation.

<details><summary><b>Answer</b></summary>

Given:

$$arr = [1, 5, 3, 2, 4, 4]$$

Sort the above series in ascending order:

$$arr = [1, 2, 3, 4, 4, 5]$$

<b>Mean($\bar{x}$)</b>

$$\bar{x}= \frac{\sum_{i=1}^{n}arr}{n}$$

Here number of elements(n) = 6

$$\bar{x} = \frac{\sum_{i=1}^{6}arr}{6} = \frac{1+2+3+4+4+5}{6} = 3.166$$

<b>Median($M$)</b>

Since $n$ is even, we can use following expression:

$$\text{M} = \frac{x_{\frac{n}{2}} + x_{\frac{n}{2}+1}}{2}$$
$$\text{M} = \frac{3+4}{2} = 3.5$$

<b>Variance($\sigma^2$)</b>

$$\text{\sigma^2} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$
$$\text{\sigma^2} = \frac{1}{6} \sum_{i=1}^{n} (x_i - 3.166)^2$$
$$\text{\sigma^2} ≈ 1.47$$

<b>Standard deviation($\sigma$)</b>

$$\text{\sigma} = \sqrt{variance} = \sqrt{1.47}$$

$$\text{\sigma} = 1.2124$$

</details>

---

Q. When should we use median instead of mean? When should we use mean instead of median?

<details><summary><b>Answer</b></summary>

Which average should be used as a measure of center depnds on the distribution of the observations.

Distribution type:

- Normal - If the distribution normal bell shaped curve then it makes sense to use mean as center of the distribution. 

- Skewed - In case of skewed distribution(right ot left), Median should be used to measure the center of the distribution.

</details>

