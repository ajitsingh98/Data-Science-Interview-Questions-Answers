## Descriptive Statistics

1. Explain frequentist vs. Bayesian statistics.

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

2. Given the array $[1,5,3,2,4,4]$, find its mean, median, variance, and standard deviation.

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

3. When should we use median instead of mean? When should we use mean instead of median?

<details><summary><b>Answer</b></summary>

Which average should be used as a measure of center depnds on the distribution of the observations.

Distribution type:

- Normal - If the distribution normal bell shaped curve then it makes sense to use mean as center of the distribution. 

- Skewed - In case of skewed distribution(right ot left), Median should be used to measure the center of the distribution.

</details>


---

4. What is a moment of function? Explain the meanings of the zeroth to fourth moments.


<details><summary><b>Answer</b></summary>

Statistical moments are additional descriptors of a curve/distribution. Moments quantify three parameters of distributions: location, shape, and scale. 

- `location` -  A distribution’s location refers to where its center of mass is along the x-axis. 
- `Scale` -  The scale refers to how spread out a distribution is. Scale stretches or compresses a distribution along the x-axis.
- `Shape` - The shape of a distribution refers to its overall geometry: is the distribution bimodal, asymmetric, heavy-tailed?

The $k$th moment of a function $f(x)$ about a non-random value $c$ is:

$$E[(X - c)^k] = \int_{-\infty}^{\infty} (x - c)^k f(x) dx$$

This generalization allows us to make an important distinction: 
- a raw moment is a moment about the origin $(c=0)$
- a central moment is a moment about the distribution’s mean $(c=E[X])$

First five moments in order from $0$th to $4$th moments: `total mass`, `mean`, `variance`, `skewness`, and `kurtosis`. 

- <b>Zeroth Moment(total mass)</b>: The zeroth moment is simply the constant value of 1. It doesn't provide much information about the distribution itself but is often used in mathematical contexts.

- <b>1st Moment(mean)</b> - The first moment is also known as the mean or expected value. It represents the center of the distribution and is a measure of the average or central location of the data points. 

$$\(\mu = \frac{1}{n} \sum_{i=1}^{n} x_i\)$$

Where:
- $\(\mu\)$ (mu) is the mean.
- $\(n\)$ is the number of data points.
- $\(x_i\)$ represents individual data points.

- <b>2nd Moment(Variance)</b> - The second moment is the variance. It measures the spread or dispersion of the data points around the mean. It is calculated as the average of the squared differences between each data point and the mean. 

$$\(\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2\)$$

Where:
  - \(\sigma^2\) (sigma squared) is the variance.

- <b>3rd Moment(Skewness)</b> - The third moment is a measure of the skewness of the distribution. It indicates whether the distribution is skewed to the left (negatively skewed) or to the right (positively skewed). 

$$\[Skewness = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \mu}{\sigma}\right)^3\]$$

- <b>4th Moment(Kurtosis)</b> - The fourth moment measures the kurtosis of the distribution. Kurtosis indicates whether the distribution is more or less peaked (leptokurtic or platykurtic) compared to a normal distribution. 

$$\[Kurtosis = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \mu}{\sigma}\right)^4\]$$

</details>

---

5. Are independence and zero covariance the same? Give a counterexample if not.

<details><summary><b>Answer</b></summary>

<b>No</b> 

Independence and zero covariance are related but not the same. Independence implies zero covariance, but the reverse is not necessarily true. Zero covariance indicates that there is no linear relationship between the variables. However, it does not necessarily mean that the variables are independent. Non-linear relationships can still exist between variables even if their covariance is zero.

Lets explain this with an example:

Consider two random variables $X$ and $Y$ defined as follows:

- A random variable $𝑋$ with $𝐸[𝑋]=0$ and $𝐸[𝑋^3]=0$, e.g. normal random variable with zero mean. 
- Take $𝑌=𝑋^2$.

Now it is clear the $X$ and $Y$ are dependent, now lets look the covariance of both

$$𝐶𝑜𝑣(𝑋,𝑌) = 𝐸[𝑋𝑌]−𝐸[𝑋]⋅𝐸[𝑌]$$

$$𝐶𝑜𝑣(𝑋,𝑌) = 𝐸[𝑋.X^2]−𝐸[𝑋]⋅𝐸[X^2]$$

$$𝐶𝑜𝑣(𝑋,𝑌) = 0$$

Now $Cov(X, Y)$ coming as zero and hence depicting the $X$ and $Y$ are independent which is not the case.

</details>

---

6. Suppose that you take $100$ random newborn puppies and determine that the average weight is $1$ pound with the population standard deviation of $0.12$ pounds. Assuming the weight of newborn puppies follows a normal distribution, calculate the $95\\%$ confidence interval for the average weight of all newborn puppies.

<details><summary><b>Answer</b></summary>

Given:

$$n = 100$$
$$\hat{x} = 1$$
$$\sigma = 0.12$$

$95\\%$ Confidence interval for $\mu_population$ can be given by following expression:

$\hat{x} \pm z_{0.95} \cdot \frac{\sigma}{\sqrt{n}}$

On substituting all the values, we get:

$1 \pm 1.96 \cdot \frac{0.12}{\sqrt{100}}$ = $(0.9768,1.0232)$

So, the $95\%$ interval is $(0.9768,1.0232)$.

</details>

---

7. Suppose that we examine $100$ newborn puppies and the $95%$ confidence interval for their average weight is $[0.9, 1.1]$ pounds. Which of the following statements is true?
    1. Given a random newborn puppy, its weight has a $95%$ chance of being between $0.9$ and $1.1$ pounds.
    1. If we examine another $100$ newborn puppies, their mean has a $95%$ chance of being in that interval.
    1. We're $95\\%$ confident that this interval captured the true mean weight.



<details><summary><b>Answer</b></summary>

$3rd$ statement seems correct interpretation of CI. $95\\%$ CI represents if we draw multiple samples and caluclate the sample statistics and confidence intervals(CI) then $95\\%$ of those intervals will contains population mean($\mu$).

Lets look at other statements:

$1^{st}$ statement is incorrect because we don't use CI for estimating individual sample weight range.

$2^{nd}$ statment is talking about sample statistics but CI is mainly used to estimate the population parameter not sample statistics.

</details>

---

8. Suppose we have a random variable X supported on $[0,1]$  from which we can draw samples. How can we come up with an unbiased estimate of the median of X?



---

9. Can the correlation be greater than 1? Why or why not? How to interpret a correlation value of 0.3?

<details><summary><b>Answer</b></summary>

<b>No</b> correlation(r) can not be greater than 1 and range of r is $[-1, 1]$. 

Lets look at the expression of correlation(r) to establish the above statement.

Suppose we have two variables X and Y and we want to investigate the relationship between them.

$$correlation(r_{XY}) = \frac{covariance(X, Y)}{{\sigma_X}{\sigma_Y}}$$

Now let's define two vectors $\arrow{u}$ and $\arrow{v}$ where $\arrow{u} = [u_1, u_2, ..., u_n]$ and $\arrow{v} = [v_1, v_2, ..., v_n]$ where elements are the deviation from the mean.

$$\u_i = x_i - \hat{x} \quad \v_i = y_i - \hat{y}$$

We can now write the sample covariance and the sample variances in vector notation as:

$$Covariance(u, v) = \frac{1}{n-1}\sum_{i=1}^n u_i v_i = \frac{1}{n-1}u \cdot v$$

similary variance in X and Y can be expressed in vectorized form:

$$Var(X) = \frac{1}{n-1}\sum_{i=1}^n {u_i}^{2} = \frac{1}{n-1} \| \mathbf{u} \|^2$$
$$Var(Y) = \frac{1}{n-1}\sum_{i=1}^n {v_i}^{2} = \frac{1}{n-1} \| \mathbf{v} \|^2$$

Now we can write correlation expresion using vectors $u$ and $v$,

$$r_{XY} = \frac{Covariance(u, v)}{\sqrt{Var(X)}{Var(Y)}}$$

$$r_{XY} = \frac{\mathbf{u} \cdot \mathbf{v}}{\| \mathbf{u} \| \| \mathbf{v} \|}$$

From cosine rule, we get

$$r_{XY} = \cos\theta$$

Since $-1 <= \cos\theta <= 1$, $r_{XY} is always between -1 and 1$ 

For give $r = 0.3$, we can deduce following conclusions:

- Relationship is positive but week.
- Increasing one variable is resulting in increase in another variable too.

Note that the above conlusions are based on assumption that both variable are linearly dependent.

</details>

---

10. The weight of newborn puppies is roughly symmetric with a mean of 1 pound and a standard deviation of 0.12. Your favorite newborn puppy weighs 1.1 pounds.
    1. Calculate your puppy’s z-score (standard score).
    1. How much does your newborn puppy have to weigh to be in the top 10% in terms of weight?
    1. Suppose the weight of newborn puppies followed a skewed distribution. Would it still make sense to calculate z-scores?

<details><summary><b>Answer</b></summary>

Given:

$$\mu = 1$$
$$\sigma = 0.12$$

1. $$z-score = \frac{x_i - \mu}{\sigma} = \frac{1.1 - 1}{0.12} = 0.83$$

2. Let x be the weight that needed to be in top $10\\%$. For this condtion $z_score >= z_{0.9}$
Using lookup table we get $z_score >= 1.645$

We can write expression for z-score:

$$z-score = \frac{x - \mu}{\sigma}$$
$$\frac{x - 1}{0.12} >= 1.645$$

On solving for $x$,

$$x >= 1.197$$

So, puppy weight should be atleast $1.197$ in order to be in top $10\\%$ of the weight.

3. If weight of newborn puppies distribution is not normal and is skewed then it does not make any sense to use z-scores for any decision making process.

</details>

---

11. Tossing a coin fifteen times resulted in 10 heads and 5 tails. How would you analyze whether a coin is fair?

<details><summary><b>Answer</b></summary>

Given:

$$n = 15$$
$$p(head) = 10/15$$
$$p(tail) = 5/15$$

Lets state the hypothesis and then based on the evidence we have from the observations we can access them using some statistic.

- Null Hypothesis($H_0$) - Coin is fair i.e $p(head) = p(tail) = \frac{1}{2}$
- Alternate Hypothesis($H_A$) - Coin is not fair i.e $p(head) ≠ \frac{1}{2}$

Since the distribution is binomial in this case with $X ~ Binomail(n=10, p = 1/2)$

We can find test statistic(z-score) using following approaches:

- Under Normal distribution approximation
- Exact binomial distribution calculation

<b>Under Normal distribution approximation:</b>

Binomial distribution can be approaximated as normal distribution if the following condition statisfies:

- $np > 5$ and $nq > 5$

In this case we have $np = 7.5$ if $H_0$ is true so we can use normal distribution approximation with some error.

test statistics under $H_0$:

- Mean($\mu$)
$$\mu = E(x) = np = \frac{15}{2} = 7.5$$

- Standard deviation($\sigma$)
$$\sigma = \sqrt{np(1-p)} = \sqrt{3.75} = 1.936$$

- $Z_score = \frac{x - \mu}{\sigma}$

$$Z_score = \frac{10-7.5}{1.936} = 1.291$$

Now with significance level($\alpha$) = $5\\%$, We can find out $Z_critical$

$$Z_critical = Z_{0.95} = 1.96$$

Since $Z_score$ that we got from the experiment is less than $Z_critical$, we can not reject the $H_0$.

So, the coin seems fair from the given observations.

<b>Exact binomial distribution calculation:</b>

We can do two-sided test to rejects $𝐻_0$ when $𝑋$ is sufficiently far from the expected value $\mu = 7.5$ under $𝐻_0$. 

For observed value $x$, the $P-value$ is $𝑃(𝑋≤𝑥)+𝑃(𝑋≥𝑛−𝑥)$.

$$p-value = P(X ≤ 10) + P(X ≥ 5)$$

We have,

$$\[P(X \leq 10) = \sum_{x=0}^{10} \binom{15}{x} (0.5)^x (0.5)^{15-x}\]$$

and

$$\[P(X \geq 5) = 1 - P(X < 5) = 1 - \sum_{x=0}^{4} \binom{15}{x} (0.5)^x (0.5)^{15-x}\]$$

putting these values to get p-value.

$$p-value = 0.99 + 0.99 = 1.98$$

since p-value > $0.05$, We can not reject null hypothesis and we will reject the alternate hypothesis i.e coin is not fair because of not having enough evidence in the given data.

</details>

---

12. Statistical significance.
    1. How do you assess the statistical significance of a pattern whether it is a meaningful pattern or just by chance?
    1. What’s the distribution of p-values?
    1. Recently, a lot of scientists started a war against statistical significance. What do we need to keep in mind when using p-value and statistical significance?

<details><summary><b>Answer</b></summary>

1. We can assess the statistical significance of a pattern using hypothesis testing. We can conduct hypothesis testing by following below steps:
    - Formulate Null hypothesis($H_0$) and alternate hypothesis($H_A$) carefully
    - Collect data to calculate test statistics that measures the strength of the observed pattern.
    - Use this test statistic to calculate a p-value, which represents the probability of obtaining such results (or more extreme) if the null hypothesis were true.
    - If the p-value is very small (typically less than a chosen significance level, often 0.05), you reject the null hypothesis and conclude that the pattern is statistically significant. 
    - Otherwise, you fail to reject the null hypothesis, suggesting the pattern may be due to chance.

2. Distribution of p-value depends on under which hypothesis we are observing.
    - Under null hypothesis($H_0$) p-value distribution will be uniformaly distributed.
    - Under alternate hypothesis($H_a$) p-value distribution will be rightly skewed.

3. There are some limitations of p-value and statistical significance and should be used with some cautions:
   - P-values only provide information about the probability of obtaining results under the null hypothesis.
   - they do not tell you the magnitude of an effect or its practical significance.
   - A small p-value doesn't prove that a result is practically significant or that it has any real-world importance.
   - Statistical significance should be considered in the context of the study design, sample size, and the relevance of the result to the research question.
   - Always consider effect sizes, confidence intervals, and the domain-specific context of the research in addition to p-values.


</details>
---

13. Variable correlation.
    1. What happens to a regression model if two of their supposedly independent variables are strongly correlated?
    1. How do we test for independence between two categorical variables?
    1. How do we test for independence between two continuous variables?

<details><summary><b>Answer</b></summary>

1. If predictors of a regression model are highly correlated, we might have following issues:
    - It will make interpretibility harder, it will be harder to get individual feature impact on final outcome.
    - If two variables are highly correlated then it makes sense to drop either of them because too many predictors may increase model's complexity and hence may cause overfitting issue. 

2. We can use Chi-Squared test of independence to assess the relationship between two categorical variables.

Here are the steps to conduct the test:

<b>Step 1: Formulate Hypotheses:</b>

- <b>Null Hypothesis (H0):</b> There is no association or independence between the two categorical variables.
- <b>Alternative Hypothesis (H1):</b> There is an association between the two categorical variables.

<b>Step 2: Create a Contingency Table:</b>

Construct a contingency table (also known as a cross-tabulation or a two-way table) that summarizes the frequencies or counts of the combinations of the two categorical variables.

Example:

```plaintext
         | Category A | Category B | Total
-----------------------------------------
Group 1  |    n11     |    n12     |  n1.
Group 2  |    n21     |    n22     |  n2.
-----------------------------------------
Total    |    n.1     |    n.2     |   N
```

<b>Step 3: Calculate Expected Frequencies:</b>

Compute the expected frequencies for each cell in the contingency table under the assumption of independence. The formula for the expected frequency in a cell (e.g., n11) is:

$$\[E = \frac{n1. \cdot n.1}{N}\]$$

Repeat this calculation for all cells in the table.

<b>Step 4: Calculate the Chi-squared Statistic:</b>

Calculate the Chi-squared $(\(\chi^2\))$ statistic using the formula:

$\[\chi^2 = \sum \frac{(O - E)^2}{E}\]$

Where:
- $\(\chi^2\)$ is the Chi-squared statistic.
- $\(O\)$ is the observed frequency in each cell.
- $\(E\)$ is the expected frequency in each cell.
- The summation $(\(\sum\))$ is taken over all cells.

<b>Step 5: Determine Degrees of Freedom:</b>

Determine the degrees of freedom $(\(df\))$ for the Chi-squared test. For a contingency table, $\(df = (r - 1)(c - 1)\)$, where $\(r\)$ is the number of rows and $\(c\)$ is the number of columns.

<b>Step 6: Calculate the p-value:</b>

Using the Chi-squared statistic and the degrees of freedom, we can calculate the p-value associated with the Chi-squared distribution. This p-value represents the probability of observing a Chi-squared statistic as extreme as the one computed if the variables were independent.

<b>Step 7: Make a Decision:</b>

Compare the calculated p-value to a significance level (e.g., $\(\alpha = 0.05\)$) to decide whether to reject the null hypothesis:

- If the p-value is less than $\(\alpha\)$, we can reject the null hypothesis and conclude that there is a statistically significant association between the two categorical variables.
- If the p-value is greater than or equal to $\(\alpha\)$, we fail to reject the null hypothesis, indicating no significant association.

3. To test for independence between two continuous variables, we can use a correlation coefficient, such as the Pearson correlation coefficient ($r$) or the Spearman rank correlation coefficient ($ρ$). 

- <b>Pearson Correlation (r):</b> The Pearson correlation coefficient is used to measure linear relationships. It ranges from -1 (perfect negative linear relationship) to 1 (perfect positive linear relationship), with 0 indicating no linear relationship. 

Expression of $r$:

$$\[r = \frac{\sum((X - \bar{X})(Y - \bar{Y}))}{\sqrt{\sum(X - \bar{X})^2 \sum(Y - \bar{Y})^2}}\]$$

- <b>Spearman Rank Correlation (ρ):</b> The Spearman rank correlation coefficient is non parametric test used to measure monotonic (nonlinear) relationships.

</details>

---

14. A/B testing is a method of comparing two versions of a solution against each other to determine which one performs better. What are some of the pros and cons of A/B testing?



---

15. You want to test which of the two ad placements on your website is better. How many visitors and/or how many times each ad is clicked do we need so that we can be $95%$ sure that one placement is better?

---

16. Your company runs a social network whose revenue comes from showing ads in newsfeeds. To double revenue, your coworker suggests that you should just double the number of ads shown. Is that a good idea? How do you find out?

---

17. Imagine that you have the prices of $ 10,000 stocks over the last 24-month period and you only have the price at the end of each month, which means you have 24 price points for each stock. After calculating the correlations of $10,000 * 9,9992$ pairs of stock, you found a pair that has a correlation to be above 0.8.
    1. What’s the probability that this happens by chance?
    1. How to avoid this kind of accidental pattern?

---

18. How are sufficient statistics and the Information Bottleneck Principle used in machine learning?
