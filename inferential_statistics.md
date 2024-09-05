# Data Science Interview Questions And Answers


## Inferential Statistics

Contents
---

- [Introduction](#introduction)
- [Point Estimation](#point-estimation)
- [Interval Estimation](#interval-estimation)
- [Hypothesis Testing](#hypothesis-testing)
- [Inference for Relationship](#inference-for-relationship)

---

### Introduction

Q. What is statistical inference?

<details><summary><b>Answer</b></summary>

The process of inferring something about population based on what is measured in the sample is called statistical inference.


</details>

---

Q. What is point estimation in statistical inference?

<details><summary><b>Answer</b></summary>

In point estimation, we estimate an unknown parameter using a single number that is calculated from the sample data.

</details>

---

Q. What do you mean by interval estimation?

<details><summary><b>Answer</b></summary>

In interval estimation, we estimate an unknown parameter using an interval of values that is likely to contain the true value of that parameter and we also state how confident we are that this interval indeed captures the true value of the parameter.

</details>

---

Q. What do we do in hypothesis testing?

<details><summary><b>Answer</b></summary>

In hypothesis testing, we have some claim about the population and we check whether or not the data obtained from the sample provide evidence against this claim.

</details>

---

Q. A blurb on a box of brand X light bulbs claimed that the mean lifetime of each lightbulb is 750 hours. A random sample of 36 light bulbs was tested in a laboratory, and it was found that their average lifetime is 745 hours. Which form of statistical inference should you use to evaluate whether the data provide enough evidence against the advertised mean lifetime on the box?

- Point Estimation
- Interval Estimation
- Hypothesis Testing

<details><summary><b>Answer</b></summary>

<b>Hypothesis Testing</b>

Here, we are assessing whether the data provide enough evidence against the claim that the mean lifetime is $750$ hours.

</details>

---

Q. A recent poll asked a random sample of 1,100 U.S. adults whether or not they support gay marriage. Based on the results of the poll, the pollsters estimated that the proportion of all U.S. adults who support gay marriage is 0.61. Which form of statistical inference should you use to evaluate this conclusion?

- Point Estimation
- Interval Estimation
- Hypothesis Testing

<details><summary><b>Answer</b></summary>

<b>Point Estimation</b>

Here, we are using the data to estimate the proportion of all U.S adults who support gay marriage by a single number $0.61$.

</details>

---

Q. Based on data collected from a random sample of 1,200 college freshmen, researchers are 95% confident that the mean number of sleep hours of all college freshmen is between 6 hours and 7.5 hours. Which form of statistical inference should you use to evaluate this conclusion?

- Point Estimation
- Interval Estimation
- Hypothesis Testing

<details><summary><b>Answer</b></summary>

<b>Interval Estimation</b>

Here, we are estimating the mean number of daily sleep hours of college freshman by an interval of values ($6$ to $7.5$ hours).

</details>

---

Q. How does the type of variable of interest(categorical/quantitative) determine the type of population parameter we need to infer?

<details><summary><b>Answer</b></summary>

It depends on the type of variable of interest:

- Categorical variable - In this case the population parameter that we need to infer about is the population proportion(p) associated with that variable.

- Quantitative variable - In this case the population parameter that we need to infer about is the population mean($\mu$) associated with that variable.

</details>

---

Q. Which of the following statements are true in context of sampling mean $\hat{X}$ and population mean $\mu$.
- Both $\hat{X}$ and $\mu$ are random varaibles.
- Only $\hat{X}$ is a random variable.
- Both are constant values.
- Only $\mu$ is random varaible.

<details><summary><b>Answer</b></summary>

Only $\hat{X}$ is a random variable.

</details>

---

### Point Estimation

Q. A study on exercise habits used a random sample of $2,540$ college students ($1,220$ females and $1,320$ males).

The study found the following:
- $818$ of the females in the sample exercise on a regular basis.
- $924$ of the males in the sample exercise on a regular basis.
- The average time that the $1742$ students who exercise on a regular basis ($818 + 924$) spend exercising per week is $4.2$ hours.

1. What is the point estimate for the proportion of all female college students who exercise on a regular basis?
2. What is the point estimate for the proportion of all college students who exercise on a regular basis?
3. Which of the following has a point estimate of $4.2$?
    - The mean time that all college students who exercise on a regular basis spend exercising per week
    - The mean time that all college students spend exercising per week
    - The percentage of all college students who exercise on a regular basis

<details><summary><b>Answer</b></summary>

Let $n$ be the total sample size and $m$ and $f$ denotes number of male and female students respectively.

1. Since $f = 1220$ and number of feamles that do excercise are $818$

$$\hat{p} = \frac{818}{1220}$$

$$\hat{p} = 0.67$$

2. point estimate for the proportion of all college students who exercise on a regular basis($\hat{p}$):

$$\hat{p} = \frac{818+924}{2540}$$
$$\hat{p} = 0.685$$

3. The mean time that all college students who exercise on a regular basis spend exercising per week

</details>

---

Q. What should be the criteria under which the point estimates are truly unbiased estimates for the population parameter?

<details><summary><b>Answer</b></summary>

Sample should be random and the study design should not be flawed.

</details>

---

Q. What should be the criteria under which the point estimates are truly unbiased estimates for the population parameter?

<details><summary><b>Answer</b></summary>

Sample should be random and the study design should not be flawed.

</details>

---

Q. A researcher wanted to estimate µ, the mean number of hours that students at a large state university spend exercising per week. The researcher collects data from a sample of 150 students who leave the university gym following a workout.

Which of the following is true regarding x̄, the average number of hours that the 150 sampled students exercise per week?
- It is an unbiased estimate for $µ$.
- It is not an unbiased estimate for $µ$ and probably underestimates $µ$.
- It is not an unbiased estimate for $µ$ and probably overestimates $µ$.

<details><summary><b>Answer</b></summary>

It is not an unbiased estimator for µ because the sample was not a random sample of 150 students from the entire student body. In addition, students who leave the university gym following a workout are likely students who exercise on a regular basis and therefore tend to exercise more, on average, than students in general.

</details>

---

Q. A study estimated that the mean number of children per family in the the United States is 1.3. This point estimate would be unbiased and most accurate if it were based on which of the following?
- A random sample of $10,000$ U.S. families with children from the state of Utah
- A random sample of $500$ U.S. families with children
- A random sample of $5,000$ U.S. families with children
- A random sample of $1,000$ U.S. families

<details><summary><b>Answer</b></summary>

<b>A random sample of $1,000$ U.S. families</b>

The estimate is based on a random sample (and is therefore unbiased) and is also based on a large sample, which makes it more accurate.

</details>

---

Q. What is the limitation of point estimation?

<details><summary><b>Answer</b></summary>

The point estimation is simple and intuitive but a little bit problematic. When we estimate $\mu$ by the sample mean $\bar{x}$, we almost guaranteed to make some kind of error. Even though we know the values of $\bar{x}$ fall around $\mu$, it is very unlikely that $\bar{x}$ will fall exactly at $\mu$.

</details>

---


### Interval Estimation

Q. How does interval estimation overcome limitation of point estimation?

<details><summary><b>Answer</b></summary>

Interval estimation enhances point estimation by supplying information about size of error attached.

</details>

---

Q. Suppose a random sample of size n is taken from a normal population of values for a quantitative variable whose mean ($μ$) is unknown, when the standard deviation ($σ$) is given. A $95%$ confidence interval (CI) for $μ$ is:

<details><summary><b>Answer</b></summary>

$$(\bar{x} - 1.96\*\frac{\sigma}{\sqrt{n}}, \bar{x} + 1.96\*\frac{\sigma}{\sqrt{n}})$$

</details>

---

Q. How should we interpret the $95%$ CI for a population mean($\mu$)?

<details><summary><b>Answer</b></summary>

$95%$ confidence interval means we are $95%$ confident that the population mean($\mu$) is covered by the interval. In other words if we take $100$ samples drawn randomly from the population then CI with $95%$ confidence of $95$ of those samples will contains population mean($\mu$).

</details>

---

Q. The IQ level of students at a particular university has an unknown mean, $μ$, and a known standard deviation, $σ = 15$. A simple random sample of $100$ students is found to have a sample mean IQ, $\hat{x} = 115$. Estimate $μ$ with $90%$, $95%$, and $99%$ confidence intervals.

<details><summary><b>Answer</b></summary>

Given:

$σ = 15$,  $\hat{x} = 115$ and $n = 100$ 

- A $90%$ confidence interval for $\mu$ is $\hat{x} \pm 1.645\frac{\sigma}{sqrt{n}} = (112.5, 117.5)$
- A $95%$ confidence interval for $\mu$ is $\hat{x} \pm 2\frac{\sigma}{sqrt{n}} = (112, 118)$
- A $99%$ confidence interval for $\mu$ is $\hat{x} \pm 2.576\frac{\sigma}{sqrt{n}} = (111, 119)$

</details>

---

Q. Explain the trade-off between the level of the confidence and the precision with which the parameter is estimated?

<details><summary><b>Answer</b></summary>

The price we have to pay for a higher level of confidence is that the unknown population mean $\mu$ will be estimated with less precision (i.e., with a wider confidence interval). If we would like to estimate $\mu$ with more precision (i.e., a narrower confidence interval), we will need to sacrifice and report an interval with a lower level of confidence.

</details>

---

Q. Write the general structure of the confidence intervals.

<details><summary><b>Answer</b></summary>

General form:

$$\bar{x} \pm z^* \dot \frac{\sigma}{\sqrt{n}}$$

Where $z^*$ is a general notation for the multiplier that depends on the level of the confidence. We can split the above expression into two parts.

- sample mean $\bar{x}$, the point estimator for the unknown population mean($\mu$)
- margin of error <b>m</b> as $z^* \dot \frac{\sigma}{\sqrt{n}}$, It represents the maximum estimation error for a given level of the confidence.

So we can also write the general form as follows:

$$estimate \pm margin \quote of \quote error$$


</details>

---

Q. Explain margin of error in interval estimation. What value does it encode?

<details><summary><b>Answer</b></summary>

Expression of margin of error(m):

$$m = z^* \dot \frac{\sigma}{\sqrt{n}}$$

Where $z^*$ represents confidence multiplier and $\frac{\sigma}{sqrt{n}} depicts standard deviation of point estimator$.

margin of error(m) is <b>in charge of the width(or precision) of the confidence interval</b>.

</details>

---

Q. How can we reduce margin or error $m$ without compromising on the level of confidence?

<details><summary><b>Answer</b></summary>

With larger sample size $n$ we can reduce the margin of error.

</details>

---

Q. Find the general expression for the required $n$ for a desired margin of error $m$ and certain level of confidence.

<details><summary><b>Answer</b></summary>

We have,

$$m = z^* \dot \frac{\sigma}{\sqrt{n}}$$

On solving for $n$,

$$n = (\frac{z^* \sigma}{m})^2$$

</details>

---

Q. Suppose that based on a random sample, a $95%$ confidence interval for the mean hours slept (per day) among graduate students was found to be $(6.5, 6.9)$. What is the margin of error of this confidence interval?

<details><summary><b>Answer</b></summary>

<b>0.2</b>

Let $m$ be the margin of the error for $95%$ CI and $\bar{x}$ be the mean of hours slept. 

Confidence interval can be given by $(\bar{x} - m, \bar{x} + m)$.

On comparing the actaul interval $(6.5, 6.9)$ and parameterized interval $(\bar{x} - m, \bar{x} + m)$, we have

$$\bar{x} - m = 6.5$$

and,

$$\bar{x} + m = 6.9$$

On solving for $m$,

$$m = 0.2$$

</details>

---

Q. IQ scores are known to vary normally with a standard deviation of $15$. How many students should be sampled if we want to estimate the population mean IQ at $99%$ confidence with a margin of error equal to $2$?


<details><summary><b>Answer</b></summary>

<b>374</b>

We have, 

$$n = (\frac{z^* \sigma}{m})^2$$

On putting $z^* = 2.576$, $\sigma  = 15$ and $m = 2$, we get

$$n = (\frac{2.576(15)}{2})^2 = 374$$


</details>

---

Q. In which case it is not safe to use confidence interval developed using CLT?
1. Variable varies normaly and sample size is small($n < 30$)
2. Variable varies normaly and sample size is large
3. Variable does not vary normal and sample size is small
4. Variable does not vary normal and sample size is large

<details><summary><b>Answer</b></summary>

3. Variable does not vary normal and sample size is small

In this case we can use non-parametric methods.

</details>

---

---

Q. How should we calculate confidence interval when the population standard deviation $\sigma$ is not known?

<details><summary><b>Answer</b></summary>

We can replace population standard deviation($\sigma$) with the sample standard deviation $s$. But in this case we loose central limit theorem and normality of $\bar{X}$, and therefore the confidence multiplier $z^*$ for the different levels of confidence are not generally accurate. The new multiplier comes from a different distribution called $t$ distribution and denoted by $t^*$ instead of $z^*$.

The confidence interval for the population mean ($\mu$) when ($\sigma$) is unknown is therefore:

$$\bar{x} \pm t^* \dot \frac{s}{\sqrt(n)}$$

Note that the quantity $\frac{s}{\sqrt(n)}$ is called the standard error of $\bar(X)$.

</details>

---

Q. <b>True/False</b> For large values of $n$, the $t^*$ multipliers are not much different from the $z^*$ multipliers?

<details><summary><b>Answer</b></summary>

<b>True</b>

</details>

---

Q. Write the expression of confidence interval when variable of interest is categorical?

<details><summary><b>Answer</b></summary>

The confidence interval for the population proportion $p$ is:

$$\hat{p} \pm z^* \dot \sqrt{\frac{\hat(p)(1 - \hat{p})}{n}}$$

</details>

---

Q. A poll asked a random sample of $1,000$ U.S. adults, "Do you think that the use of marijuana should be legalized?" $560$ of those asked answered yes.
1. Based on the poll's results, estimate p, the proportion of all U.S. adults who believe the use of marijuana should be legalized, with a 95% confidence interval.
2. Give an interpretation of the margin of error in context.
3. Do the results of this poll give evidence that the majority of U.S. adults believe that the use of marijuana should be legalized?

<details><summary><b>Answer</b></summary>
 
1. The sample proportion $\hat{p}$ is $\frac{56}{1000} = 0.56$ and therefore  $95%$ confidence interval for $p$ is:

$$0.56 \pm 2\dot \sqrt{0.56(1-0.56)}{1000} = 0.56 \pm 0.03$$

So we are $95%$ confident that the proportion of U.S adults who believe that marijuana should be legalized is between $0.53$ and $0.59$.

2. The margin of error is $0.03$ i.e $3%$. With $95%$ certainty, the sample proportion we got $56%$ is within $3%$ of(or no more than $3%$ away from) the proportion of U.S adults who believe that the use of marijiuana should be legalized.

3. Yes. All of the values in our $95%$ confidence interval for p $(.53, .59)$, which represents the set of plausible values for p, lies above $.5$, which provides evidence (at the $95%$ confidence level) that the majority of U.S. adults believe that the use of marijuana should be legalized.
</details>

---

Q. Under what condition we can use to construct CI in case of estimating $p$ using $z^*$?

<details><summary><b>Answer</b></summary>
 
$$n \dot \hat{p} \geq 10 \quote n \dot (1 - \hat{p}) \geq 10  $$

</details>

---

Q. Suppose that you take $100$ random newborn puppies and determine that the average weight is $1$ pound with the population standard deviation of $0.12$ pounds. Assuming the weight of newborn puppies follows a normal distribution, calculate the $95\\%$ confidence interval for the average weight of all newborn puppies.

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

Q. Suppose that we examine $100$ newborn puppies and the $95%$ confidence interval for their average weight is $[0.9, 1.1]$ pounds. Which of the following statements is true?
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

Q. Suppose we have a random variable X supported on $[0,1]$  from which we can draw samples. How can we come up with an unbiased estimate of the median of X?

<details><summary><b>Answer</b></summary>

</details>

---

Q. The weight of newborn puppies is roughly symmetric with a mean of 1 pound and a standard deviation of 0.12. Your favorite newborn puppy weighs 1.1 pounds.
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

Q. When should you use a Z-Test instead of a T-Test?

<details><summary><b>Answer</b></summary>

</details>

---

### Hypothesis Testing

Q. Define statistical hypothesis testing. Explain in detail how does it work?

<details><summary><b>Answer</b></summary>

Assessing evidence provided by the data in favour of or against some claim about population is statistical hypothesis testing.

Here is how the process of statistical hypothesis testing works:

1. We have two claims about what is going on in the population. Let's call them for now claim 1 and claim 2.
2. We choose a sample, collect relevant data and summarize them.
3. We figure out how likely it is to observe data like the data we got, had claim 1 been true.
4. Based on what we found in the previous step, we make our decision:
    - If we find that if claim 1 were true it would be extremely unlikely to observe the data that we observed, then we have strong evidence against claim 1, and we reject it in favor of claim 2.
    - If we find that if claim 1 were true observing the data that we observed is not very unlikely, then we do not have enough evidence against claim 1, and therefore we cannot reject it in favor of claim 2.

</details>

---

Q. Define $p-value$ in context of hypothesis testing.

<details><summary><b>Answer</b></summary>

p-value is the probability of getting data like those observed when $H_0$ is true. In more general the $p-value$ is the probability of observing a test statistic as extreme as that observed (or even more extreme) assuming that the null hypothesis is true.

By "extreme" we mean extreme in the direction of the alternative hypothesis.

Specifically, for the z-test for the population proportion:

1. If the alternative hypothesis is $H_a : p < p_0$ (less than), then "extreme" means small, and the p-value is:

The probability of observing a test statistic as small as that observed or smaller if the null hypothesis is true.

2. If the alternative hypothesis is $H_a : p > p_0$ (greater than), then "extreme" means large, and the p-value is:

The probability of observing a test statistic as large as that observed or larger if the null hypothesis is true.

3. if the alternative is $H_a : p ≠ p_0$ (different from), then "extreme" means extreme in either direction either small or large (i.e., large in magnitude), and the p-value therefore is:

The probability of observing a test statistic as large in magnitude as that observed or larger if the null hypothesis is true.

</details>

---

Q. State steps involve in hypothesis testing for population proportion.

<details><summary><b>Answer</b></summary>

1. State the appropriate null and alternative hypotheses, $H_0$ and $H_a$.

$$H_0 : p = p_0$$

$$H_a: p \neq p_0 \cup p > p_0 \cup p < p_0$$

2. Obtain the data from a sample and:

Check whether the data satisfy the following condition:

- Random sample
- $n \dot p_0 \geq 10$, $n \dot (1 - p_0) \geq 10$

Calculate the sample proportion $\hat{p}$ and summarize the data using test statistic:

$$z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1 - p_0)}{n}}}$$

3. Find the p-value of the test:

- for $H_a : p < p_0$ : $P(Z \leq z)$
- for $H_a : p > p_0$ : $P(Z \geq z)$
- for $H_a : p \neq p_0$ : $P(Z \geq |z|)$

4. Based on the p-value, decide whether or not the results are significant, and draw your conclusions in context.

</details>

---

Q. Explain significance level of the test and its use in hypothesis testing.

<details><summary><b>Answer</b></summary>

Significance level of the test is usually denoted by the Greek letter $\alpha$. The most commonly used significance level is $\alpha = .05$ (or $5%$). This means that:
- if the $p-value < \alpha$ (usually $.05$), then the data we got is considered to be "rare (or surprising) enough" when Ho is true, and we say that the data provide significant evidence against $H_0$, so we reject Ho and accept $H_a$.
- if the $p-value > \alpha$ (usually $.05$), then our data are not considered to be "surprising enough" when Ho is true, and we say that our data do not provide enough evidence to reject $H_0$ (or, equivalently, that the data do not provide enough evidence to accept $H_a$).

</details>

---

Q. In hypothesis testing when do we conclude the statistical significance of the result?

<details><summary><b>Answer</b></summary>

It depends on $p-value$ and $\alpha$:

- The results are statistically significant - when the $p-value < \alpha$.

- The results are not statistically significant - when the $p-value > \alpha$.

</details>

---

Q. There are rumors that students in a certain liberal arts college are more inclined to use drugs than U.S. college students in general. Suppose that in a simple random sample of $400$ students from the college, 76 admitted to marijuana use. Do the data provide enough evidence to conclude that the proportion of marijuana users among the students in the college (p) is higher than the national proportion, which is $0.157$? 

<details><summary><b>Answer</b></summary>

1. State the hypothesis:

$$H_0: p = 0.157$$

$$H_0: p > 0.157$$

2. Calculate test-statistic $z$:

$$\hat{p} = \frac{76}{400} = 0.19$$

$$z = \frac{0.19 - 0.157}{\sqrt{\frac{0.157(1 - 0.157)}{400}}} = 1.81$$

3. Calculate p-value using software we can get:
$$p - value = 0.035$$

4. Make conclusion based on $p$ value and $\alpha$:

For default value of $\alpha = 0.05$, since $p-value < \alpha$ the result seems significant and alternate hypothesis seems true. 

</details>

---

Q. Write the general form that can be taken by null hypothesis $H_0$ and alternate hypohesis $H_a$.


<details><summary><b>Answer</b></summary>

General form of null hypothesis:

$$H_0 : p = p_0$$

The alternative hypothesis takes one of the following three forms (depending on the context):

$$Ha: p < p_0(one-sided)$$

$$Ha: p > p_0(one-sided)$$

$$Ha: p ≠ p_0(two-sided)$$

</details>

---

Q. How does null hypothesis is related with confidence interval? Explain it with an example.


<details><summary><b>Answer</b></summary>

Suppose we want to test $H_0 : \mu = \mu_0$ vs $H_a : \mu \neq \mu_0$ using a significance level of $\alpha = 0.05$. An alternative way to perform this test is to find a $95%$ confidence interval for $\mu$ and make following conclusions:

- If $\mu_0$ falls outside the confidence interval, reject $H_0$.
- If $\mu_0$ falls inside the confidence interval, do not reject $H_0$.


</details>

---

Q. Explain the difference between z-distribution and t-distribution.

<details><summary><b>Answer</b></summary>



</details>

---

Q. What is Type I and Type II error in case of hypothesis testing?

<details><summary><b>Answer</b></summary>

- <b>Type I Error</b> - If the null hypothesis is true but we reject it.
- <b>Type II Error</b>> - If null hypothesis is true but we fail to reject it.


</details>


---

Q. What do you mean by test statistic in hypothesis testing?


<details><summary><b>Answer</b></summary>

Test statistic captures the essence of the test. Larger the test statistic, the further the data are from $H_0$ and therefore the more evidence the data provide against $H_0$.

</details>

---

Q. Tossing a coin fifteen times resulted in 10 heads and 5 tails. How would you analyze whether a coin is fair?

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


Q. Statistical significance.
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
    - Under null hypothesis($H_0$) p-value distribution will be uniformly distributed.
    - Under alternate hypothesis($H_a$) p-value distribution will be rightly skewed.

3. There are some limitations of p-value and statistical significance and should be used with some cautions:
   - P-values only provide information about the probability of obtaining results under the null hypothesis.
   - they do not tell you the magnitude of an effect or its practical significance.
   - A small p-value doesn't prove that a result is practically significant or that it has any real-world importance.
   - Statistical significance should be considered in the context of the study design, sample size, and the relevance of the result to the research question.
   - Always consider effect sizes, confidence intervals, and the domain-specific context of the research in addition to p-values.


</details>

### Inference for Relationship



Q. Variable correlation.
1. What happens to a regression model if two of their supposedly independent variables are strongly correlated?
1. How do we test for independence between two categorical variables?
1. How do we test for independence between two continuous variables?

<details><summary><b>Answer</b></summary>

1. If predictors of a regression model are highly correlated, we might have following issues:
    - It will make interpretability harder, it will be harder to get individual feature impact on final outcome.
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

Q. What is the difference between parametric and non-parametric tests in machine learning?

<details><summary><b>Answer</b></summary>
</details>

---

Q. A/B testing is a method of comparing two versions of a solution against each other to determine which one performs better. What are some of the pros and cons of A/B testing?

<details><summary><b>Answer</b></summary>

A/B testing is a powerful technique used in a variety of fields including web design, marketing, product development, and more, to make data-driven decisions. 

<p>
<b>Pros:</b>

1. Empirical Evidence - A/B testing provides concrete, quantitative data on how to versions of a solutions perform against each other. It mitigate the error of naive assumptions or gut feelings.
2. Risk Mitigation - It allows to test changes on small portion of users before rolling out to everyone, reducing the risk of implementing a change that could negatively affect the user experience.
3. Data driven decision making
</p>

<p>
<b>Cons</b>

1. Time consuming setup - Setting up A?B tests can be time consuming, especially if you're testing minor changes.
2. Limited by sample size - To get statistically significant results a large sample size is needed.
3. Can Be Misleading: If not properly designed (e.g., not randomizing the sample properly), tests can produce misleading results. Misinterpretation of results can lead to incorrect conclusions.

</p>

</details>

---

Q. You want to test which of the two ad placements on your website is better. How many visitors and/or how many times each ad is clicked do we need so that we can be $95%$ sure that one placement is better?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Your company runs a social network whose revenue comes from showing ads in newsfeeds. To double revenue, your coworker suggests that you should just double the number of ads shown. Is that a good idea? How do you find out?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Imagine that you have the prices of $ 10,000 stocks over the last 24-month period and you only have the price at the end of each month, which means you have 24 price points for each stock. After calculating the correlations of $10,000 * 9,9992$ pairs of stock, you found a pair that has a correlation to be above 0.8.
1. What’s the probability that this happens by chance?
1. How to avoid this kind of accidental pattern?
    
<details><summary><b>Answer</b></summary>
    
</details>

---

Q. How are sufficient statistics and the Information Bottleneck Principle used in machine learning?

<details><summary><b>Answer</b></summary>

<p>
Sufficient statistics and the Information Bottleneck (IB) Principle are fundamental concepts in statistics and information theory that have found important applications in machine learning. 
</p>

<b>Sufficient Statistics</b>

It is a statistic that captures all the information about a parameter of interest in a dataset, without needing to access the entire dataset again.

**Use in Machine Learning:**

- **Parameter Estimation:** Sufficient statistics are used in parameter estimation for probabilistic models. For example, in estimating the parameters of a Gaussian distribution, the sample mean and sample variance are sufficient statistics for the distribution's mean and variance, respectively. This efficiency reduces computational complexity and storage requirements.
- **Model Simplification:** By identifying sufficient statistics, machine learning practitioners can simplify models and reduce the dimensionality of data, focusing only on the parts of the data that are informative for the task at hand.

<b>Information Bottleneck Principle</b>

<p>
The Information Bottleneck (IB) Principle is a method for finding the relevant information in a random variable (X) about another variable (Y). It seeks to compress (X) into a compact representation (T) that preserves as much information about (Y) as possible. The principle balances the trade-off between compression (minimizing the mutual information (I(T;X))) and prediction accuracy (maximizing the mutual information (I(T;Y))).
</p>

**Use in Machine Learning:**

- **Feature Selection and Dimensionality Reduction**: The IB principle can guide the selection of features that are most informative of the target variable, effectively reducing the dimensionality of the input data while retaining its predictive power.
- **Deep Learning Architectures**: In deep neural networks, the layers can be viewed as performing successive information bottleneck operations, progressively compressing the input data into representations that are increasingly informative about the output. This perspective helps in understanding and designing neural network architectures.
- **Clustering and Data Compression**: The IB principle is used in clustering and data compression algorithms to find compact representations of the data that preserve relevant information for specific tasks.
- **Regularization and Generalization**: By focusing on the information that is relevant for predicting the target variable, the IB principle can help in designing regularization techniques that improve the generalization of machine learning models.

</details>


