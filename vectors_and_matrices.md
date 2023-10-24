# Maths Questions

## Contents
- [Vector](#vector)
- [Matrices](#matrices)
- [Calculus and Convex Optimization](#calculus-and-convex-optimization)
- [Statistics](#statistics)
- [Probability](#probability)

## Vector

1. Dot product
    1. What’s the geometric interpretation of the dot product of two vectors?
    1. Given a vector $u$ , find vector $v$  of unit length such that the dot product of $u$  and $v$  is maximum.
1. Outer product
    1. Given two vectors $a=[3,2,1]$  and $b=[−1,0,1]$. Calculate the outer product $a^Tb$ ?
    1. Give an example of how the outer product can be useful in ML.
1. What does it mean for two vectors to be linearly independent?
1. Given two sets of vectors $A=a_1,a_2,a_3,...,a_n$  and $B=b_1,b_2,b_3,...,b_m$. How do you check that they share the same basis?
1. Given $n$  vectors, each of $d$  dimensions. What is the dimension of their span?
1. Norms and metrics
    1. What's a norm? What is  $L_0,L_1,L_2,L_{norm}$?
    1. How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?

## Matrices

1. Why do we say that matrices are linear transformations?
2. What’s the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?
3. What does the determinant of a matrix represent?
4. What happens to the determinant of a matrix if we multiply one of its rows by a scalar  $t×R$ ?
5. A $4×4$  matrix has four eigenvalues $3,3,2,−1$. What can we say about the trace and the determinant of this matrix?
6. Given the following matrix:
```math
\begin{bmatrix}
1 & 4 & -2\\
-1 & 3 & 2 \\
3 & 5 & -6
\end{bmatrix}
```
Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?

7. What’s the difference between the covariance matrix $A^TA$  and the Gram matrix $AA^T$ ?
8. Given $A∈R^{n×m}$  and $b∈R^n$ 
    1. Find $x$ such that: $Ax=b$.
    1. When does this have a unique solution?
    1. Why is it when $A$ has more columns than rows, $Ax=b$ has multiple solutions?
    1. Given a matrix $A$ with no inverse. How would you solve the equation  $Ax=b$ ? What is the pseudoinverse and how to calculate it?
9. Derivative is the backbone of gradient descent.
    1. What does derivative represent?
    1. What’s the difference between derivative, gradient, and Jacobian?
10. Say we have the weights $w∈R^{d×m}$  and a mini-batch $x$  of $n$  elements, each element is of the shape $1×d$  so that $x∈R^{n×d}$. We have the output $y=f(x;w)=xw$. What’s the dimension of the Jacobian $\frac{δy}{δx}$?
11. Given a very large symmetric matrix $A$ that doesn’t fit in memory, say $A∈R^{1M×1M}$  and a function $f$ that can quickly compute $f(x)=Ax$ for $x∈R1M$. Find the unit vector $x$ so that $x^TAx$  is minimal.

## Calculus and Convex Optimization

1. Differentiable functions
    1.  What does it mean when a function is differentiable?
    1. Give an example of when a function doesn’t have a derivative at a point.
    1. Give an example of non-differentiable functions that are frequently used in machine learning. How do we do backpropagation if those functions aren’t differentiable?
2. Convexity
    1. What does it mean for a function to be convex or concave? Draw it.
    1. Why is convexity desirable in an optimization problem?
    1. Show that the cross-entropy loss function is convex.
3. Given a logistic discriminant classifier:
$
p(y=1|x)=σ(w^Tx)
$
where the sigmoid function is given by:
$
σ(z)=(1+exp(−z))^{−1}
$
The logistic loss for a training sample $x_i$  with class label $y_i$  is given by $L(yi,xi;w)=−logp(y_i|x_i)$
    1. Show that  $p(y=−1|x)=σ(−w^Tx)$.
    1. Show that  $Δ_wL(y_i,x_i;w)=−y_i(1−p(y_i|x_i))x_i$.
    1. Show that  $Δ_wL(y_i,x_i;w)$  is convex.
4. Most ML algorithms we use nowadays use first-order derivatives (gradients) to construct the next training iteration.
    1. How can we use second-order derivatives for training models?
    1. Pros and cons of second-order optimization.
    1. Why don’t we see more second-order optimization in practice?
5. How can we use the Hessian (second derivative matrix) to test for critical points?
6. Jensen’s inequality forms the basis for many algorithms for probabilistic inference, including Expectation-Maximization and variational inference.. Explain what Jensen’s inequality is.
7. Explain the chain rule.
8. Let $x∈R_n$ , $L=crossentropy(softmax(x),y)$ in which $y$  is a one-hot vector. Take the derivative of $L$  with respect to $x$.
9. Given the function $f(x,y)=4x^2−y$  with the constraint $x^2+y^2=1$. Find the function’s maximum and minimum values.

## Statistics

1. Explain frequentist vs. Bayesian statistics.
2. Given the array $[1,5,3,2,4,4]$, find its mean, median, variance, and standard deviation.
3. When should we use median instead of mean? When should we use mean instead of median?
4. What is a moment of function? Explain the meanings of the zeroth to fourth moments.
5. Are independence and zero covariance the same? Give a counterexample if not.
6. Suppose that you take $100$ random newborn puppies and determine that the average weight is $1$ pound with the population standard deviation of $0.12$ pounds. Assuming the weight of newborn puppies follows a normal distribution, calculate the $95%$ confidence interval for the average weight of all newborn puppies.
7. Suppose that we examine $100$ newborn puppies and the $95%$ confidence interval for their average weight is $[0.9,1.1]$ pounds. Which of the following statements is true?
    1. Given a random newborn puppy, its weight has a $95%$ chance of being between $0.9$ and $1.1$ pounds.
    1. If we examine another $100$ newborn puppies, their mean has a $95%$ chance of being in that interval.
    1. We're 95% confident that this interval captured the true mean weight.
8. Suppose we have a random variable X  supported on $[0,1]$  from which we can draw samples. How can we come up with an unbiased estimate of the median of X ?
9. Can correlation be greater than 1? Why or why not? How to interpret a correlation value of 0.3?
10. The weight of newborn puppies is roughly symmetric with a mean of 1 pound and a standard deviation of 0.12. Your favorite newborn puppy weighs 1.1 pounds.
    1. Calculate your puppy’s z-score (standard score).
    1. How much does your newborn puppy have to weigh to be in the top 10% in terms of weight?
    1. Suppose the weight of newborn puppies followed a skew distribution. Would it still make sense to calculate z-scores?
11. Tossing a coin ten times resulted in 10 heads and 5 tails. How would you analyze whether a coin is fair?
12. Statistical significance.
    1. How do you assess the statistical significance of a pattern whether it is a meaningful pattern or just by chance?
    1. What’s the distribution of p-values?
    1. Recently, a lot of scientists started a war against statistical significance. What do we need to keep in mind when using p-value and statistical significance?
13. Variable correlation.
    1. What happens to a regression model if two of their supposedly independent variables are strongly correlated?
    1. How do we test for independence between two categorical variables?
    1. How do we test for independence between two continuous variables?
14. A/B testing is a method of comparing two versions of a solution against each other to determine which one performs better. What are some of the pros and cons of A/B testing?
15. You want to test which of the two ad placements on your website is better. How many visitors and/or how many times each ad is clicked do we need so that we can be $95%$ sure that one placement is better?
16. Your company runs a social network whose revenue comes from showing ads in newsfeed. To double revenue, your coworker suggests that you should just double the number of ads shown. Is that a good idea? How do you find out?
17. Imagine that you have the prices of $10,000$ stocks over the last 24 month period and you only have the price at the end of each month, which means you have 24 price points for each stock. After calculating the correlations of $10,000 * 9,9992$ pairs of stock, you found a pair that has the correlation to be above 0.8.
    1. What’s the probability that this happens by chance?
    1. How to avoid this kind of accidental patterns?
18. How are sufficient statistics and Information Bottleneck Principle used in machine learning?

## Probability

1. Given a uniform random variable X  in the range of [0,1]  inclusively. What’s the probability that X=0.5 ?
2. Can the values of PDF be greater than 1? If so, how do we interpret PDF?
3. What’s the difference between multivariate distribution and multimodal distribution?
4. What does it mean for two variables to be independent?
5. It’s a common practice to assume an unknown variable to be of the normal distribution. Why is that?
6. How would you turn a probabilistic model into a deterministic model?
7. Is it possible to transform non-normal variables into normal variables? How?
8. When is the t-distribution useful?
9. Assume you manage an unreliable file storage system that crashed 5 times in the last year, each crash happens independently.
    1. What's the probability that it will crash in the next month?
    1. What's the probability that it will crash at any given moment?
10. Say you built a classifier to predict the outcome of football matches. In the past, it's made 10 wrong predictions out of 100. Assume all predictions are made independently, what's the probability that the next 20 predictions are all correct?
11. Given two random variables $X$  and $Y$. We have the values $P(X|Y)$  and $P(Y)$  for all values of $X$  and $Y$. How would you calculate $P(X)$?
12. You know that your colleague Jason has two children and one of them is a boy. What’s the probability that Jason has two sons? 
13. There are only two electronic chip manufacturers: $A$ and $B$, both manufacture the same amount of chips. A makes defective chips with a probability of $30%$, while B makes defective chips with a probability of $70%$.
    1. If you randomly pick a chip from the store, what is the probability that it is defective?
    1. Suppose you now get two chips coming from the same company, but you don’t know which one. When you test the first chip, it appears to be functioning. What is the probability that the second electronic chip is also good?
14. There’s a rare disease that only 1 in 10000 people get. Scientists have developed a test to diagnose the disease with the false positive rate and false negative rate of 1%.
    1. Given a person is diagnosed positive, what’s the probability that this person actually has the disease?
    1. What’s the probability that a person has the disease if two independent tests both come back positive?
15. A dating site allows users to select $10$ out of $50$ adjectives to describe themselves. Two users are said to match if they share at least $5$ adjectives. If Jack and Jin randomly pick adjectives, what is the probability that they match?
16. Consider a person A whose sex we don’t know. We know that for the general human height, there are two distributions: the height of males follows $h_m=N(μ_m,σ^{2}_m)$  and the height of females follows $h_j=N(μ_j,σ^{2}_j)$ . Derive a probability density function to describe A’s height.
17. There are three weather apps, each the probability of being wrong $\frac{1}{3}$ of the time. What’s the probability that it will be foggy in San Francisco tomorrow if all the apps predict that it’s going to be foggy in San Francisco tomorrow and during this time of the year, San Francisco is foggy $50%$ of the time?
18. Given n  samples from a uniform distribution $[0,d]$. How do you estimate $d$? (Also known as the German tank problem)
19. You’re drawing from a random variable that is normally distributed, $X∼N(0,1)$, once per day. What is the expected number of days that it takes to draw a value that’s higher than $0.5$?
20. You’re part of a class. How big the class has to be for the probability of at least a person sharing the same birthday with you is greater than $50%$?
21. You decide to fly to Vegas for a weekend. You pick a table that doesn’t have a bet limit, and for each game, you have the probability $p$ of winning, which doubles your bet, and $1−p$ of losing your bet. Assume that you have unlimited money (e.g. you bought Bitcoin when it was 10 cents), is there a betting strategy that has a guaranteed positive payout, regardless of the value of $p$?
22. Given a fair coin, what’s the number of flips you have to do to get two consecutive heads?
23. In national health research in the US, the results show that the top 3 cities with the lowest rate of kidney failure are cities with populations under $5,000$. Doctors originally thought that there must be something special about small town diets, but when they looked at the top 3 cities with the highest rate of kidney failure, they are also very small cities. What might be a probabilistic explanation for this phenomenon?
24. Derive the maximum likelihood estimator of an exponential distribution.