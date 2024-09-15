# Data Science Interview Questions And Answers

## Descriptive Statistics

Contents
----

- [Basics](#basics)
- [Examining Distribution - Univariate](#examining-distribution---univariate)
- [Examining Relationships - Multivariate](#examining-relationships---multivariate)
- [Sampling](#sampling)
- [Design Studies](#design-studies)

---

## Basics

Q. What do you mean by Data?

<details><summary><b>Answer</b></summary>

Data are pieces of information about individuals organized into variables.

</details>

---

Q. Define dataset.

<details><summary><b>Answer</b></summary>

A dataset is a set of data identified with particular circumstances. Datasets are typically displayed in tables, in which rows represent individuals and columns represent variables.

</details>

---

Q. What are the two main types of variables?

<details><summary><b>Answer</b></summary>

- Categorical/qualitative variables - It take category or label values and place an individual into one of several groups.
- Quantitative variables - It takes numerical values and represent some kind of measurement.

</details>

---

## Examining Distribution - Univariate

Q. How can we summarize distribution of categorical variable?

<details><summary><b>Answer</b></summary>

The distribution of a categorical variable is summarized using:
- Graphical display: pie chart or bar chart
- Numerical summaries: category counts and percentages

</details>

---

Q. State some methods for visualizing the numerical summaries of categorical data?

<details><summary><b>Answer</b></summary>

- Pie Chart
- Bar graph 
- Pictogram

</details>

---

Q. State the difference between pie chart and bar chart?

<details><summary><b>Answer</b></summary>

- Pie charts are used to show the proportion or percentage of categories as parts of a whole. They illustrate how each category contributes to the total.
- Bar charts are used to compare quantities across different categories. They show the magnitude of each category in a visual format.

</details>

---

Q. What are graphical methods to visualize the distribution of a quantitative variable?

<details><summary><b>Answer</b></summary>

- Histogram
- Stemplot
- Boxplot

</details>

---

Q. Define histogram?

<details><summary><b>Answer</b></summary>

The histogram is a graphical display of the distribution of a quantitative variable.

</details>

---

Q. Write the steps of plotting histogram?

<details><summary><b>Answer</b></summary>

- Collect and Organize Data: Sort the data into intervals or bins. Each bin will represent a range of values.
- Determine Bin Intervals: Decide the width of each bin, which depends on the range and distribution of your data. This step often involves balancing between having too many narrow bins or too few wide bins.
-  Count Frequencies: Count how many data points fall into each bin. This gives you the frequency of data points within each interval.
- Set Up the Axes: X-axis(bin intervals) and Y-axis(frequency or count of data points)
- Draw the Bars: For each bin, draw a bar that reaches up to the frequency count on the Y-axis. The width of each bar corresponds to the width of the bin interval, and the height corresponds to the frequency.
- Label the Histogram: Add title, axes labels and legends

</details>

---

Q. How can we numerically summarize a distribution represented by histogram?

<details><summary><b>Answer</b></summary>

We can using following features of the histogram to summarize it

- Overall pattern
   - Shape
   - Center
   - Spread
- Outliers

</details>

---

Q. What insights can be gained from examining the shape of a histogram?

<details><summary><b>Answer</b></summary>

- Symmetry/skewness of the distribution.
- Peakedness (modality) — The number of peaks (modes) the distribution has.

</details>

---

Q. What is the difference between a unimodal, a bimodal and a multimodal histogram?

<details><summary><b>Answer</b></summary>

- Unimodal histogram has only one peak or mode.
- A bimodal histogram has two distinct peaks or modes. 
- A multimodal histogram has more than two distinct peaks 

<table align='center'>
<tr>
<td align="center">
    <img src="img/unimodal_bimodal_graph.png" alt= "Unimodal, Bimodal and Multimodal Histogram" style="max-width:70%;"/>
</td>
</tr>
<tr>
<td align="center"> Unimodal, Bimodal and Multimodal Histogram </td>
</tr>
</table>

</details>

---

Q. State the modality of an uniform distribution?

<details><summary><b>Answer</b></summary>

Uniform distributions has no modes i.e no values around which observations are concentrated.

</details>

---

Q. Under what conditions does a distribution become skewed to the right?

<details><summary><b>Answer</b></summary>

A distribution is said to be skewed right (or positively skewed) when the right tail (the higher value side) of the distribution is longer or fatter than the left tail.

</details>

---

Q. What does right skewed histogram indicates about the dataset?

<details><summary><b>Answer</b></summary>

It indicates that the bulk of the data values are clustered towards the lower end of the range, with a few extreme values stretching out towards the higher end.

</details>

---

Q. Can you draw a right skewed histogram?

<details><summary><b>Answer</b></summary>

<table align='center'>
<tr>
<td align="center">
    <img src="img/right_skewed_hist.png" alt= "Right skewed histogram" style="max-width:70%;"/>
</td>
</tr>
<tr>
<td align="center"> Skewed Right Distribution </td>
</tr>
</table>


</details>

---

Q. State the relationship between mean, median and mode in case of skewed right distributions?

<details><summary><b>Answer</b></summary>

For a right-skewed distribution, the general order is:

$$\text{Mode} < \text{Median} < \text{Mean}$$

</details>

---

Q. State a real-life scenario of a distribution which is skewed right?

<details><summary><b>Answer</b></summary>

Salary of workers across industries. Most people earn in the low/medium range of salaries, with a few exceptions (CEOs, professional athletes etc.) that are distributed along a large range (long "tail") of higher values.

</details>

---

Q. Under what conditions does a distribution become skewed to the left?

<details><summary><b>Answer</b></summary>

A distribution is said to be skewed left (or negatively skewed) when the left tail (the lower value side) of the distribution is longer or fatter than the right tail.

</details>

---

Q. What does left skewed histogram indicates about the dataset?

<details><summary><b>Answer</b></summary>

It indicates that the bulk of the data values are clustered towards the higher end of the range, with a few extreme values stretching out towards the lower end i.e outliers are on lower end of the distribution.

</details>

---

Q. Can you draw a left skewed histogram?

<details><summary><b>Answer</b></summary>

<table align='center'>
<tr>
<td align="center">
    <img src="img/left_skewed_hist.png" alt= "Left skewed histogram" style="max-width:70%;"/>
</td>
</tr>
<tr>
<td align="center"> Skewed Left Distribution </td>
</tr>
</table>

</details>

---

Q. State the relationship between mean, median and mode in case of skewed left distributions?

<details><summary><b>Answer</b></summary>

For a left-skewed distribution, the general order is:

$$\text{Mode} > \text{Median} > \text{Mean}$$

</details>

---

Q. State the relationship between mean, median and mode in case of symmetric distributions?

<details><summary><b>Answer</b></summary>

For a left-skewed distribution, the general order is:

$$\text{Mode} ~ \text{Median} ~ \text{Mean}$$

</details>

---

Q. State a real-life scenario of a distribution which is skewed left?

<details><summary><b>Answer</b></summary>

An example of a real life variable that has a skewed left distribution is age of death from natural causes (heart disease, cancer etc.). Most such deaths happen at older ages, with fewer cases happening at younger ages.

</details>

---

Q. What is a stemplot? What are benefits of using it?

<details><summary><b>Answer</b></summary>

The stemplot is a simple but useful visual display of quantitative data distribution.

Benefits of using it:
- Easy and quick to construct for small, simple datasets.
- Retains the actual data.
- Sorts (ranks) the data.

</details>

---

Q. State the main numerical measures of center of a quantitative variable distribution?

<details><summary><b>Answer</b></summary>

The three main numerical measures for the center of a distribution are:
- Mode
- Mean
- Median

</details>

---

Q. What is mode of a distribution?

<details><summary><b>Answer</b></summary>

Mode is the most commonly occurring value in a distribution.

</details>

---


Q. What is mean of a distribution?

<details><summary><b>Answer</b></summary>

The mean is the average of a set of observations (i.e., the sum of the observations divided by the number of observations). If the n observations are $x_1, x_2,..,x_n$ their mean, which we denote by $\bar{x}$

$$
\bar{x} = \frac{x_1 + x_2 + ... + x_n}{n}
$$

</details>

---

Q. What is median of a distribution? 

<details><summary><b>Answer</b></summary>

The median M is the midpoint of the distribution. It is the number such that half of the observations fall above, and half fall below.

</details>

---

Q. How can we calculate median of a distribution? 

<details><summary><b>Answer</b></summary>

To find the median:

- Order the data from smallest to largest.
- Consider whether $n$, the number of observations, is even or odd.
   - If $n$ is odd, the median $M$ is the center observation in the ordered list. This observation is the one "sitting" in the $(n + 1)/2$ spot in the ordered list.
   - If $n$ is even, the median $M$ is the mean of the two center observations in the ordered list. These two observations are the ones "sitting" in the $n/2$ and $n/2 + 1$ spots in the ordered list.

</details>

---

Q. How can we calculate median of a distribution? 

<details><summary><b>Answer</b></summary>

To find the median:

- Order the data from smallest to largest.
- Consider whether $n$, the number of observations, is even or odd.
   - If $n$ is odd, the median $M$ is the center observation in the ordered list. This observation is the one "sitting" in the $(n + 1)/2$ spot in the ordered list.
   - If $n$ is even, the median $M$ is the mean of the two center observations in the ordered list. These two observations are the ones "sitting" in the $n/2$ and $n/2 + 1$ spots in the ordered list.

</details>

---

Q. State the main difference between mean and median?

<details><summary><b>Answer</b></summary>

The mean is very sensitive to outliers (because it factors in their magnitude), while the median is resistant to outliers.

</details>

---

Q. How can we decide which measure should we choose as center of distribution?

<details><summary><b>Answer</b></summary>

The mean is an appropriate measure of center only for symmetric distributions with no outliers. In all other cases, the median should be used to describe the center of the distribution.

</details>

---

Q. What are commonly used measures of spread?

<details><summary><b>Answer</b></summary>

- Range
- Inter-quartile range (IQR)
- Standard deviation

</details>

---

Q. What do you mean by range of a distribution?

<details><summary><b>Answer</b></summary>

The range covered by the data is the most intuitive measure of variability. The range is exactly the distance between the smallest data point (min) and the largest one (Max).

$$
\text{Range} = \text{Max} - \text{Min}
$$

</details>

---

Q. What is IQR? How can we calculate it?

<details><summary><b>Answer</b></summary>

IQR measures the variability of a distribution by giving us the range covered by the middle $50%$ of the data.

<table align='center'>
<tr>
<td align="center">
    <img src="img/IQR.png" alt= "IQR" style="max-width:70%;"/>
</td>
</tr>
<tr>
<td align="center"> Inter Quartile Range(IQR) </td>
</tr>
</table>

IQR calculations:

- Arrange the data in increasing order, and find the median M. Recall that the median divides the data, so that 50% of the data points are below the median, and 50% of the data points are above the median.

- Find the median of the lower 50% of the data($Q1$) and upper 50% of the data $Q3$.

- The middle 50% of the data falls between $Q1$ and $Q3$, and therefore:

$$
\text{IQR} = Q3 - Q1
$$


<table align='center'>
<tr>
<td align="center">
    <img src="img/IQR_calc.png" alt= "IQR" style="max-width:70%;"/>
</td>
</tr>
<tr>
<td align="center"> IQR calculations </td>
</tr>
</table>

</details>

---

Q. Can we use IQR to detect outliers?

<details><summary><b>Answer</b></summary>

Yes, we can use IQR as basis for a rule of thumb for identifying outliers. 

It is $1.5(IQR)$ criteria for outliers. An observation is considered a suspected outlier if it is:

- below Q1 - 1.5(IQR) or
- above Q3 + 1.5(IQR)

<table align='center'>
<tr>
<td align="center">
    <img src="img/IQR_for_outliers.png" alt= "IQR for outlier detection" style="max-width:70%;"/>
</td>
</tr>
<tr>
<td align="center"> IQR for outlier detection </td>
</tr>
</table>

</details>

---

Q. How to deal with outliers in the data?

<details><summary><b>Answer</b></summary>

We can handle outliers with following approach:
- Identifying the outlier
- Understanding the outlier
- Decide how to handle the outlier

</details>

---

Q. What are the five-number summary statistics of a distribution?

<details><summary><b>Answer</b></summary>

The five-number summary of a distribution consists of the median (M), the two quartiles (Q1, Q3) and the extremes (min, Max).

</details>

---

Q. What is a box-plot?

<details><summary><b>Answer</b></summary>

The boxplot graphically represents the distribution of a quantitative variable by visually displaying the five-number summary and any observation that was classified as a suspected outlier using the *1.5(IQR)* criterion.

<table align='center'>
<tr>
<td align="center">
    <img src="img/box-plot.png" alt= "Box plot" style="max-width:70%;"/>
</td>
</tr>
<tr>
<td align="center"> Box plot </td>
</tr>
</table>

</details>

---

Q. State one benefit of using box-plot?

<details><summary><b>Answer</b></summary>

Boxplots are most useful when presented side-by-side to compare and contrast distributions from two or more groups.

</details>

---

Q. What does the standard deviation represent in a distribution?

<details><summary><b>Answer</b></summary>

standard deviation quantifies the spread of a distribution by measuring how far the observations are from their mean $\bar{x}$. The standard deviation gives the average (or typical distance) between a data point and the mean $\bar{x}$.

$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$

where:
- $N$ is the number of observations in the population.
- $x_i$ represents each individual observation.
- $\mu$ is the mean of the population.

</details>

---

Q. What is the benefit of using IQR over SD for measure of spread?

<details><summary><b>Answer</b></summary>

The SD is strongly influenced by outliers in the data.

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

Q. How do we determine which numerical summaries to use for describing a distribution?

<details><summary><b>Answer</b></summary>

- Use $\bar{x}$ (the mean) and the standard deviation $\sigma$ as measures of center and spread only for reasonably symmetric distributions with no outliers.

- Use the five-number summary (which gives the median, IQR and range) for all other cases.

</details>

---

Q. State standard deviation rule?

<details><summary><b>Answer</b></summary>

For distributions having bell shaped (also known as the normal shape), the following rule applies:
- Approximately $68%$ of the observations fall within 1 standard deviation of the mean.
- Approximately $95%$ of the observations fall within 2 standard deviations of the mean.
- Approximately $99.7%$ (or virtually all) of the observations fall within $3$ standard deviations of the mean.

<table align='center'>
<tr>
<td align="center">
    <img src="img/std_rules.webp" alt= "Standard Deviation Rule" style="max-width:70%;"/>
</td>
</tr>
<tr>
<td align="center"> Standard Deviation Rule </td>
</tr>
</table>

</details>

---

## Examining Relationships - Multivariate

Q. Define explanatory and response variable?

<details><summary><b>Answer</b></summary>

- The explanatory variable (also commonly referred to as the independent variable) — the variable that claims to explain, predict, or affect the response
- The response variable (also commonly referred to as the dependent variable) — the outcome of the study.

</details>

--- 

Q. How can we examine the relationship between a categorical explanatory variable and quantitative response variable?

<details><summary><b>Answer</b></summary>

We can use following techniques:
- Side by side boxplots
- Descriptive Statistics
   - Five Number summary for each categorical value

</details>

---

Q. How can we examine the relationship between a categorical explanatory variable and categorical response variable?

<details><summary><b>Answer</b></summary>

The relationship between two categorical variables is summarized using
- Data display: Two-way table, supplemented by
- Numerical summaries: Conditional percentages.

</details>

---

Q. How can we examine the relationship between a numerical explanatory variable and numerical response variable?

<details><summary><b>Answer</b></summary>

The relationship between two quantitative variables is visually displayed using the scatterplot, where each point represents an individual. We always plot the explanatory variable on the horizontal X axis, and the response variable on the vertical Y axis.

<table align='center'>
<tr>
<td align="center">
    <img src="img/Q_Q_relationship.png" alt= "Q->Q relationship" style="max-width:70%;"/>
</td>
</tr>
<tr>
<td align="center"> Q -> Q relationship </td>
</tr>
</table>

</details>

---

Q. What is Simpsons Paradox?

<details><summary><b>Answer</b></summary>

Simpsons Paradox is a statistical phenomenon that occurs when you combine subgroups into one group. The process of aggregating data can cause the apparent direction and strength of the relationship between two variables to change.


</details>

---

Q. Why Does Simpson’s Paradox Occur?

<details><summary><b>Answer</b></summary>

Simpson’s Paradox occurs because a third variable can affect the relationship between a pair of variables.

</details>

---



## Sampling

## Design Studies
