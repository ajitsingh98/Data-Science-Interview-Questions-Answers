# Deep Learning Interview Questions

Topics
---

- [Logistic Regression](#logistic-regression)


## Logistic Regression

Contents
---
- [General Concepts](#general-concepts)
- [Odds and Log-odds](#odds-and-log-odds)
- [The Sigmoid](#the-sigmoid)
- [Truly Understanding Logistic Regression](#truly-understanding-logistic-regression)
- [The Logit Function and Entropy](#the-logit-function-and-entropy)

### General Concepts

1. **True or False**: For a fixed number of observations in a data set, introducing more variables normally generates a model that has a better fit to the data. What may be the drawback of such a model fitting strategy?

<details style='color: red;'>
    <summary><b>Answer</b></summary>
    <p style='color: red'>
    <b>True</b> But if the inducted features do not provide enough information and act like redundant predictors, then it does not make sense to add those predictors to the model. It unnecessarily increases the complexity of the model and may cause overfitting issues.
    </p>
</details>

---
2. Define the term **“odds of success”** both qualitatively and formally. Give a numerical example that stresses the relation between probability and odds of an event occurring.
---
3. Answer the following:
    1. Define what is meant by the term **"interaction"**,in the context of a logistic regression predictor variable?
    1. What is the simplest form of an interaction? Write its formulae.
    1. What statistical tests can be used to attest the significance of an interaction term?
---
4. **True or False**: In machine learning terminology, unsupervised learning refers to the
mapping of input covariates to a target response variable that is attempted at being predicted when the labels are known.
---
5. **Complete the following sentence**: In the case of logistic regression, the response variable is the log of the odds of being classified in `[...]`.
---
6. Describe how in a logistic regression model, a transformation to the response variable is applied to yield a probability distribution. Why is it considered a more informative representation of the response?
---
7. Complete the following sentence: Minimizing the negative log likelihood also means
maximizing the `[...]` of selecting the `[...]` class.
---
8. Assume the probability of an event occurring is `p = 0.1`.
    1. What are the `odds` of the event occurring?
    2. What are the `log-odds` of the event occurring?
    3. Construct the `probability` of the event as a ratio that equals 0.1
---

### Odds and Log-odds

9. **True or False**: If the odds of success in a binary response is $4$, the corresponding probability of success is $0.8$.
---

10. Draw a graph of odds to probabilities, mapping the entire range of probabilities to
their respective odds.

---

11. The logistic regression model is a subset of a broader range of machine learning models known as generalized linear models (GLMs), which also include analysis of variance (ANOVA), vanilla linear regression, etc. There are three components to a GLM; identify these three components for binary logistic regression.

---

12. Let us consider the logit transformation, i.e., log-odds. Assume a scenario in which the
logit forms the linear decision boundary, for a given vector of systematic components X and predictor variables θ. Write the mathematical expression for the hyperplane that describes the decision boundary.

$$
\log{\frac{Pr(Y = 1 | X)}{Pr(Y = 0|X)}} = \theta_0 + \theta^TX
$$

---

13. **True or False**: The logit function and the natural logistic (sigmoid) function are inverses
of each other.

---

### The Sigmoid

14. Compute the derivative of the natural sigmoid function:

$$
\sigma(x) = \frac{1}{1+e^{-x}} \epsilon (0, 1)
$$

---

15. Remember that in logistic regression, the hypothesis function for some parameter vector
β and measurement vector x is defined as:

$$
h_\beta(x) = g(\beta^Tx) =  \frac{1}{1+e^{-\beta^Tx}}\\
= P(y = 1|x;\beta)
$$

where y holds the hypothesis value. Suppose the coefficients of a logistic regression model with independent variables are as follows: $\beta_0 = -1.5, \beta_1 = 3, \beta_2 = -0.5$. Assume additionally, that we have an observation with the following values for the independent variables: $x_1 = 1, x_2 = 5$. As a result, the logit equation becomes: $logit = \beta_0 + \beta_1x_1 + \beta_2x_2$.
1. What is the value of the logit for this observation?
2. What is the value of the odds for this observation?
3. What is the value of $P(y = 1)$ for this observation?
   
---
### Truly Understanding Logistic Regression

16. Proton therapy (PT) is a widely adopted form of treatment for many types of cancer including breast and lung cancer (Fig. 2.2).
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-1.png" alt="A multi-detector positron scanner used to locate tumors" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Pulmonary nodules (left) and breast cancer (right) </td>
  </tr>
</table>
A PT device which was not properly calibrated is used to simulate the treatment of cancer. As a result, the PT beam does not behave normally. A data scientist collects information relating to this simulation. The covariates presented in Table 2.1 are collected during the experiment. The columns Yes and No indicate if the tumour was eradicated or not, respectively.
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-2.png" alt="A multi-detector positron scanner used to locate tumors" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center">Tumour eradication statistics</td>
  </tr>
</table>
Referring to Table 2.1: Answer the following questions

1. What is the explanatory variable and what is the response variable?
2. Explain the use of relative risk and odds ratio for measuring association.
3. Are the two variables positively or negatively associated? Find the direction and strength of the association using both relative risk and odds ratio.
4. Compute a 95% confidence interval (CI) for the measure of association.
5. Interpret the results and explain their significance.
   
---
   
17. Consider a system for radiation therapy planning (Fig. 2.3). Given a patient with a malignant tumour, the problem is to select the optimal radiation exposure time for that patient. A key element in this problem is estimating the probability that a given tumour will be erad- icated given certain covariates. A data scientist collects information relating to this radiation therapy system.
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-3.png" alt="A multi-detector positron scanner used to locate tumors"style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center">A multi-detector positron scanner used to locate tumors</td>
  </tr>
</table>

The following covariates are collected; $X_1$ denotes time in milliseconds that a patient is irradiated with, $X_2$ = holds the size of the tumour in centimeters, and $Y$ notates a binary response variable indicating if the tumour was eradicated. Assume that each response’ variable $Y_i$ is a Bernoulli random variable with success parameter $p_i$, which holds:

$$
p_i = \frac{e^{\beta_0+\beta_1x_1+\beta_2x_2}}{1+e^{\beta_0+\beta_1x_1+\beta_2x_2}}
$$

The data scientist fits a logistic regression model to the dependent measurements and produces these estimated coefficients:

$$
\hat{\beta_{0}} = -6 \\
\hat{\beta_{1}}= 0.05\\
\hat{\beta_{2}} = 1
$$

1. Estimate the probability that, given a patient who undergoes the treatment for $40 \ milliseconds$ and who is presented with a tumour sized $3.5\ cm$, the system eradicates the tumour.
2. How many milliseconds the patient in part (a) would need to be radiated with to have exactly a $50%$ chance of eradicating the tumour?
   
---

18. Recent research suggests that heating mercury containing dental amalgams may cause the release of toxic mercury fumes into the human airways. It is also presumed that drinking hot coffee, stimulates the release of mercury vapour from amalgam fillings (Fig. 2.4).
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-4.png" alt="A multi-detector positron scanner used to locate tumors"style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> A dental amalgam </td>
  </tr>
</table>
To study factors that affect migraines, and in particular, patients who have at least four dental amalgams in their mouth, a data scientist collects data from $200K$ users with and without dental amalgams. The data scientist then fits a logistic regression model with an indicator of a second migraine within a time frame of one hour after the onset of the first migraine, as the binary response variable (e.g., migraine=1, no migraine=0). The data scientist believes that the frequency of migraines may be related to the release of toxic mercury fumes.
There are two independent variables:
    1. $X_1 = 1$ if the patient has at least four amalgams; $0$ otherwise.
    2. $X_2$ = coffee consumption (0 to 100 hot cups per month).
The output from training a logistic regression classifier is as follows:
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-5.png" alt= "A dental amalgam" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> A dental amalgam </td>
  </tr>
</table>
    1. Using $X_1$ and $X_2$, express the odds of a patient having a migraine for a second time. 
    2. Calculate the probability of a second migraine for a patient that has at least four amalgams and drank 100 cups per month?
    3. For users that have at least four amalgams, is high coffee intake associated with an increased probability of a second migraine?
    4. Is there statistical evidence that having more than four amalgams is directly associated with a reduction in the probability of a second migraine?
    
---

19. To study factors that affect Alzheimer’s disease using logistic regression, a researcher
considers the link between gum (periodontal) disease and Alzheimer as a plausible risk factor. The predictor variable is a count of gum bacteria (Fig. 2.5) in the mouth.
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-7.png" alt= "A chain of spherical bacteria." style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> A chain of spherical bacteria. </td>
  </tr>
</table>
The response variable, Y , measures whether the patient shows any remission (e.g. yes=1).

The output from training a logistic regression classifier is as follows:
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-8.png" alt= " output from training a logistic regression classifier" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> output from training a logistic regression classifier </td>
  </tr>
</table>

1. Estimate the probability of improvement when the count of gum bacteria of a patient is 33.
2. Find out the gum bacteria count at which the estimated probability of improvement is 0.5.
3. Find out the estimated odds ratio of improvement for an increase of 1 in the total gum bacteria count.
4. Obtain a 99% confidence interval for the true odds ratio of improvement increase of 1 in the total gum bacteria count. Remember that the most common confidence levels are 90%, 95%, 99%, and 99.9%. Table 9.1 lists the z values for these levels.

<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-9.png" alt= "Common confidence levels" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Common confidence levels </td>
  </tr>
</table>

---

20.  Recent research suggests that cannabis (Fig. 2.6) and cannabinoids administration in particular, may reduce the size of malignant tumours in rats.
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-10.png" alt= "Cannabis" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Cannabis </td>
  </tr>
</table>
To study factors affecting tumour shrinkage, a deep learning researcher collects data from two groups; one group is administered with placebo (a substance that is not medicine) and the other with cannabinoids. His main research revolves around studying the relationship (Table 2.3) between the anticancer properties of cannabinoids and tumour shrinkage:
<table align='center'>
  <tr>
    <td align="center">
      <img src="img/lr-11.png" alt= "Tumour shrinkage in rats" style="max-width:70%;" />
    </td>
  </tr>
  <tr>
    <td align="center"> Tumour shrinkage in rats </td>
  </tr>
</table>

For the true odds ratio:
1. Find the sample odds ratio.
2. Find the sample log-odds ratio.
3. Compute a 95% confidence interval $(z_{0.95} = 1.645; z_{0.975} = 1.96)$ for the true log odds ratio and true odds ratio.

---

### The Logit Function and Entropy

21. The entropy of a single binary outcome with probability p to receive 1 is defined as:
    $$H(p) ≡ −p\log{p}−(1−p)\log(1−p)$$
    1. At what p does H(p) attain its maximum value?
    2. What is the relationship between the entropy H(p) and the logit function, given p?
<details><summary><b>Answer</b></summary>
    
---
</details>
