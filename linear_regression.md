# Data Science Interview Questions And Answers

Topics
---

- [Linear Regression](#linear-regression)
- [Ridge and Lasso Regularization](#ridge-and-lasso-regularization)


## Linear Regression

1. What is linear regression, and how does it work?

<details><summary><b>Answer</b></summary>

Linear regression is a statistical model that assumes the regression function $E(Y|X)$ is linear or nearly linear. 

It takes the following form:

$$f(X) = \beta_{0} + \sum^{p}_{j=1}X_{j}\beta_{j}$$

Note that here $\beta_{j}$'s are unknown parameter or coefficients and the variables $X_{j}$ can come from different sources like
- Quantitative inputs or its transformations
- Basis expansion, such as $X_{2} = X_{1}^2$, $X_{3} = X_{1}^3$ leading to a polynomial representation
- Encoded categorical values
- Interaction between variables like $X_{3} = X_{1} \dot X_{2}$

It uses **least squares** as a estimation method to calculate the values of coefficients.

</details>

3. How to determine the coefficients of a simple linear regression model?

<details><summary><b>Answer</b></summary>

Suppose we have a set of training data $(x_1, y_1),...,(x_n, y_n)$ from which we need to estimate the parameters $\beta$. Linear regression uses least squares estimation method to get values of the parameters. We pick the coefficients $\beta = (\beta_{0}, \beta_{1},....,\beta_{p}^{T})$ to minimize the residual sum of squares(RSS):
$$RSS(\beta) = \sum_{i=1}^{N}(y_{i} - f(x_{i}))^2 = \sum_{i=1}^{N}(y_{i} - \beta_{0} - \sum_{j=1}^{p}x_{ij}\beta_{j})^2$$

Alternatively we can rewrite the above equations as:
$$RSS(\beta) = (y - X\beta)^{T}(y - X\beta)$$

In order to minimize the above expression, differentiating with respect to $\beta$ we get,
$$\frac{\del{RSS}}{\del{\beta}} = -2X^{T}(y - X\beta)$$
$$\frac{\del^{2}{RSS}}{\del{\beta}\del{\beta^{T}}} = 2X^{T}X$$

Assuming that $X$ has full column rank, hence $X^{T}X$ is positive definite so minima exists, we set first derivative to zero
$$X^{T}(y - X\beta) = 0$$

To obtain the unique solution
$$\beta_{cap} = (X^{T}X)^{-1}X^{T}y$$

</details>


2. In which scenarios linear model can outperforms fancier non linear models?

<details><summary><b>Answer</b></summary>

In following cases in may happen:
- Low signal to noise ratio
- Near perfect linearity between predictors and the target variable
- Sparse data
- Small number of training instances

</details>

3. Suppose a model takes form of $f(X) = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{1}^{2}....$, Is it a linear model?

<details><summary><b>Answer</b></summary>

Yes model is still linear in nature. This is polynomial representation of a linear model.

We can write the given form in its linear mode:
$$f(X) = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2}....$$
Where $X_{2} = X_{1}^{2}$

No matter the source of the $X$, the model is linear in its parameter.
</details>


2. What are the assumptions of linear regression?

<details><summary><b>Answer</b></summary>
The main assumptions of linear models are following:
- Linear relationship between predictors and response
    - If not then model may underfit and give bias predictions
- Predictors should be independent of each other(Non-Collinearity)
    - Otherwise it makes interpretation of output messy and unnecessary complicate model
- Homoscedasticity : Constant variance in the error terms(residual)
    - The standard errors, confidence intervals and hypothesis testing rely on this assumption
- Uncorrelated error terms(residuals)
    - If residuals are correlated then we may have pseudo confidence in our model
- Data should not have outliers
    - Can messed up the predictions if we have heavy outliers
</details>

3. Explain the difference between simple linear regression and multiple linear regression.

<details><summary><b>Answer</b></summary>

The key difference between simple linear regression and multiple linear regression lies in the number of independent variables used to predict the dependent variable.

**Differences**

1. Number of Independent Variables:
   - Simple Linear Regression: One independent variable.
   - Multiple Linear Regression: Two or more independent variables.

2. Complexity:
   - Simple Linear Regression: Simpler and easier to interpret since it involves only one predictor.
   - Multiple Linear Regression: More complex due to the involvement of multiple predictors, and it requires more sophisticated techniques for interpretation and model validation.

3. Equation Form**:
   - Simple Linear Regression: $Y = \beta_0 + \beta_1 X + \epsilon$
   - Multiple Linear Regression: $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_k X_k + \epsilon$

</details>

4. What is Residual Standard Error(RSE) and how to interpret it?
<details><summary><b>Answer</b></summary>

The RSE is an estimate of the standard deviation of residuals($\epsilon$). It is the average amount by which the response will deviate from the true regression line.

It is computed using the formula:
$$RSE = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}{y_i - \hat{y}_i}}$$

It is considered as the lack of the fit of the data. Lower values indicates model fits the data very well.

</details>


4. What is the purpose of the coefficient of determination (R-squared) in linear regression?
<details><summary><b>Answer</b></summary>

$R^2$ statistic provides goodness of fit and it is a unit less quantity so its better than the residual standard error.

It takes the form of a proportion *proportion of the variance explained* it always takes values between 0 and 1 and it is not dependent on the scale of $Y$. 

To calculate the $R^2$, we have following expressions:
$$R^2 = \frac{TSS - RSS}{TSS} = 1 - frac{RSS}{TSS}$$

Here, 
$$TSS(Total Sum of Squares) = \sum{(y_i - \hat{y})^2}$$
And,
$$RSS(Residual Sum of Squares) = \sum{(y_i - \cap{y})^2}$$

Statistically, it measures the proportion of variability in $Y$ that can be explained using $X$. 
</details>

4. How to interpret the values of $R^2$ statistic?
<details><summary><b>Answer</b></summary>

A number near 0 indicates the regression does not explain the variability in the response, whereas 1 indicates a large proportion of the variability in the response is explained by the regression.

</details>

5. How do you interpret the coefficients in a linear regression model?

<details><summary><b>Answer</b></summary>

Suppose we have a model of form:
$$Y = \beta_0 + \beta_{1}X_1 + \beta_{2}X_2 + \epsilon$$

Here's how to interpret them:
1. Intercept($\beta_0$):
    - It is the point where the regression line crosses the y-axis.
2. Slope Coefficient($\beta_i$):
    - For each independent variable, the slope coefficient ($\beta_{i}$) indicates the expected change in the dependent variable for a one-unit increase in the independent variable, holding all other variables constant.
    - positive/negative values means increase/decrease in independent variables lead to increase/decrease in response
3. Statistical Significance:
    - The p-value associated with each coefficient helps determine if the relationship is statistically significant. 
4. Magnitude:
    - The magnitude of the coefficient shows the strength of the relationship between the independent and dependent variables.

</details>


6. What is the difference between correlation and regression?

<details><summary><b>Answer</b></summary>

- Correlation quantifies the degree to which two variables are related, without distinguishing between dependent and independent variables.
- Regression models the dependence of a variable on one or more other variables, providing a predictive equation and allowing for an analysis of the effect of each predictor.

</details>

7. What are the methods to assess the goodness of fit of a linear regression model?

<details><summary><b>Answer</b></summary>
There are several methods to measure goodness of fit with some pros and cons:

- R-squared ($R^2$)
- Adjusted R-squared
- Residual Standard Error (RSE) or Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC)

We can use combinations of above statistic to evaluate the model performance.
</details>



8. What is the purpose of the F-statistic in linear regression?

<details><summary><b>Answer</b></summary>

F-statistic is mainly used for hypothesis testing where we want to assess whether at least one of the predictors $X_1, X_2, ..., X_p$ is useful in predicting the response.

For example null hypothesis:
$$H_0 = \beta_1 = \beta_2 = ... = \beta_p = 0$$
alternate hypothesis:
$$H_a = at least one \beta_j is non-zero$$

here hypothesis test is performed by computing the F-statistic, 
$$F = \frac{(TSS - RSS)/p}{(RSS/(n-p-1))}$$

If the linear model assumptions are true:
$$E{(TSS-RSS)/p} = \sigma^2$$
and provided $H_0$ is true,
$$E{(TSS-RSS)/p} = \sigma^2$$

So, when there is no relationship between predictors and the response then F-statistic is near to 1 and if $H_a$ is true the  F-statistic will be greater than 1.

</details>


9. What are the potential problems in linear regression analysis, and how can you address them?

<details><summary><b>Answer</b></summary>

Linear regression model may suffer from following issues mainly:

1. **Non-linearity**: Transform variables or use polynomial regression.
2. **Multicollinearity**: Remove or combine correlated predictors, use regularization.
3. **Heteroscedasticity**: Use robust standard errors, transform the dependent variable.
4. **Outliers**: Identify and handle outliers using diagnostic plots or robust regression.
5. **Overfitting**: Use cross-validation, simplify the model, or apply regularization.
6. **Non-normality of Residuals**: Transform variables or use non-parametric methods.

</details>

10. What are some regularization techniques used in linear regression, and when are they applicable?

11. Can you explain the concept of bias-variance trade-off in the context of linear regression?

## Ridge and Lasso Regularization

12. What's the main purpose of L1 and L2 regularization in linear regression?

13. How do L1 and L2 regularization affect the model's coefficients?

14. What are the hyperparameters associated with L1 and L2 regularization?

15. When would you choose L1 regularization over L2, and vice versa?

16. What is Elastic Net regularization, and how does it relate to L1 and L2 regularization?

17. How do you choose the optimal regularization strength (alpha) for L1 and L2 regularization?

18. What are the consequences of multicollinearity?

<details><summary><b>Answer</b></summary>

Multi collinearity can pose problems in regression context:
- It can be difficult to separate out the individual effects of collinear variables on the response.
- It reduces the accuracy of the estimate of the regression coefficients
- It reduces the power of the hypothesis testing - like *probability of correctly detecting a non zero coefficient* 

</details>

19. How can you detect collinearity in a regression model?

<details><summary><b>Answer</b></summary>

A simple way to detect collinearity is to look at the correlation matrix of the predictors. A large absolute value in that matrix indicates a pair of highly correlated variables.

</details>

19. How can you detect multicollinearity in a regression model?

<details><summary><b>Answer</b></summary>

We can detect multicollinearity using *variance inflation factor(VIF)*. VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity.

The VIF is the ratio of the variance of $\beta_j$ when fitting the full model divided by the variance of $\beta_j$ if fit on its own. 
$$ VIF(\beta_j) = \frac{1}{1-R^{2}_{X_{j}|X_{-j}}}$$

where $R^{2}_{X_{j}|X_{-j}}$ is the $R^2$ from a regression of $X_j$ onto all the other predictors.

The smallest possible value of VIF is 1, which indicates complete absence of collinearity. In practice we have small collinearity among the predictors so VIF greater tha 5 or 10 depicts problematic amount of collinearity.

</details>


20. How can you address multicollinearity?

<details><summary><b>Answer</b></summary>

There are several methods to address multicollinearity:
- Remove Highly Correlated Predictors
- Combine Correlated Predictors
- Principal Component Analysis (PCA)
- Use ridge or lasso regression or combination of both(elastic net regression) for modeling

</details>

21. Can you have perfect multicollinearity?

<details><summary><b>Answer</b></summary>

Yeah, It may occur when one predictor variable in a regression model is an exact linear combination of one or more other predictor variables. In other words, the correlation between the variables is exactly 1 (or -1).

</details>