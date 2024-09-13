# Data Science Interview Questions And Answers


## Performance Metrics

Contents
---

- [Regression Metrics](#linear-regression)
- [Classification Metrics](#ridge-and-lasso-regularization)
- [Clustering Metrics]()
- [Metrics in NLP]()

---

## Regression Metrics

Q. List down metrics for evaluating regression tasks?

<details><summary><b>Answer</b></summary>

We can use following regression metrics for evaluation

- Mean Squared Error(MSE)
- Root Mean Squared Error(RMSE)
- Mean Absolute Error(MAE)
- R-squared(Goodness of fit)
- Adjusted R-squared
- Pearson correlation
- Spearman rank correlation

</details>

---

Q. List down metrics for evaluating regression tasks?

<details><summary><b>Answer</b></summary>

We can use following regression metrics for evaluation

- Mean Squared Error(MSE)
- Root Mean Squared Error(RMSE)
- Mean Absolute Error(MAE)
- R-squared(Goodness of fit)
- Adjusted R-squared
- Pearson correlation
- Spearman rank correlation

</details>

---

Q. Write down mathematical expression of MSE?

<details><summary><b>Answer</b></summary>

$$
\text{MSE} = \frac{1}{n}\sum_{i=i}^{n}(y_{pred, i} - y_{true,i})^2
$$

Where

- $y_{pred, i}$ : Predicted value by model for input vector $x_i$
- $y_{true, i}$ : Ground truth value/label by model for input vector $x_i$

</details>

---

Q. What are benefits of using MSE as loss function?

<details><summary><b>Answer</b></summary>

- Smooth and Differentiable : Suitable for gradient-based optimization methods like gradient descent.
- Penalizes Larger Errors More Heavily: MSE squares the error, which means that larger errors contribute significantly more to the total loss compared to smaller errors. 
- Convexity Ensures a Global Minimum: MSE is a convex function, which means that the optimization landscape has a single global minimum.

</details>

---

Q. State the range of MSE?

<details><summary><b>Answer</b></summary>

Bounds of mean-squared error(mse)

$$
[0, \inf)
$$

with $0$ the best mse

</details>

---

Q. How can we interpret mse value?

<details><summary><b>Answer</b></summary>

MSE estimates the average squared difference of predicted values and actual values. 

- $MSE = 0$: Indicates perfect predictions
- Low MSE: Predictions are closer to actual values
- High MSE: Larger difference between predicted and actual values

</details>

---

Q. What value MSE encode?

<details><summary><b>Answer</b></summary>

MSE measures seeks to summarize the errors made by regression model. The smaller value indicates, model's prediction closer to actual values. It is intuitively like counterpart to accuracy for classifier.

</details>

---

Q. Why do we square the differences in MSE?

<details><summary><b>Answer</b></summary>

Squaring the differences in MSE is done to emphasize larger errors, ensure non-negative values, maintain mathematical properties needed for optimization, align with the concept of variance, and avoid the cancellation of errors.

</details>

---


Q. What is the significance of a low MSE value?

<details><summary><b>Answer</b></summary>

A low Mean Squared Error (MSE) value signifies that a model's predictions are close to the actual values, indicating that the model performs well in terms of accuracy. 

</details>

---


Q. How can you minimize MSE in a machine learning model?

<details><summary><b>Answer</b></summary>

Minimizing MSE involves a combination of techniques:

- Feature Engineering
- Model Selection and Tuning
- Regularization

</details>

---

Q. What are weaknesses of mean-squared error(mse)?

<details><summary><b>Answer</b></summary>

- Scale dependent values: MSE values are highly dependent on scale of the output variable
- Hard to interpret in isolation: Scale dependency makes it hard to interpret in isolation
- Sensitive to outliers: If prediction is an outlier then we may get inflated MSE
- Treats all errors equally irrespective to magnitude and direction
- Overestimation and underestimations are weighted equally 
- May ignore small errors as it squares the difference between predicted and actual values
- It assumes errors should be gaussian distributed 

</details>

---

Q. Write the expression for Root Mean Squared Error(RMSE)?

<details><summary><b>Answer</b></summary>

RMSE is basically root of mean-squared error

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=i}^{n}(y_{pred, i} - y_{true,i})^2}
$$

</details>

---

Q. What is the benefit of using RMSE over MSE?

<details><summary><b>Answer</b></summary>

- RMSE unit is in sync with target variable unit
- Easy to interpret the output and can be used directly to compare multiple models performance
- Does not influence much from outliers unlike MSE

</details>

---

Q. What are weaknesses of RMSE?

<details><summary><b>Answer</b></summary>

- Not much robust to outliers
- Always positive so not useful where direction of error make sense
- Not scale invariant 
- Can be affected by sample size
- Bias towards large values

</details>

---

Q. Write the mathematical expression for Mean Absolute Error(MAE)?

<details><summary><b>Answer</b></summary>

MAE is average of absolute difference of predicted and actual values.

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_{true, i} - y_{pred, i}|
$$

</details>

---

Q. What is the benefit of using MAE over RMSE or MSE?

<details><summary><b>Answer</b></summary>

It is more robust to outliers unlike MSE and RMSE

</details>

---

Q. State one drawback of using MAE over RMSE or MSE?

<details><summary><b>Answer</b></summary>

MAE uses `|.|` function which is not differential at zero

</details>

---


Q. What is the difference between MSE and RMSE (Root Mean Square Error)?

<details><summary><b>Answer</b></summary>

- Units: MSE is in squared units of the target variable, while RMSE is in the same units as the target variable. This makes RMSE generally easier to interpret and communicate.
- Sensitivity to Outliers: Both MSE and RMSE are sensitive to outliers, but since RMSE is the square root of MSE, it is less exaggerated in terms of units. However, it still reflects the impact of large errors due to the squaring operation in MSE.
- Interpretation: RMSE can be more intuitive for understanding the average prediction error in the context of the data, whereas MSE’s squared units can make interpretation less straightforward.

</details>

---

Q. What is Mean Absolute Percentage Error (MAPE), and how is it calculated?

<details><summary><b>Answer</b></summary>

MAPE is a metric used to evaluate the accuracy of a forecasting or regression model. It measures the average absolute percentage error between predicted and actual values. It is calculated as:

$$
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100
$$

where $y_i$ is the actual value, $\hat{y}_i$ is the predicted value, and $n$ is the number of observations.

</details>

---

Q. Why might MAPE be misleading if the actual values of the target variable are very small?

<details><summary><b>Answer</b></summary>

MAPE can be misleading if the actual values are very small because the percentage error becomes very large when the actual values approach zero. This can lead to exaggerated error values that do not accurately reflect the model’s performance. For example, if the actual value is close to zero, even a small absolute error can result in a very high percentage error, skewing the metric.

</details>

---

Q. In which scenarios would MAPE be preferred over metrics like RMSE or MAE?

<details><summary><b>Answer</b></summary>

MAPE is preferred over metrics like RMSE or MAE when you want to evaluate the model’s performance in terms of relative error, especially when comparing forecasts across different scales or units. It is useful when the goal is to understand the average percentage error relative to the actual values, making it easier to communicate the model’s accuracy in percentage terms.

</details>

---

Q. How does MAPE handle negative values in the actual data, and why might this be problematic?

<details><summary><b>Answer</b></summary>

MAPE is not well-suited for datasets with negative values because the percentage error calculation involves dividing by the actual value. When actual values are negative, this division results in negative or undefined percentage errors, which can be problematic and lead to inaccurate or misleading metrics. 

</details>

---

Q. Can MAPE be used to evaluate a model’s performance if the target variable contains zero values?

<details><summary><b>Answer</b></summary>

MAPE cannot be reliably used when the target variable contains zero values, as it involves division by the actual values. When the actual value is zero, the percentage error calculation becomes undefined or infinite.

</details>

---

Q. What are the benefits of using MAPE in a business context, and what potential pitfalls should be considered?

<details><summary><b>Answer</b></summary>

*Benefits*

- MAPE provides an easy-to-understand measure of prediction accuracy expressed as a percentage, which is straightforward to communicate to stakeholders.
- It allows for the comparison of models across different scales and units, making it useful for evaluating performance in a relative context.

*Pitfalls*

- MAPE can be misleading with very small or zero actual values, leading to exaggerated error metrics.
- It does not handle negative values or account for asymmetry in errors, which can affect its suitability for certain types of data or business contexts.

</details>

---

Q. How does MAPE handle asymmetric errors, and what might be the implications for model evaluation?

<details><summary><b>Answer</b></summary>

MAPE does not account for the direction of errors (whether they are positive or negative), which means it treats overestimation and underestimation equally. This can be a limitation if the cost of overestimation is different from underestimation, as MAPE does not differentiate between these scenarios.

</details>

---

Q. What is R-squared ($R^2$)?

<details><summary><b>Answer</b></summary>

R-squared is also called coefficient of determination. It is a statistical measure which evaluates the goodness of fit of a regression model. It represents the proportion of the variance in the dependent variable that is explained by the independent variables in the model. 

</details>

---

Q. When R-squared is more suitable to use?

<details><summary><b>Answer</b></summary>

R-squared is more useful when we have linear relationship between dependent and independent variables.

</details>

---

Q. What is the range of R-squared values?

<details><summary><b>Answer</b></summary>

R-squared ranges from $0$ to $1$, with $1$ indicating a perfect fit and $0$ indicating no fit at all.

</details>

---

Q. What does an R-squared value of 0.75 mean?

<details><summary><b>Answer</b></summary>

An R-squared value of $0.75$ means that 75% of the variance in the dependent variable (target) can be explained by the independent variables (predictors) in the model. In other words, the model accounts for $75%$ of the variability in the data, indicating a relatively good fit.

</details>

---

Q. State the expression for R-squared?

<details><summary><b>Answer</b></summary>

Mathematically, It can be expressed as:

$$
\text{R-squared} = 1 - \frac{SSR}{SST} = 1 - \frac{\sum_{i}^{N}(y_i - \hta{y}_{i})^2}{\sum_{i}^{N}(y_i - \bar{y}_{i})^2}
$$

Where:

- SSR(Sum of Squared residuals) represents the sum of the squared differences between the predicted values and the actual values
- SST(Sum of Squared Total) represents the sum of the squared differences between actual values and mean of the dependent values.

</details>

---

Q. Can R-squared be negative?

<details><summary><b>Answer</b></summary>

Yeah, `R-squared` value can be negative due to outliers i.e when `MSE(model) > MSE(Baseline)`. It is the case when we have heavy outliers in predictions. 

$$
\text{SSR} > \text{SST}
$$

</details>

---

Q. What are the benefits of using R-squared as performance indicator?

<details><summary><b>Answer</b></summary>

- Intuitive Interpretation: R-squared of 0.80 means that 80% of the variance is explained by the model, which is easy to interpret and communicate.
- Scale-Independent: Since R-squared is a proportion of explained variance, it is not affected by the scale of the dependent variable.
- Goodness of Fit: R-squared is a commonly used metric to assess the goodness of fit for linear regression models.

</details>

---

Q. How do you interpret R-squared?

<details><summary><b>Answer</b></summary>

- $R² = 1$: The model explains all the variability in the target variable. Predictions perfectly match the actual values.
- $0 < R² < 1$: R-squared is between 0 and 1, indicating the proportion of variance explained by the model. 
- $R² = 0$: The model does not explain any of the variance in the target variable
- $R² < 0$: Worse Than a Simple Mean Model

</details>

---

Q. What are the limitations of R-squared?

<details><summary><b>Answer</b></summary>

- It does not tell us about magnitude of error
- A high R-squared does not guaranteed accurate predictions, as the model may still have residual errors or maybe overfitting the data
- Inability to determine causality: Higher R-squared values maybe due to presence of some latent/confounding variables
- Limited to linear relationship: If the relationship is non-linear, R-squared may not accurately reflect the goodness of fit
- Influence by number of predictors in the data

</details>

---

Q. When should you use R-squared as an evaluation metric?

<details><summary><b>Answer</b></summary>

Use R-squared as an evaluation metric when you are working with linear regression models and want to assess how well your model explains the variability in the dependent variable. 

</details>

---

Q. Why do we need adjusted R-squared instead of standard R-squared?

<details><summary><b>Answer</b></summary>

R-squared suffers from problem that the scores keep improving on addition of predictors even if the additional predictors do not significantly improve the model's ability to explain the variation in the dependent variable. This can result in overfitting, where the model appears to fit the data well but may not generalize well to new data. To overcome the above problem associated with R-squared, adjusted R-squared adjusts for the number of predictors in the model, penalizing models with more predictors if the additional predictors do not contribute significantly to the model's ability to explain the variance in the dependent variable.

</details>

---

Q. What is the expression of adjusted R-squared?

<details><summary><b>Answer</b></summary>

Mathematically It can be expressed as:

$$
\text{Adjusted R-squared} = 1 - (\frac{(1-R^2).(n-1)}{n - k -1})
$$

Where:

- R-squared(Goodness of fit)
- n: Total number of observations in the dataset
- k: number of predictors(independent variables) in the regression model

</details>

---

Q. What is the limitation of adjusted R-squared?

<details><summary><b>Answer</b></summary>

Adjusted R-squared value is sensitive to sample size. It means the value may decrease as we we increase the sample size even if the model is performing better.

</details>

---

Q. What is pearson correlation?

<details><summary><b>Answer</b></summary>

Pearson Correlation is a statistical method that measures the similarity or correlation between two data objects by comparing their attributes and calculating a score ranging from $-1$ to $+1$.

</details>

---

Q. Write the expression of pearson correlation $\rho$?

<details><summary><b>Answer</b></summary>

The Pearson correlation coefficient between two vectors $y$ and $\cap{y}$ of dimension is:

$$
\rho(y, \hat(y)) = \frac{\text{Cov}(y, \hat{y})}{\sigma_{y} \sigma_{\hat{y}}}
$$

</details>

---

Q. What is the range of pearson correlations?

<details><summary><b>Answer</b></summary>

Bound of $\rhi$ is $[0, 1]$, $+1$ is complete positive linear correlation, and $0$ is no linear correlation at all.

</details>

---

Q. State the weakness of pearson correlation?

<details><summary><b>Answer</b></summary>

- Pearson correlations are highly sensitive to the magnitude of the differences between the actual and predicted values. As a result, they are also very sensitive to outliers.
- More useful to detect correlation in linearly dependent values

</details>

---

Q. What is spearman correlation?

<details><summary><b>Answer</b></summary>

Spearman's rank correlation coefficient, often denoted as $\rho_s$ or $r_s$, is a non-parametric measure of correlation that assesses the strength and direction of the monotonic relationship between two variables. Unlike Pearson’s correlation, which measures linear relationships, Spearman’s correlation evaluates how well the relationship between two variables can be described by a monotonic function. 

</details>

---

Q. What is the benefit of using spearman correlation over pearson correlation?

<details><summary><b>Answer</b></summary>

- None parametric: Does not require relationship to be linear or data should be normally distributed
- Detects Monotonic Relationships: Capable of identifying monotonic relationships that are not strictly linear.
- Not sensitive to outliers
- Better for non-linear trends
- Applicable to Ordinal Data: Can handle ranked data effectively.

</details>

---

Q. Write the formula of spearman correlation?

<details><summary><b>Answer</b></summary>

The formula for Spearman’s rank correlation is:

$$
\rho_s = 1 - \frac{6\sum d_{i}^2}{n(n^2-1)}
$$

Where $d_i$ is the difference between the ranks of each pair of values, and $n$ is the number of observations.

</details>

---

Q. How can you compute the spearman correlation between two variable $X$ and $Y$?

<details><summary><b>Answer</b></summary>

1. Rank the data

- Assign ranks to values of both variables and if there are ties, assign the average rank to the tied values.

2. Calculate the differences and squared sum $\sum d_{i}^2$

- Compute the difference between the ranks for each pair of values

3. Apply the expression of spearman correlation

$$
\rho_s = 1 - \frac{6\sum d_{i}^2}{n(n^2-1)}
$$

</details>

---

Q. How to interpret spearman correlation values?

<details><summary><b>Answer</b></summary>

- $+1$: Perfect positive monotonic correlation (as one variable increases, the other variable also increases).
- $−1$: Perfect negative monotonic correlation (as one variable increases, the other variable decreases).
- $0$: No monotonic correlation (the variables do not exhibit a monotonic relationship).

</details>

---

Q. State the one scenario where spearman correlation is more appropriate than pearson coefficients?

<details><summary><b>Answer</b></summary>

*Scenario*

You are analyzing a dataset where you have customer satisfaction scores and purchase frequencies. The customer satisfaction scores are obtained on an ordinal scale (e.g., ratings from 1 to 5), and the purchase frequencies are counts of how often each customer makes a purchase within a given period.

Data Characteristics:

- Ordinal data

Customer satisfaction scores are ordinal, not interval. This means that while the scores indicate relative satisfaction levels, the intervals between them are not necessarily uniform. For example, the difference between a rating of 1 and 2 might not be the same as the difference between 4 and 5 in terms of satisfaction.

- Non-Linear Relationship

You suspect that the relationship between satisfaction and purchase frequency might be monotonic but not necessarily linear. For instance, higher satisfaction might generally lead to more frequent purchases, but the rate of increase in purchases might not be consistent across the satisfaction scale.

In the above scenario spearman is more appropriate than the pearson coefficients.

</details>

Q. What is the limitation of spearman correlation?

<details><summary><b>Answer</b></summary>

- Monotonic Relationship Assumption: Not suitable for non-linear relationship
- Data Transformation Sensitivity: The correlation might be affected by transformations that alter the ranks of the data.

</details>

---

Q. How can we handle asymmetric errors?

<details><summary><b>Answer</b></summary>

We can use following metrics/loss to handle asymmetric errors:

- Weighted Absolute Error
- Asymmetric Loss Functions
- Quantile Loss
- Huber Loss

</details>

---


## Classification Metrics

Q. What metrics we can use to evaluate a classifier?

<details><summary><b>Answer</b></summary>

We can use following metrics for evaluation:

- Accuracy score
- Precision
- Recall
- F-scores
- Precision Recall curve
- ROC curve

</details>

---

Q. Define accuracy score?

<details><summary><b>Answer</b></summary>

Accuracy scores signifies the portion of true predictions among total cases examined. 

</details>

---

Q. Write the mathematical expression of accuracy score?

<details><summary><b>Answer</b></summary>

It is mathematically expressed as:

$$
\text{accuracy score} = \frac{TP + TN}{TP + FP + FN + TN}
$$

Where:

- TP (True Positives) = Number of scenarios where the model's predicted labels match with the positive labels
- FP (False Positives) = Number of scenarios where the actual label is negative, but the model predicts it as positive(Type I Error)
- FN (False Negatives) = Number of scenarios where the actual label is positive, but the model predicts it as negative(Type I Error)
- TN (True Negative) = Number of instances where both the actual label and the model's prediction are negative

</details>

---

Q. What is the range of accuracy score?

<details><summary><b>Answer</b></summary>

*Accuracy bounds*

$[0, 1]$, with $0$ the worst and $1$ the best.

</details>

---

Q. How does accuracy is related with negative log(logistic) loss?

<details><summary><b>Answer</b></summary>

Accuracy of a classifier on a validation set will be negatively correlated with the negative log (logistic, cross-entropy) loss, which is a common loss for classifiers. In this sense, these classifiers are optimizing for accuracy.

</details>

---

Q. What is the main issue with accuracy score?

<details><summary><b>Answer</b></summary>

Accuracy score provides an overall evaluation of the model's performance across all classes, but it may be deceptive in cases where the label distribution is skewed or imbalanced. In such cases, the accuracy score may be high due to the majority class dominating the predictions, while the minority class may be misclassified.

</details>

---

Q. How can we overcome with accuracy score limitations?

<details><summary><b>Answer</b></summary>

To address accuracy score limitation issue, it's recommended to use other performance metrics along with accuracy score, such as precision, recall, F1-score, or area under the Receiver Operating Characteristic (ROC) curve. These metrics provide insights on the model's performance at a class-level, allowing for a more comprehensive evaluation of the model's predictive accuracy, especially in imbalanced datasets.

</details>

---

Q. What do you mean by precision? How can we calculate it?

<details><summary><b>Answer</b></summary>

Precision is the portion of true positive class among predicted positive classes. 

It can be expressed as:

$$
\text{Precision Score} = \frac{TP}{TP+FP}
$$

Note that precision is per-class notion.

</details>

---

Q. When precision is preferred?

<details><summary><b>Answer</b></summary>

It is preferable choice when you want to be very sure of your predictions.

</details>

---

Q. What is the range of precision value?

<details><summary><b>Answer</b></summary>

$[0, 1]$, with $0$ the worst and $1$ the best. 

</details>

---

Q. What are the weaknesses of precision as a metric in classification, and how can it be misleading in certain scenarios?

<details><summary><b>Answer</b></summary>

A key weakness of precision is that it can be artificially inflated by rarely predicting a class, especially when the class is infrequent or unimportant compared to others.

For example, By focusing on precision, the model may become overly conservative, only predicting positive when it’s very certain, which can result in a high number of false negatives. This is problematic in scenarios where missing positive cases (false negatives) carries a significant cost, such as in medical diagnoses or fraud detection.

</details>

---

Q. State one scenario where you should use precision of the system as evaluation metric?

<details><summary><b>Answer</b></summary>

In an email marketing campaign, high precision means emails are sent mostly to recipients likely to respond, reducing costs by avoiding those unlikely to engage. However, if the model rarely predicts engagement to maintain high precision, it might miss many potential customers, limiting the campaign's reach and overall effectiveness. This highlights how precision can be misleading if used alone, as it prioritizes correctness over coverage.

</details>

---

Q. What is the other name of recall?

<details><summary><b>Answer</b></summary>

True Positive Rate

</details>

---

Q. Define recall score?

<details><summary><b>Answer</b></summary>

Recall is the sum of correct predictions divided by the sum of all true instances. Note that recall is per-class notion. 

</details>

---

Q. What does recall measures? How is it calculated?

<details><summary><b>Answer</b></summary>

Recall measures the proportion of actual positives cases that are correctly predicted by the model

It is calculated as:

$$
\text{recall score} = \frac{TP}{TP+FN}
$$

</details>

---

Q. What does higher value of recall indicates? 

<details><summary><b>Answer</b></summary>

A higher recall score indicates that the model is effectively capturing a larger proportion of the actual positive cases.

</details>

---

Q. When should we use recall? 

<details><summary><b>Answer</b></summary>

It is a suitable metric when the goal is to minimize false negatives, i.e., correctly identifying as many positive cases as possible.

</details>

---

Q. What is the range of recall metric? 

<details><summary><b>Answer</b></summary>

$[0, 1]$, with $0$ the worst and $1$ the best.

</details>

---

Q. What is the main limitation of recall metric? 

<details><summary><b>Answer</b></summary>

Recall's dangerous edge case is that one can achieve very high recall for a category by always guessing it. This could mean a lot of incorrect guesses, but recall sees only the correct ones.

</details>

---

Q. State one scenario where recall is more desirable? 

<details><summary><b>Answer</b></summary>

Recall is prioritized when the cost of missing positive instances (false negatives) is high, as in healthcare. For example, missing a person who needs a flu shot can lead to serious health issues, while giving a shot to someone who doesn't need it incurs minimal cost. Therefore, healthcare providers often prioritize recall over precision, opting to vaccinate a broader audience to avoid missing those who need it.

</details>

---

Q. Explain the tradeoff between recall and precision? 

<details><summary><b>Answer</b></summary>

The tradeoff between precision and recall arises because adjusting the classification threshold impacts both metrics: raising the threshold reduces false positives (boosting precision) but increases false negatives (lowering recall), and lowering the threshold has the opposite effect. This means you can't minimize both errors at once without a perfect classifier, so enhancing one metric usually compromises the other.

</details>

---

Q. Define F Scores?

<details><summary><b>Answer</b></summary>

F scores combine precision and recall via their harmonic mean, with a value  that can be used to emphasize one or the other. Like precision and recall, this is a per-category notion.

It captures the tradeoff between precision and recall:

$$
\text{F Score} = (\beta^2 + 1)\frac{\text{precision} \cdot \text{recall}}{\beta^{2} \cdot \text{precision} + \text{recall}}
$$

</details>

---

Q. What is F-1 Score? When should we use it?

<details><summary><b>Answer</b></summary>

F-1 Score is a special case of F scores when we give equal weighage to precision and recall. In this setup $\beta  = 1$

$$
\text{F-1 Score} = \frac{2 * \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

The F1 score is a suitable choice when both precision and recall are equally important, and we want to balance between the two.

</details>

---

Q. What is range of F Scores?

<details><summary><b>Answer</b></summary>

Bounds of F scores: $[0, 1]$, with $0$ the worst and $1$ the best, and guaranteed to be between precision and recall.

</details>

---

Q. What is the purpose of a confusion matrix?

<details><summary><b>Answer</b></summary>



</details>

---


Q. Explain True Positive (TP) and False Positive (FP).

<details><summary><b>Answer</b></summary>



</details>

---


Q. Define True Negative (TN) and False Negative (FN).

<details><summary><b>Answer</b></summary>



</details>

---


Q. What is accuracy, and how is it calculated using a confusion matrix?

<details><summary><b>Answer</b></summary>



</details>

---

Q. What are precision and recall, and how are they calculated from a confusion matrix?

<details><summary><b>Answer</b></summary>



</details>

---


Q. How can you use a confusion matrix to choose an appropriate threshold for a binary classifier?

<details><summary><b>Answer</b></summary>



</details>

---


Q. What is the F1 score, and how is it related to precision and recall?

<details><summary><b>Answer</b></summary>



</details>

---


Q. Explain the difference between Type I and Type II errors in the context of a confusion matrix.

<details><summary><b>Answer</b></summary>



</details>

---

Q. F1 score.
1. What’s the benefit of F1 over the accuracy?
1. Can we still use F1 for a problem with more than two classes. How?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Your team is building a system to aid doctors in predicting whether a patient has cancer or not from their X-ray scan. Your colleague announces that the problem is solved now that they’ve built a system that can predict with 99.99% accuracy. How would you respond to that claim?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Given a binary classifier that outputs the following confusion matrix.

$$
\begin{bmatrix} 
    \textbf{""} & \textbf{Predicted True} & \textbf{Predicted False} \\\\
    \textbf{Actual True} & 30 & 20 \\\\
    \textbf{Actual False} & 5 & 40 \\\\
\end{bmatrix}
$$

1. Calculate the model’s precision, recall, and F1.
1. What can we do to improve the model’s performance?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Consider a classification where $99%$ of data belongs to class A and $1%$ of data belongs to class B.
1. If your model predicts A 100% of the time, what would the F1 score be? Hint: The F1 score when A is mapped to 0 and B to 1 is different from the F1 score when A is mapped to 1 and B to 0.
1. If we have a model that predicts A and B at a random (uniformly), what would the expected $F_1$ be?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. For classification tasks with more than two labels (e.g. MNIST with $10$ labels), why is cross-entropy a better loss function than MSE?

<details><summary><b>Answer</b></summary>
    
</details>

---


Q. Consider a language with an alphabet of $27$ characters. What would be the maximal entropy of this language?

<details><summary><b>Answer</b></summary>
    
</details>

---

Q. Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than $10%$ from the actual price. Which metric would you use?

<details><summary><b>Answer</b></summary>
    
</details>

---


## Clustering Metrics


## Metrics in NLP