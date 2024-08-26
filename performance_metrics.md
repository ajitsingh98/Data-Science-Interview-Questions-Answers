# Data Science Interview Questions And Answers

Topics
---

- [Regression Metrics](#linear-regression)
- [Classification Metrics](#ridge-and-lasso-regularization)
- [Clustering Metrics]()
- [Metrics in NLP]()

## Regression Metrics

1. What is R-squared (R2)?

<details><summary><b>Answer</b></summary>



</details>

---

2. What does an R-squared value of 0.75 mean?

<details><summary><b>Answer</b></summary>



</details>

---


3. What is the range of possible values for R-squared?

<details><summary><b>Answer</b></summary>



</details>

---


4. Can R-squared be negative?

<details><summary><b>Answer</b></summary>



</details>

---


5. How do you interpret R-squared?

<details><summary><b>Answer</b></summary>



</details>

---


6. What are the limitations of R-squared?

<details><summary><b>Answer</b></summary>



</details>

---


7. What is the difference between adjusted R-squared and R-squared?

<details><summary><b>Answer</b></summary>



</details>

---


8. When should you use R-squared as an evaluation metric?

<details><summary><b>Answer</b></summary>



</details>

---


9. What is Mean Square Error (MSE)?

<details><summary><b>Answer</b></summary>



</details>

---


10. Why do we square the differences in MSE?

<details><summary><b>Answer</b></summary>



</details>

---


11. What is the significance of a low MSE value?

<details><summary><b>Answer</b></summary>



</details>

---


12. Can MSE be used for classification problems?

<details><summary><b>Answer</b></summary>



</details>

---


13. What are the limitations of MSE?

<details><summary><b>Answer</b></summary>



</details>

---


14. How can you minimize MSE in a machine learning model?

<details><summary><b>Answer</b></summary>



</details>

---


15. What is the difference between MSE and RMSE (Root Mean Square Error)?

<details><summary><b>Answer</b></summary>



</details>

---



## Classification Metrics

1. What is the purpose of a confusion matrix?

<details><summary><b>Answer</b></summary>



</details>

---


2. Explain True Positive (TP) and False Positive (FP).

<details><summary><b>Answer</b></summary>



</details>

---


3. Define True Negative (TN) and False Negative (FN).

<details><summary><b>Answer</b></summary>



</details>

---


4. What is accuracy, and how is it calculated using a confusion matrix?

<details><summary><b>Answer</b></summary>



</details>

---


5. What are precision and recall, and how are they calculated from a confusion matrix?

<details><summary><b>Answer</b></summary>



</details>

---


6. How can you use a confusion matrix to choose an appropriate threshold for a binary classifier?

<details><summary><b>Answer</b></summary>



</details>

---


7. What is the F1 score, and how is it related to precision and recall?

<details><summary><b>Answer</b></summary>



</details>

---


8. Explain the difference between Type I and Type II errors in the context of a confusion matrix.

<details><summary><b>Answer</b></summary>



</details>

---

7. F1 score.
    1. What’s the benefit of F1 over the accuracy?
    1. Can we still use F1 for a problem with more than two classes. How?

<details><summary><b>Answer</b></summary>
    
</details>

---

6. Your team is building a system to aid doctors in predicting whether a patient has cancer or not from their X-ray scan. Your colleague announces that the problem is solved now that they’ve built a system that can predict with 99.99% accuracy. How would you respond to that claim?

<details><summary><b>Answer</b></summary>
    
</details>

---

8. Given a binary classifier that outputs the following confusion matrix.

    $$
    \begin{bmatrix} 
        "" & \textbf{Predicted True} & \textbf{Predicted False} \\
        \textbf{Actual True} & 30 & 20\\
        \textbf{Actual False} & 5 & 40 \\
        \end{bmatrix}
    $$

    1. Calculate the model’s precision, recall, and F1.
    1. What can we do to improve the model’s performance?

<details><summary><b>Answer</b></summary>
    
</details>

---

9. Consider a classification where $99%$ of data belongs to class A and $1%$ of data belongs to class B.
    1. If your model predicts A 100% of the time, what would the F1 score be? Hint: The F1 score when A is mapped to 0 and B to 1 is different from the F1 score when A is mapped to 1 and B to 0.
    1. If we have a model that predicts A and B at a random (uniformly), what would the expected $F_1$ be?

<details><summary><b>Answer</b></summary>
    
</details>

---

11. When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?

<details><summary><b>Answer</b></summary>
    
</details>

---

12. Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.

<details><summary><b>Answer</b></summary>
    
</details>

---

13. For classification tasks with more than two labels (e.g. MNIST with $10$ labels), why is cross-entropy a better loss function than MSE?

<details><summary><b>Answer</b></summary>
    
</details>

---


14. Consider a language with an alphabet of $27$ characters. What would be the maximal entropy of this language?

<details><summary><b>Answer</b></summary>
    
</details>

---

17. Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than $10%$ from the actual price. Which metric would you use?

<details><summary><b>Answer</b></summary>
    
</details>

---