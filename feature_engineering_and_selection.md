# Feature Engineering & Selection Interview Questions

## Table of Contents
- [Categorical Encoding Techniques](#categorical-encoding-techniques)
- [Numerical Scaling & Transformation](#numerical-scaling--transformation)
- [Missing Data Imputation](#missing-data-imputation)
- [Feature Selection Algorithms](#feature-selection-algorithms)
- [Explainability (SHAP & LIME)](#explainability-shap--lime)

---

## Categorical Encoding Techniques

### Q: Compare Target Encoding vs One-Hot Encoding. How do you prevent Target Leakage?
<details>
<summary><b>💡 Show Answer</b></summary>

- **One-Hot Encoding**: Creates binary columns per category. High memory overhead for high-cardinality features.
- **Target Encoding**: Replaces category $c $ with mean target value$\mathbb{E}[y \mid category = c]$.
- **Leakage Prevention**: Compute target statistics strictly out-of-fold using K-Fold cross-validation with smoothing/additive noise.

</details>

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)
