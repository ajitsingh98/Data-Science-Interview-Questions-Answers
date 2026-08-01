# Machine Learning System Design Interview Questions and Answers

## Table of Contents
- [System Design Framework & Fundamentals](#system-design-framework--fundamentals)
- [Problem Formulation & Metrics](#problem-formulation--metrics)
- [Data Pipeline & Feature Engineering](#data-pipeline--feature-engineering)
- [Model Design & Selection](#model-design--selection)
- [Scalability & Real-Time Serving](#scalability--real-time-serving)
- [System Monitoring & Maintenance](#system-monitoring--maintenance)
- [End-to-End Case Studies](#end-to-end-case-studies)

---

## System Design Framework & Fundamentals


### Q: What is the step-by-step framework to approach a Machine Learning System Design interview?

<details>
<summary><b>💡 Show Answer</b></summary>

1. **Clarify Requirements & Constraints**:
   - Functional goals (e.g., recommend products, detect fraud).
   - Business metrics (e.g., CTR, conversion rate, latency budget < 100ms, throughput 10k QPS).
2. **Problem Formulation**:
   - Map business goal to ML task (Binary classification, Multi-class, Ranking, Regression).
   - Define inputs, outputs, and offline/online evaluation metrics.
3. **Data Pipeline & Engineering**:
   - Data collection, labelling strategies, sampling techniques.
   - Batch vs Streaming pipelines (Kafka, Flink, Spark), feature store design.
4. **Model Architecture & Training**:
   - Baseline models vs advanced architectures (Deep & Cross Networks, Two-Tower models).
   - Training loss functions, hyperparameter tuning, distributed training strategy.
5. **Serving & Deployment**:
   - Real-time online serving vs batch pre-computation.
   - Model compression (quantization, pruning, distillation), Caching layer (Redis).
6. **Monitoring, Failure Recovery & Retraining**:
   - Data drift & Concept drift monitoring, feedback loops, fallback mechanisms.

</details>

---

## Problem Formulation & Metrics


### Q: How do you choose between Batch Inference vs Real-Time Online Inference?

<details>
<summary><b>💡 Show Answer</b></summary>

| Criteria | Batch Inference | Real-Time Online Inference |
| :--- | :--- | :--- |
| **Latency** | Pre-computed (ms lookup) | On-demand computation (50-200ms) |
| **Context Freshness** | Low (updated hourly/daily) | High (uses real-time user context) |
| **Cost & Complexity** | Lower cost, simpler setup | Higher infrastructure cost & operational complexity |
| **Best Used For** | Daily movie recommendations, churn prediction | Search ranking, ad CTR prediction, fraud detection |

</details>

---

## Data Pipeline & Feature Engineering


### Q: How do you prevent Data Leakage between feature stores, training data, and online serving?

<details>
<summary><b>💡 Show Answer</b></summary>

1. **Point-in-Time Correct Joins (Time-Travel)**: When constructing training sets from historical events, ensure features for event time $t$only include data recorded strictly before$t$.
2. **Centralized Feature Store**: Use a Feature Store (e.g., Feast, Tecton) that guarantees feature definition parity between offline training and online serving key-value stores.
3. **Strict Pipeline Isolation**: Fit all transformations (scalers, encoders, target imputation) strictly on training split folds before applying to validation/test splits.

</details>

---

## Model Design & Selection


### Q: Explain the Two-Tower Architecture for Large-Scale Candidate Generation (Retrieval).

<details>
<summary><b>💡 Show Answer</b></summary>

In systems with millions of items (e.g., YouTube, E-commerce):
1. **User Tower**: Deep Neural Network mapping user context/history features to a user embedding vector $\vec{u} \in \mathbb{R}^d$.
2. **Item Tower**: Deep Neural Network mapping item features to an item embedding vector $\vec{v} \in \mathbb{R}^d$.
3. **Scoring & Approximate Nearest Neighbor (ANN)**:
   - Similarity is dot product $\langle \vec{u}, \vec{v} \rangle$.
   - Item embeddings are pre-indexed offline in an ANN vector database (Faiss, HNSW).
   - At runtime, query vector $\vec{u}$ retrieves top-100 candidates in < 5ms.

</details>

---

## Scalability & Real-Time Serving


### Q: What techniques reduce deep learning model latency for online real-time serving (< 20ms QPS)?

<details>
<summary><b>💡 Show Answer</b></summary>

1. **Quantization**: Convert FP32 model weights to INT8 precision (up to $4\times$footprint reduction,$2-3\times$ speedup with minimal accuracy loss).
2. **Knowledge Distillation**: Train a compact "Student" network to mimic outputs of an ensemble "Teacher".
3. **Graph Optimizations & Engines**: Use ONNX Runtime or TensorRT to fuse operations (e.g., Conv + ReLU fusion).
4. **Multi-Tier Caching**: Cache top query results in Redis/Memcached.

</details>

---

## System Monitoring & Maintenance


### Q: How do you detect and handle Concept Drift and Covariate Shift in production ML systems?

<details>
<summary><b>💡 Show Answer</b></summary>

- **Covariate Shift**: $P(X)$changes while$P(Y|X)$ remains constant.
  - *Detection*: Compare input feature distributions over time using Kolmogorov-Smirnov (KS) test or Population Stability Index (PSI).
- **Concept Drift**: $P(Y|X)$ changes (e.g., user preferences shift post-pandemic).
  - *Detection*: Monitor online metrics (CTR, conversion rate, precision@k) and compare prediction probability distributions vs true feedback.
- **Remediation**:
  - Automated continuous retraining pipelines (airflow/KubeFlow) triggered by PSI threshold breaches.
  - Fallback to rule-based heuristics if model metrics degrade past critical guardrails.

</details>

---

## End-to-End Case Studies


### Q: Design an End-to-End E-Commerce Product Recommendation System.

<details>
<summary><b>💡 Show Answer</b></summary>

1. **Stage 1: Retrieval (Candidate Generation)**
   - Input: User ID, recent views, location.
   - Model: Two-Tower Neural Network / Matrix Factorization retrieving Top 500 items from 10M catalog in ~10ms via FAISS.
2. **Stage 2: Ranking**
   - Model: Deep & Cross Network (DCN-v2) scoring candidate items using rich interactions (user-category history, price sensitivity).
   - Loss: Multi-task learning (pCTR $\times$ pCVR).
3. **Stage 3: Re-ranking & Business Logic**
   - Diversity filtering, out-of-stock exclusion, freshness boosting, sponsored ad insertion.
4. **Infrastructure**:
   - Kafka for clickstream streaming, Flink for real-time feature aggregation, Feast Feature Store, Triton Inference Server.

</details>

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)
