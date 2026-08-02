# Natural Language Processing Interview Questions and Answers

## Table of Contents
- [RNNs, LSTMs & Language Models](#rnns-lstms--language-models)
- [Density Estimation & Training Paradigms](#density-estimation--training-paradigms)
- [Word Embeddings](#word-embeddings)
- [TF-IDF & Cosine Similarity](#tf-idf--cosine-similarity)
- [N-Gram Language Models](#n-gram-language-models)

---

## RNNs, LSTMs & Language Models

### Q: What is the primary motivation for RNNs and LSTMs in NLP? How do you apply dropout in RNNs?

<details>
<summary><b>💡 Show Answer</b></summary>

- **RNN Motivation**: Natural language inputs have dynamic lengths and sequential dependencies. Standard feedforward networks cannot handle variable lengths or preserve temporal context.
- **LSTM Motivation**: Recurrent networks suffer from vanishing/exploding gradients. LSTMs introduce a gated Cell State ($C_t$) that enables additive gradient flow over long sequences.
- **Dropout in RNNs**: Standard dropout applied to recurrent connections corrupts sequence memory over time. Techniques like **Variational Dropout** (Gal & Ghahramani) apply the same dropout mask across all time steps, or apply dropout only to non-recurrent feedforward connections between layers (Zaremba et al.).

</details>

---

## Density Estimation & Training Paradigms

### Q: What is Density Estimation? Why is a Language Model considered a Density Estimator?

<details>
<summary><b>💡 Show Answer</b></summary>

- **Density Estimation**: The task of estimating the probability distribution $P(X)$ from observed sample data $X$.
- **LM as Density Estimator**: A language model models the joint probability distribution of sequences of words $W = (w_1, w_2, \dots, w_T)$  using the chain rule of probability:


$$P(W) = \prod_{t=1}^{T} P(w_t \mid w_1, w_2, \dots, w_{t-1})$$


  It assigns probabilities to token sequences, identifying fluent vs ungrammatical sequences.

</details>

---

### Q: Language models are often called unsupervised, but some argue their mechanism is self-supervised/supervised. What are your thoughts?

<details>
<summary><b>💡 Show Answer</b></summary>

Language modeling is best described as **Self-Supervised Learning**:
- **Data Source**: It uses raw, unlabeled text without human manual annotations (which resembles unsupervised learning).
- **Learning Objective**: It formulates a supervised predictive objective (e.g., predicting next token $w_t$ given context $w_{<t}$) using standard cross-entropy loss.
- Thus, supervision signals are automatically derived from the data itself.

</details>

---

## Word Embeddings

### Q: Why do we need Word Embeddings? Compare Count-Based vs Prediction-Based Embeddings.

<details>
<summary><b>💡 Show Answer</b></summary>

- **Why Embeddings?**: One-hot encodings are high-dimensional, sparse, and orthogonal (do not capture semantic similarity between words like "cat" and "kitten"). Dense embeddings project words into a continuous vector space where distance reflects semantic relationship.
- **Count-Based vs Prediction-Based**:
  - *Count-Based* (TF-IDF, LSA, GloVe): Computes co-occurrence statistics across a corpus and performs matrix factorization. Fast to compute globally.
  - *Prediction-Based* (Word2Vec CBOW/Skip-Gram): Trains a shallow neural network to predict target words given context (or vice versa). Better at capturing complex analogical relationships ($\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$).

</details>

---

## TF-IDF & Cosine Similarity

### Q: Given 5 documents and Query Q: "The early bird gets the worm", how does TF-IDF rank document relevance?

<details>
<summary><b>💡 Show Answer</b></summary>

1. **Term Frequency ($\text{TF}_{t,d}$)**: Number of occurrences of term$ t $in document$ d$.
2. **Inverse Document Frequency ($\text{IDF}_t$)**: Downweights terms that appear frequently across all documents:


$$\text{IDF}_t = \log\left(\frac{N}{\text{DF}_t}\right)$$


3. **Cosine Similarity**: Measures directional alignment between query vector $\vec{q}$ and document vector $\vec{d}$:


$$\text{Sim}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$


Rare, informative words (e.g., "worm", "early") contribute heavily to TF-IDF score, while common stop words ("the") are penalized by low IDF.

</details>

---

## N-Gram Language Models

### Q: Should you choose an N-Gram or Neural Language Model for a tiny dataset (~10,000 tokens)?

<details>
<summary><b>💡 Show Answer</b></summary>

For a tiny dataset of 10,000 tokens:
- **N-Gram with Smoothing** (e.g., Kneser-Ney smoothing) is preferred.
- Neural Language Models have thousands to millions of parameters and will severely overfit on only 10k tokens without pre-trained embeddings or heavy regularization.

</details>

---

### Q: Does increasing context length $N$  in N-gram models always improve performance?

<details>
<summary><b>💡 Show Answer</b></summary>

No:
1. **Data Sparsity**: As $N$ increases, the probability of seeing specific $N$-gram sequences in training data drops sharply (zero-frequency problem).
2. **Memory Overhead**: Storage grows exponentially $O(|V|^N)$ with vocabulary size $|V|$.
3. **Diminishing Returns**: Without backoff/smoothing, long contexts cause high variance and poor generalization.

</details>

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)
