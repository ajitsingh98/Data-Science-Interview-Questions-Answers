# Sequence Modelling Interview Questions and Answers

## Table of Contents
- [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks-rnns)
- [Gated Architectures (LSTM & GRU)](#gated-architectures-lstm--gru)
- [Sequence-to-Sequence & Attention Mechanisms](#sequence-to-sequence--attention-mechanisms)
- [Transformers & Self-Attention](#transformers--self-attention)
- [Training Dynamics & Challenges](#training-dynamics--challenges)

---

## Recurrent Neural Networks (RNNs)

### Q: What is the primary motivation for using Recurrent Neural Networks (RNNs) over standard Feedforward Neural Networks for sequential data?

<details>
<summary><b>💡 Show Answer</b></summary>

Feedforward Neural Networks (FNNs) assume all inputs and outputs are independent of each other and require fixed-length input vectors. In contrast, sequential data (such as text, time-series, or audio) has variable length and strong temporal dependencies.

RNNs address this by maintaining an internal **hidden state** $h_t$ that acts as a memory of past inputs:

$$

h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)

$$

This allows parameter sharing across time steps, enabling the network to process sequences of arbitrary length while capturing temporal context.

</details>

---

### Q: What are the main limitations of basic (vanilla) RNNs?

<details>
<summary><b>💡 Show Answer</b></summary>

1. **Vanishing and Exploding Gradients**: During Backpropagation Through Time (BPTT), gradients multiplied repeatedly across many time steps exponentially decay to zero or explode to infinity.
2. **Short-Term Memory**: Due to vanishing gradients, vanilla RNNs struggle to retain context over long-range dependencies (> 10-20 steps).
3. **Sequential Computation**: Processing step $t$ requires step $t-1$, preventing parallelization across time during training.

</details>

---

## Gated Architectures (LSTM & GRU)

### Q: Explain the architecture of a Long Short-Term Memory (LSTM) network and how it solves vanishing gradients.

<details>
<summary><b>💡 Show Answer</b></summary>

LSTMs introduce a **Cell State** ($C_t$), which acts as an information highway with minimal linear interactions, controlled by three specialized gating mechanisms:

1. **Forget Gate ($f_t$)**: Decides what information to discard from the previous cell state.

$$

f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)

$$

2. **Input Gate ($i_t $) & Candidate State ($\tilde{C}_t$)**: Decides what new information to store in the cell state.
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$3. **Cell State Update**:

$$

C_t = f_t * C_{t-1} + i_t * \tilde{C}_t

$$

4. **Output Gate ($o_t $)**: Controls what part of the cell state is emitted as the hidden state$ h_t$.
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$

h_t = o_t * \tanh(C_t)

$$

Because the cell state update uses additive gradients ($\frac{\partial C_t}{\partial C_{t-1}} \approx f_t$), gradients can flow back uninterrupted over long sequences without exponentially decaying.

</details>

---

### Q: Compare LSTMs and Gated Recurrent Units (GRUs). What are the trade-offs?

<details>
<summary><b>💡 Show Answer</b></summary>

| Feature | LSTM | GRU |
| :--- | :--- | :--- |
| **Gates** | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **States** | Separate Cell State ($C_t $) & Hidden State ($ h_t $) | Single Hidden State ($ h_t$) |
| **Parameters** | More ($4 \times$ weight matrices) | Fewer ($3 \times$  weight matrices) |
| **Training Speed** | Slower | Faster |
| **Data Efficiency** | Performs better on very large datasets | Often matches LSTM performance on smaller datasets |

</details>

---

## Sequence-to-Sequence & Attention Mechanisms

### Q: Describe the Encoder-Decoder (Seq2Seq) architecture and its bottleneck.

<details>
<summary><b>💡 Show Answer</b></summary>

The Encoder-Decoder architecture consists of:
1. **Encoder**: An RNN/LSTM that reads the input sequence $X = (x_1, \dots, x_T)$ step-by-step and compresses it into a single fixed-length context vector $c = h_T$.
2. **Decoder**: An RNN/LSTM initialized with context vector $c$ that generates the target sequence $Y = (y_1, \dots, y_{T'})$  auto-regressively.

**Bottleneck**: Compressing a long input sentence (e.g., 50+ words) into a single fixed-size vector $c$  creates an information bottleneck, severely degrading performance for long sentences.

</details>

---

### Q: How does Bahdanau (Additive) Attention solve the Seq2Seq bottleneck?

<details>
<summary><b>💡 Show Answer</b></summary>

Instead of relying on a single static context vector $c $, Attention allows the decoder to dynamically dynamically look back at all encoder hidden states$(h_1, \dots, h_T)$ at each decoding step $i$:

1. Compute alignment scores $e_{ij}$ between decoder state $s_{i-1}$ and encoder state $h_j$:

$$

e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)

$$

2. Normalize with Softmax to get attention weights $\alpha_{ij}$:

$$

\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^T \exp(e_{ik})}

$$

3. Compute dynamic context vector $c_i$:

$$

c_i = \sum_{j=1}^T \alpha_{ij} h_j

$$

</details>

---

## Transformers & Self-Attention

### Q: What is Scaled Dot-Product Attention in Transformers, and why is scaling factor $\sqrt{d_k}$  necessary?

<details>
<summary><b>💡 Show Answer</b></summary>

Self-attention computes representations by comparing Queries ($Q $), Keys ($ K $), and Values ($ V$):

$$

\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V

$$

**Why scale by $\sqrt{d_k}$?**
For large dimensionality $d_k $, dot products$ Q K^T $grow large in magnitude. Large values push the softmax function into regions with extremely small gradients (vanishing gradient problem during backpropagation). Dividing by$\sqrt{d_k}$  normalizes the variance to 1.

</details>

---

### Q: Why do Transformers require Positional Encodings?

<details>
<summary><b>💡 Show Answer</b></summary>

Unlike RNNs which process tokens sequentially, Self-Attention is permutation invariant: it treats input sequence as an unordered set of vectors.

To inject word order, Transformers add **Positional Encodings** $PE_{(pos, 2i)}$  to input embeddings using sinusoidal functions:

$$

PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)

$$

$$

PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)

$$

This allows the model to easily learn relative positions because $PE_{pos + k}$ can be expressed as a linear function of $PE_{pos}$.

</details>

---

## Training Dynamics & Challenges

### Q: What is Teacher Forcing in sequence training, and what is Exposure Bias?

<details>
<summary><b>💡 Show Answer</b></summary>

- **Teacher Forcing**: A training technique for auto-regressive decoders where ground-truth tokens from the dataset are fed as input to the next time step instead of model predictions from the previous step. It accelerates convergence.
- **Exposure Bias**: During inference, ground-truth targets are unavailable, so the model relies on its own past predictions. If it makes an error early on, errors accumulate. This discrepancy between training (teacher forcing) and inference (free generation) is known as Exposure Bias.

</details>

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)
