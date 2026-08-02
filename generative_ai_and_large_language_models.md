# Generative AI & Large Language Models (LLMs) Interview Questions

> 🎯 **Data Science Interview Questions & Answers** — Part of the [complete interview prep series](./README.md)

## Table of Contents
- [LLM Architectures & Foundations](#llm-architectures--foundations)
- [Tokenization & Embeddings](#tokenization--embeddings)
- [Attention, KV Cache & Efficient Inference](#attention-kv-cache--efficient-inference)
- [Decoding Strategies & Prompt Engineering](#decoding-strategies--prompt-engineering)
- [Fine-Tuning Techniques (LoRA, QLoRA, PEFT)](#fine-tuning-techniques-lora-qlora-peft)
- [Retrieval-Augmented Generation (RAG) & Vector Databases](#retrieval-augmented-generation-rag--vector-databases)
- [Alignment & Preference Optimization (RLHF, DPO)](#alignment--preference-optimization-rlhf-dpo)
- [Agents, Memory & Production](#agents-memory--production)

---

## LLM Architectures & Foundations

### Q: What defines a Large Language Model (LLM)?

<details>
<summary><b>💡 Show Answer</b></summary>

LLMs are AI systems trained on vast text corpora to understand and generate human-like language. With billions of parameters, they excel in tasks like translation, summarization, and question answering, leveraging contextual learning for broad applicability.

</details>

---

### Q: What is a Language Model?

<details>
<summary><b>💡 Show Answer</b></summary>

A language model is a probability distribution over sequences of tokens (words, characters, subwords, etc.). Given a finite vocabulary $\mathcal{V}$, it assigns a probability to a sequence $x_1, x_2, \ldots, x_L \in \mathcal{V}$:


$$p(x_1, x_2, \ldots, x_L)$$


This probability reflects how likely the sequence is to occur in natural language.

</details>

---

### Q: What are autoregressive language models, and how do they use the chain rule?

<details>
<summary><b>💡 Show Answer</b></summary>

Autoregressive language models generate or score sequences by modeling the joint probability as a product of conditionals — predicting each token from its own past outputs:


$$p(x_1, \ldots, x_L) = \prod_{i=1}^{L} p(x_i \mid x_1, \ldots, x_{i-1})$$


Each term $p(x_i \mid x_{<i})$ is the probability of the next token given all previous tokens. This is the standard training objective for GPT-style decoder-only models.

</details>

---

### Q: What is the core difference between Encoder-only (BERT), Decoder-only (GPT/Llama), and Encoder-Decoder (T5) architectures?

<details>
<summary><b>💡 Show Answer</b></summary>

- **Encoder-Only (BERT, RoBERTa)**: Bidirectional self-attention. Produces contextual embeddings for NLU tasks (classification, NER, sentence embeddings). Cannot natively generate completions; needs ad-hoc objectives (e.g., MLM).
- **Decoder-Only (GPT, Llama, Mistral)**: Causal (left-to-right) masked self-attention. Naturally generates completions with a simple MLE next-token objective. Contextual embeddings for $x_i$ depend only on left context $x_{<i}$.
- **Encoder-Decoder (T5, BART)**: Bidirectional encoder + cross-attention decoder. Ideal for sequence-to-sequence tasks (translation, summarization). Supports bidirectional input context and generative output, but training objectives are more specialized.

</details>

---

### Q: What is the difference between proprietary and open-source Large Language Models (LLMs)?

<details>
<summary><b>💡 Show Answer</b></summary>

Proprietary LLMs like GPT-4 and Claude are closed-source models accessed via APIs. They are easy to use, require no local computing power, and offer high performance but come with costs, limited customization, and data privacy concerns.

Open-source LLMs, such as LLaMA and Mistral, provide full access to model weights and architecture, allowing local use, fine-tuning, and better data control. However, they require powerful hardware and technical expertise to run and manage.

</details>

---

### Q: How are Large Language Models (LLMs) trained? What are the main stages?

<details>
<summary><b>💡 Show Answer</b></summary>

There are typically three stages that produce a high-quality aligned LLM:

1. **Pretraining**: Next-token prediction on massive internet-scale text → a general-purpose foundation model that learns grammar, facts, and patterns.
2. **Supervised Fine-Tuning (SFT)**: Train on curated instruction–response pairs so the model follows instructions and performs tasks.
3. **Preference Tuning**: Align outputs with human preferences via RLHF (reward model + PPO) or DPO.

Pretraining is by far the most compute-heavy; fine-tuning and preference tuning are comparatively cheaper adaptations for real-world use.

</details>

---

### Q: What is the context window in LLMs, and why does it matter?

<details>
<summary><b>💡 Show Answer</b></summary>

The context window refers to the maximum number of tokens an LLM can process at once, defining its “memory” for understanding or generating text. A larger window, like 32,000 tokens, allows the model to consider more context, improving coherence in tasks like summarization. However, it increases computational costs. Balancing window size with efficiency is crucial for practical LLM deployment.

</details>

---

### Q: How do autoregressive and masked models differ in LLM training?

<details>
<summary><b>💡 Show Answer</b></summary>

Autoregressive models, like GPT, predict tokens sequentially based on prior tokens, excelling in generation tasks such as text completion. Masked models like BERT predict masked tokens using bidirectional context, making them ideal for understanding tasks like classification. Their training objectives shape their strengths in generative vs comprehensive.

</details>

---

### Q: What is the difference between representation models and generative models?

<details>
<summary><b>💡 Show Answer</b></summary>

Representation models mainly focus on representing language, for example creating embeddings and typically do not generate texts. In contrast, generative models focus primarily on generating text and typically are not trained to generate embeddings.

</details>

---

### Q: What are emergent properties of LLMs?

<details>
<summary><b>💡 Show Answer</b></summary>

Emergent properties are abilities that appear as models scale, even though training only optimized next-token prediction. Pretrained LLMs can often summarize, translate, answer questions, classify, and reason — skills not explicitly labeled during pretraining. Emergence is typically discussed as capabilities that become reliable only past certain scale thresholds.

</details>

---

### Q: How does Mixture of Experts (MoE) enhance LLM scalability?

<details>
<summary><b>💡 Show Answer</b></summary>

MoE uses a gating function to activate specific expert sub-networks per input, reducing computational load. For example, only *10%* of a model's parameters might be used per query, enabling billion-parameter models to operate efficiently while maintaining high performance.

</details>

---

### Q: What challenges do LLMs face in deployment?

<details>
<summary><b>💡 Show Answer</b></summary>

LLM challenges include:

- Resource Intensity: High computational demands.
- Bias: Risk of perpetuating training data biases.
- Interpretability: Complex models are hard to explain.
- Privacy: Potential data security concerns.

Addressing these ensures ethical and effective LLMs are.

</details>

---

## Tokenization & Embeddings

### Q: What does tokenization entail, and why is it critical for LLMs?

<details>
<summary><b>💡 Show Answer</b></summary>

Tokenization involves breaking down text into smaller units, or tokens, such as words, subwords, or characters. For example, “artificial” might be split into “art”, “ific” and “ial”. This process is vital because LLMs process numerical representations of tokens, not raw text. Tokenization enables models to handle diverse languages, manage rare or unknown words, and optimize vocabulary size, enhancing computational efficiency and model performance.

</details>

---

### Q: What are the different parameters of a tokenizer?

<details>
<summary><b>💡 Show Answer</b></summary>

1. Vocabulary size: How many tokens to keep in tokenizer’s vocabulary?
2. Special tokens: What special tokens do we want the model to keep track of? We can add as many of these as we want, especially if we want to build an LLM for special use cases. Common choices include:
   1. Beginning of text token (e.g., <s>)
   2. End of text token
   3. Padding token
   4. Unknown token
   5. CLS token
   6. Masking token
3. Capitalization: Case-sensitive vs case-insensitive

</details>

---

### Q: What is Byte-Pair Encoding (BPE) and how does it work?

<details>
<summary><b>💡 Show Answer</b></summary>

Byte-Pair Encoding is a subword tokenization technique that builds a fixed-size vocabulary by repeatedly merging the most frequent adjacent symbol pairs in a corpus.

1. **Pre-tokenize**: Split text into words (e.g., by whitespace).
2. **Initialize**: Break every word into characters; count word frequencies.
3. **Merge loop**:
   - Count all adjacent symbol-pair frequencies.
   - Merge the most frequent pair into a new token and add it to the vocabulary.
   - Repeat until the target vocabulary size is reached.
4. **Tokenize new text**: Apply merges greedily (longest match first); unseen symbols map to `<unk>`.

BPE lets models handle rare/OOV words by composing known subwords (e.g., “cryptocurrency” → “crypto” + “currency”).

</details>

---

### Q: Illustrate with a simple example how BPE works.

<details>
<summary><b>💡 Show Answer</b></summary>

Corpus (word, frequency): `("hug",10), ("pug",5), ("pun",12), ("bun",4), ("hugs",5)`

1. Start with character base: `["h","u","g","p","n","b","s"]`
2. Merge most frequent pairs iteratively:
   - `"u"+"g"` occurs 20× → add `"ug"`
   - `"u"+"n"` occurs 16× → add `"un"`
   - `"h"+"ug"` occurs 15× → add `"hug"`
3. Resulting vocab (example stop): `["h","u","g","p","n","b","s","ug","un","hug"]`
4. New words: `"bug"` → `["b","ug"]`; `"mug"` → `["<unk>","ug"]` if `"m"` was never seen.

</details>

---

### Q: What are positional encodings, and why are they used? What types exist?

<details>
<summary><b>💡 Show Answer</b></summary>

Self-attention is permutation-invariant, so transformers need positional information added to token embeddings.

Main types:
- **Absolute — Sinusoidal**: Fixed sine/cosine waves by position; no extra parameters; can extrapolate somewhat.
- **Absolute — Learned**: Trainable vector per position; flexible but limited to max training length.
- **Relative**: Encodes distance between query and key rather than absolute index; better for variable lengths and local patterns.
- **Rotary (RoPE)**: Rotates query/key vectors by token index, injecting absolute and relative info without added parameters. Used in LLaMA, Mistral, and many modern LLMs.

</details>

---

## Attention, KV Cache & Efficient Inference

### Q: How does the attention mechanism function in transformer models?

<details>
<summary><b>💡 Show Answer</b></summary>

Attention lets the model weigh how much each token should attend to others. Input embeddings are projected into **Queries (Q)**, **Keys (K)**, and **Values (V)**.

1. **Relevance scoring**: For position $t$, compute $Q_t$ and take dot products with all keys to get similarity scores.
2. **Softmax**: Convert scores to attention weights that sum to 1.
3. **Weighted sum**: Multiply weights by value vectors to form a context vector for that position.

Scaled dot-product attention:


$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$$


Multi-head attention runs this in parallel subspaces (e.g., one head on syntax, another on semantics) and concatenates the results.

</details>

---

### Q: What is the KV cache in transformer models? What are its trade-offs?

<details>
<summary><b>💡 Show Answer</b></summary>

The **KV cache** speeds up autoregressive generation. Instead of recomputing keys/values for all past tokens at every step, the model stores previous $K$ and $V$ tensors. For each new token it only computes the latest key/value and attends over the cached history.

- **Advantage**: Without caching, cumulative attention cost is $O(n^2)$; with KV cache, per-step work is $O(n)$, which greatly speeds long generations.
- **Disadvantage**: Memory grows linearly with sequence length (and with layers × heads × dim). Long contexts can exhaust GPU memory; mitigations include cache truncation, quantization, and paging (e.g., PagedAttention).

</details>

---

### Q: What is Flash Attention?

<details>
<summary><b>💡 Show Answer</b></summary>

Flash Attention is a GPU‑optimized implementation of the Transformer attention mechanism. It minimizes data movement between high‑bandwidth memory (HBM) and on‑chip shared memory (SRAM) by reordering and fusing operations, delivering significant speedups and memory savings during both training and inference.

</details>

---

### Q: What is local (sparse) attention and how does it work?

<details>
<summary><b>💡 Show Answer</b></summary>

Local/sparse attention restricts each token’s context to a limited neighborhood instead of full $O(n^2)$ attention. Examples:
- **Sliding window** (Longformer): each token attends to a fixed window (e.g., ±128).
- **Block-sparse** (Sparse Transformer): attend within blocks and optionally to selected distant blocks.

This reduces compute for long sequences. Some models alternate full-attention and sparse blocks to keep global dependencies while staying efficient.

</details>

---

### Q: What is model distillation, and how does it benefit LLMs?

<details>
<summary><b>💡 Show Answer</b></summary>

Model distillation trains a smaller “student” model to mimic a larger “teacher” model's outputs, using soft probabilities rather than hard labels. This reduces memory and computational requirements, enabling deployment on devices like smartphones while retaining near-teacher performance, ideal for real-time applications.

</details>

---

### Q: What distinguishes LoRA from QLoRA in the context of memory-efficient fine-tuning / quantization?

<details>
<summary><b>💡 Show Answer</b></summary>

- **LoRA**: Freezes base weights and trains low-rank adapters $\Delta W = BA$ (rank $r \ll d$), cutting trainable parameters dramatically.
- **QLoRA**: First quantizes the frozen base model to 4-bit (e.g., NormalFloat / blockwise quantization), then applies LoRA adapters in higher precision. This can fine-tune ~70B models on a single GPU with near–16-bit LoRA quality at much lower memory.

</details>

---

## Decoding Strategies & Prompt Engineering

### Q: Explain the decoding strategies used in LLMs. How do greedy decoding, temperature, top-k, and top-p differ?

<details>
<summary><b>💡 Show Answer</b></summary>

LLMs output a distribution over the vocabulary; decoding chooses the next token:

- **Greedy**: Always pick $\arg\max$. Fast but often repetitive/boring.
- **Temperature $T$**: Softens/sharpens logits before sampling.
  - $T = 1$: raw model distribution
  - $T < 1$: more deterministic (favors high-probability tokens)
  - $T > 1$: more random/creative
  - $T \to 0$: approaches greedy
- **Top-k**: Sample only from the $k$ highest-probability tokens.
- **Top-p (nucleus)**: Sample from the smallest set whose cumulative probability ≥ $p$ (e.g., 0.95); adapts set size to context.
- **Beam search**: Keep top-$k$ partial sequences at each step; good for translation, less creative than sampling.

</details>

---

### Q: Why is prompt engineering crucial for LLM performance? What techniques should you know?

<details>
<summary><b>💡 Show Answer</b></summary>

Prompt design strongly affects relevance and reliability, especially in zero-/few-shot settings.

Useful techniques:
- **Specificity**: Clear constraints (length, tone, format).
- **Instruction placement**: Put critical instructions at the start or end (primacy/recency).
- **Role prompting**: “You are a data science mentor…”
- **Few-shot**: Provide input→output examples.
- **Chain-of-Thought**: Ask for step-by-step reasoning.
- **Hallucination mitigation**: “If unsure, say you don’t know.”
- **Format constraints**: Bullets, JSON schema, max length.

</details>

---

### Q: What is Chain-of-Thought (CoT) prompting, and how does it aid reasoning?

<details>
<summary><b>💡 Show Answer</b></summary>

CoT prompting guides LLMs to solve problems step-by-step, mimicking human reasoning. For example, in math problems, it breaks down calculations into logical steps, improving accuracy and interpretability in complex tasks like logical inference or multi-step queries.

</details>

---

### Q: What is self-consistency in prompting, and what are its trade-offs?

<details>
<summary><b>💡 Show Answer</b></summary>

Self-consistency is a prompting technique that involves asking the same prompt multiple times, allowing the model to generate diverse outputs by leveraging randomness through parameters like temperature and top_p. Instead of relying on a single response—which can be affected by luck or randomness—this method samples multiple outputs and selects the most common (majority-voted) answer as the final result.

Benefits:

- Reduces variability and random errors in responses.
- Improves accuracy and robustness of answers, especially for reasoning-based tasks.

Trade-offs:

- Slower inference: The process is n times slower, as it requires generating multiple responses for a single query.
- Computational overhead: More resources are consumed due to multiple forward passes.

</details>

---

### Q: What is zero-shot vs few-shot learning for LLMs?

<details>
<summary><b>💡 Show Answer</b></summary>

- **Zero-shot**: Perform a task from instructions alone, using knowledge from pretraining (e.g., “Classify this review as positive or negative”).
- **Few-shot**: Provide a handful of labeled examples in the prompt (in-context learning). Benefits: low data needs, fast adaptation, no weight updates — ideal for niche tasks when fine-tuning is unnecessary.

</details>

---

### Q: Can we use generative models for classification tasks? If so, how?

<details>
<summary><b>💡 Show Answer</b></summary>

Yes, generative models like GPT can be used for classification in two main ways:

- Prompt-based Classification (Zero/Few-shot):
  - Frame the task as text completion.
  - Example:
    - Input: "Review: I love this phone! Sentiment:"
    - Output: "positive"
- Fine-tuned Generation:
  - Fine-tune the model on input–label pairs where the label is treated as a text generation target.
  - Example:
    - Input: "News: Markets fell sharply today. Category:"
    - Output: "Economy"

</details>

---

### Q: How can we control the output of a generative model?

<details>
<summary><b>💡 Show Answer</b></summary>

Controlling the output of a generative model involves guiding it to produce consistent, structured, and format-adherent results. There are three widely used methods:

1. Providing Examples (Few-shot Prompting): By giving one or more examples in the prompt (zero-shot, one-shot, or few-shot), we guide the model toward the desired format, style, and content.

- Pros: Simple to implement; improves structure and coherence.
- Cons: No strict guarantee that the model will follow the pattern.

2. Grammar-based Constraints (Constrained Decoding): This method limits the token selection process using tools like Guidance, Guardrails, or LMQL, ensuring outputs strictly conform to a defined grammar (e.g., JSON).

- Example: Restricting sentiment output to “positive,” “neutral,” or “negative.”
- Pros: Enforces format and content-level constraints.
- Cons: Adds complexity to implementation.

3. Fine-tuning: Customizing a model using a dataset with the desired output format and tone.

- Pros: High control and reliability.
- Cons: Resource-intensive and requires model retraining.

</details>

---

## Fine-Tuning Techniques (LoRA, QLoRA, PEFT)

### Q: What is the difference between parameter-efficient fine-tuning (PEFT) and full fine-tuning? Which is better?

<details>
<summary><b>💡 Show Answer</b></summary>

- Full fine-tuning: Updates all model weights on your downstream task.
  - Pros: Maximum capacity to learn task-specific patterns.
  - Cons: High GPU/memory cost, slower, prone to overfitting, costly to store separate models per task.
- Parameter-efficient fine-tuning (e.g., adapters, LoRA, prefix-tuning): Freezes the base model and only trains a small subset of parameters (injected adapters or low-rank matrices).

  - Pros: Orders-of-magnitude lower compute and storage, faster training, easy multi-task sharing, similar accuracy to full fine-tuning.
  - Cons: Slightly lower peak performance on some tasks.

PEFT is generally preferred unless you need the absolute best single-task accuracy and have unlimited compute/storage. It delivers near–full-fine-tuning performance at a fraction of the cost.

</details>

---

### Q: Explain Low-Rank Adaptation (LoRA) and how it achieves parameter-efficient fine-tuning.

<details>
<summary><b>💡 Show Answer</b></summary>

LoRA freezes pretrained weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d,k)$:


$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}(BA)$$


- **Initialization**: $A \sim \mathcal{N}(0, \sigma I)$, $B = 0$ so $\Delta W = 0$ at start.
- **Rank choice**: Small $r$ (4, 8, 16) reflecting the intrinsic dimension of the task update; tune on validation.
- Reduces trainable parameters from $d \cdot k$ to roughly $r(d+k)$ (often >99% reduction) while matching full fine-tuning quality on many tasks.

**vs Adapters**: LoRA merges into existing weights (no extra depth); adapters insert new bottleneck modules (~3–10% params vs often <1% for LoRA).

</details>

---

### Q: What are adapters, and how do they enable PEFT?

<details>
<summary><b>💡 Show Answer</b></summary>

Adapters are lightweight, trainable modules inserted into each Transformer block typically after the attention and feed-forward layers. During PEFT, you freeze the base model’s weights and only train these adapters. This approach updates under 5% of parameters yet achieves performance within ~0.4% of full fine-tuning on benchmarks like GLUE, drastically reducing compute, memory, and storage costs.

</details>

---

### Q: What is “intrinsic dimension” in the context of model fine-tuning, and why does it matter?

<details>
<summary><b>💡 Show Answer</b></summary>

Intrinsic dimension refers to the minimal number of degrees of freedom needed to capture task-specific updates in a pretrained model. Although models have millions or billions of parameters, the actual fine-tuning changes lie on a much lower-dimensional manifold. Recognizing this lets us use parameter-efficient methods (like LoRA or adapters) that only optimize a small subspace.

</details>

---

### Q: What is QLoRA, and how does it compare to standard LoRA?

<details>
<summary><b>💡 Show Answer</b></summary>

QLoRA augments LoRA by first quantizing the pretrained model’s original weights to 4-bit precision using a distribution-aware, blockwise scheme, then applying Low-Rank Adaptation (LoRA) on top of these quantized weights.

- **LoRA:** Decomposes large weight updates into two small matrices (A, B) of rank r, reducing trainable parameters from d^2 to 2dr.
- **QLoRA:** Adds a preprocessing quantization step mapping full-precision weights into 4-bit “NormalFloat” blocks before freezing them and fine-tuning only the LoRA matrices.

By combining quantization with low-rank updates, QLoRA maintains near–full-precision accuracy while reducing memory use by up to 4× over 16-bit LoRA, enabling efficient fine-tuning of very large models on limited hardware.

</details>

---

### Q: How does PEFT mitigate catastrophic forgetting?

<details>
<summary><b>💡 Show Answer</b></summary>

Parameter-Efficient Fine-Tuning (PEFT) updates only a small subset of parameters, freezing the rest to preserve pretrained knowledge. Techniques like LoRA ensure LLMs adapt to new tasks without losing core capabilities, maintaining performance across domains.

</details>

---

### Q: How can LLMs avoid catastrophic forgetting during fine-tuning?

<details>
<summary><b>💡 Show Answer</b></summary>

Catastrophic forgetting occurs when fine-tuning erases prior knowledge. It can be mitigated by following these strategies:

- Rehearsal: Mixing old and new data during training.
- Elastic Weight Consolidation: Prioritizing critical weights to preserve knowledge.
- Modular Architectures: Adding task-specific modules to avoid overwriting.

These methods ensure LLMs retain versatility across tasks.

</details>

---

### Q: What is continued pretraining with Masked Language Modeling, and why is it useful?

<details>
<summary><b>💡 Show Answer</b></summary>

Continued pretraining inserts an extra MLM step between general pretraining and task-specific fine-tuning. You take a pretrained model (e.g. BERT), further train it on domain-specific text using the MLM objective, then fine-tune on your downstream task. This adapts subword embeddings to domain vocabulary (e.g. medical or movie reviews), improving performance without the huge cost of full pretraining.

</details>

---

### Q: When would you use RAG over fine-tuning and vice versa?

<details>
<summary><b>💡 Show Answer</b></summary>

Use RAG when:

- You need up-to-date or dynamic information
- You want to avoid costly fine-tuning
- Your data is large or frequently updated
- You need source citations or explainability
- You serve multiple domains or clients

Use Fine-tuning:

- You have task-specific, stable data
- You want faster, retrieval-free inference
- You need structured or consistent outputs
- You aim for deep customization or improved accuracy
- The task doesn’t require external knowledge access

</details>

---

## Retrieval-Augmented Generation (RAG) & Vector Databases

### Q: What are the steps in Retrieval-Augmented Generation (RAG)?

<details>
<summary><b>💡 Show Answer</b></summary>

RAG Involves:

- Retrieval: Fetching relevant documents using query embeddings.
- Ranking: Sorting documents by relevance.
- Generation: Using retrieved context to generate accurate responses.

RAG enhances factual accuracy in tasks like question answering.

</details>

---

### Q: What is Dense Retrieval and how does it work? What are its caveats?

<details>
<summary><b>💡 Show Answer</b></summary>

Dense retrieval embeds queries and documents in the same vector space so semantically similar texts are nearby. At query time, embed the query, search a vector DB (often via ANN), and return nearest document chunks.

**Caveats:**
1. May return irrelevant text if the answer isn’t in the corpus.
2. Weaker at exact phrase / keyword match than sparse retrievers (BM25).
3. Domain mismatch if the embedder wasn’t trained for that domain.
4. Hard when answers span many sentences/chunks.

</details>

---

### Q: What are the best chunking strategies in RAG systems?

<details>
<summary><b>💡 Show Answer</b></summary>

- Chunk by Semantics: Use meaningful units like paragraphs or logical sections.
- Avoid Over-Granularity: Sentence-level chunks may be too narrow to capture full context.
- Optimal Size: Use 3–8 sentences or ~256–512 tokens per chunk.
- Overlapping Chunks: Include text before/after a chunk to preserve context across boundaries.
- Add Metadata: Include titles or headings with each chunk to boost relevance.
- Dynamic Chunking: Use LLMs to split text into semantically coherent units.

</details>

---

### Q: What is ANN and how is it useful in RAG systems?

<details>
<summary><b>💡 Show Answer</b></summary>

ANN (Approximate Nearest Neighbor) is used in RAG systems to quickly retrieve the most relevant document chunks from a large vector database by finding vectors close to a query embedding. It speeds up retrieval compared to exact search, enabling low-latency, scalable semantic search with minimal loss in accuracy.

</details>

---

### Q: What is query rewriting in RAG systems, and why is it important?

<details>
<summary><b>💡 Show Answer</b></summary>

Query rewriting leverages an LLM to transform a user’s often verbose or context-dependent question into a concise, focused search query.

Why Query Rewriting Matters:

- Clarity: Users’ natural questions can include tangents or context (e.g., “We have an essay due tomorrow… Where do they live for example?”), which may confuse retrieval engines.
- Precision: A rewritten query (e.g., “Where do dolphins live”) directly targets the key information, yielding better search results.
- Context Awareness: Rewriting can incorporate prior conversational context to improve relevance without overloading the retrieval step.

</details>

---

### Q: What is Multi-query RAG? How does it differ from multi-hop RAG?

<details>
<summary><b>💡 Show Answer</b></summary>

- **Multi-query RAG**: Split one complex question into several *independent* searches, then merge results.
  - Example: “Compare Nvidia’s 2020 vs 2023 results” → “Nvidia 2020 financials” + “Nvidia 2023 financials”.
- **Multi-hop RAG**: *Sequential* searches where each hop depends on prior results.
  - Example: find largest car makers in 2023, then query each for EV offerings.

Multi-query boosts recall via parallel reformulations; multi-hop handles questions requiring compositional reasoning.

</details>

---

### Q: What is query routing in RAG systems?

<details>
<summary><b>💡 Show Answer</b></summary>

Query routing directs the search query to the most appropriate data source based on its topic or intent. For example, questions about HR policies are routed to the HR knowledge base (e.g., Notion), while questions about customer orders go to the CRM (e.g., Salesforce). This ensures that each query is answered from the most relevant repository, boosting accuracy and response quality.

</details>

---

### Q: What is Agentic RAG?

<details>
<summary><b>💡 Show Answer</b></summary>

Agentic RAG extends traditional RAG by giving the LLM more autonomy to reason, decide what information is needed, and use multiple data sources or tools. The LLM doesn't just retrieve information; it acts like an agent, capable of querying, synthesizing, and even interacting with external systems (e.g., reading from and writing to tools like Notion).

</details>

---

### Q: How do we evaluate the retrieval and generation components of a RAG system?

<details>
<summary><b>💡 Show Answer</b></summary>

**Retrieval** (over docs, queries, and relevance judgments):
- **Precision@k**: Fraction of top-$k$ results that are relevant.
- **Average Precision (AP)** / **MAP**: Rewards retrieving relevant docs early; MAP averages AP over queries.

**Generation**:
- Human ratings: fluency, utility, citation recall/precision.
- LLM-as-judge on the same axes.
- Frameworks like **Ragas**: faithfulness, answer relevance, context precision/recall.

</details>

---

### Q: How does fine-tuning embedding models improve dense retrieval?

<details>
<summary><b>💡 Show Answer</b></summary>

Fine-tuning improves dense retrieval by optimizing text embeddings based on task-specific relevance. It involves training on pairs of queries and their relevant documents, so that the model learns to place related query-document pairs closer in embedding space. The model is fine-tuned to pull relevant embeddings closer and push irrelevant ones farther, improving retrieval precision. This is typically done using contrastive loss functions like triplet loss or InfoNCE.

</details>

---

## Alignment & Preference Optimization (RLHF, DPO)

### Q: What is preference tuning, and why is it important?

<details>
<summary><b>💡 Show Answer</b></summary>

Preference tuning is a final alignment step where an LLM is optimized to produce outputs that humans prefer. You collect example prompts, have a human (or reward) model rank multiple generations, then fine-tune the LLM often via reinforcement learning (e.g., PPO) to maximize those preference scores. This ensures the model doesn’t just follow instructions but also generates responses that align with human values, styles, and priorities, improving usefulness, safety, and user satisfaction.

</details>

---

### Q: What is a reward model, and how does it automate preference evaluation in LLM alignment?

<details>
<summary><b>💡 Show Answer</b></summary>

A reward model is a classifier derived from an instruction-tuned LLM by replacing its language-modeling head with a regression head that outputs a single “quality” score for a given prompt–completion pair. You train it on human preference data pairs of completions ranked by annotators so that, at inference, it predicts how well a new completion matches human judgments. This automated scoring then guides the subsequent preference-tuning (e.g., via reinforcement learning) to align the LLM’s outputs with desired human preferences.

</details>

---

### Q: What is Proximal Policy Optimization (PPO) in preference tuning (RLHF), and how does it work?

<details>
<summary><b>💡 Show Answer</b></summary>

PPO is the RL algorithm commonly used in RLHF. The LLM is a policy that generates responses; a reward model scores them.

Steps:
1. **Collect trajectories**: Current policy generates responses to prompts.
2. **Compute rewards**: Reward model scores each response (often with a KL penalty vs a reference model).
3. **Clipped policy update**: Maximize expected advantage while limiting how much the policy can change:


$$L(\theta) = \mathbb{E}\big[\min\big(r(\theta)A,\; \mathrm{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A\big)\big]$$


where $r(\theta)$ is the probability ratio of new vs old policy and $A$ is the advantage.

Clipping keeps updates stable so alignment improves without collapsing the policy.

</details>

---

### Q: What is Direct Preference Optimization (DPO), and how does it function?

<details>
<summary><b>💡 Show Answer</b></summary>

DPO aligns an LLM on preference pairs **without** a separate reward model or PPO sampling loop.

1. Collect pairs $(y^+, y^-)$ — preferred vs dispreferred responses for each prompt.
2. Optimize a classification-style loss that increases the likelihood of $y^+$ over $y^-$ relative to a frozen reference model $\pi_{\mathrm{ref}}$:


$$\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\log \sigma\Big(\beta \log \frac{\pi_\theta(y^+\mid x)}{\pi_{\mathrm{ref}}(y^+\mid x)} - \beta \log \frac{\pi_\theta(y^-\mid x)}{\pi_{\mathrm{ref}}(y^-\mid x)}\Big)$$


3. Train with standard supervised gradient descent.

**Trade-off**: Simpler and often more stable than RLHF; may be less flexible when you need explicit reward shaping or online exploration.

</details>

---

### Q: How would you fix an LLM generating biased or incorrect outputs?

<details>
<summary><b>💡 Show Answer</b></summary>

To address biased or incorrect outputs:

- Analyze Patterns: Identify bias sources in data or prompts.
- Enhance Data: Use balanced datasets and debiasing techniques.
- Fine-Tune: Retrain with curated data or adversarial methods.

These steps improve fairness and accuracy.

</details>

---

## Agents, Memory & Production

### Q: What do you mean by LLM memory, and why is it important?

<details>
<summary><b>💡 Show Answer</b></summary>

By default, Large Language Models (LLMs) are stateless, meaning they do not remember previous interactions. To enable memory and create more coherent interactions, we can extend LLMs with different memory mechanisms. These allow the model to recall user context, personalize responses, and maintain state across multiple turns.

Types of LLM Memory:

- Conversation Buffer Memory: Appends the full chat history to each prompt.
- Windowed Buffer Memory: Retains only the last *k* exchanges.
- Conversation Summary Memory: Summarizes past conversations using another LLM and stores the distilled version.

</details>

---

### Q: What are LLM agents, and how do they extend capabilities beyond static chains?

<details>
<summary><b>💡 Show Answer</b></summary>

LLM agents are systems that empower language models to decide and plan actions dynamically rather than following a fixed, predefined flow. Unlike standard chains, which follow a static sequence of steps, agents can reason about a task, select appropriate tools, and determine the best order of actions to achieve a goal.

Key Components of LLM Agents:

- Tools: External functionalities that agents can call (e.g., calculator, web search, APIs). These tools allow LLMs to overcome their inherent limitations (e.g., poor math skills or outdated knowledge).
- Agent Type (Planner): The mechanism that enables the agent to reason about the problem and decide which tools to use and when.

</details>

---

### Q: What is the ReAct framework, and how does it enable step-by-step reasoning in LLM agents?

<details>
<summary><b>💡 Show Answer</b></summary>

The ReAct (Reasoning and Acting) framework empowers language model agents to think, act, and learn in a closed loop, combining their strong reasoning abilities with external tool use. It operates through an iterative Thought → Action → Observation cycle:

- Thought: The agent generates an internal rationale: “What should I do next and why?”
- Action: Based on its thought, the agent invokes an external tool (calculator, web search, API).
- Observation: The agent receives and processes the tool’s output, integrating it into its next reasoning step.

</details>

---

### Q: How does knowledge graph integration improve LLMs?

<details>
<summary><b>💡 Show Answer</b></summary>

Knowledge graphs provide structured, factual data, enhancing LLMs by:

- Reducing Hallucinations: Verifying facts against the graph.
- Improving Reasoning: Leveraging entity relationships.
- Enhancing Context: Offering structured context for better responses.

This is valuable for question answering and entity recognition.

</details>

---

### Q: What is a rate limit issue in closed LLM APIs, and how can it be handled?

<details>
<summary><b>💡 Show Answer</b></summary>

Rate limiting occurs when an API restricts the number of requests you can make within a given time (e.g., per minute or hour). If you exceed this limit, the API returns a **rate limit error** (often HTTP 429).

To handle it:

- Exponential Backoff: Retry the request after increasing delays (e.g., 1s, 2s, 4s...) until it succeeds or hits a retry cap.
- Jitter: Add randomness to retry delays to avoid simultaneous retries.
- Use Rate Headers: Respect API-provided headers like Retry-After to determine wait times.
- Queue Requests: Manage high-volume traffic with a request queue to avoid bursts.

</details>

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)
