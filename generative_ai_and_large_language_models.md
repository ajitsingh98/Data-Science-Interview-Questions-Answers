# Generative AI & Large Language Models (LLMs) Interview Questions

## Table of Contents
- [LLM Architectures & Foundations](#llm-architectures--foundations)
- [Fine-Tuning Techniques (LoRA, QLoRA, PEFT)](#fine-tuning-techniques-lora-qlora-peft)
- [Retrieval-Augmented Generation (RAG) & Vector Databases](#retrieval-augmented-generation-rag--vector-databases)
- [Alignment & Preference Optimization (RLHF, DPO)](#alignment--preference-optimization-rlhf-dpo)
- [Quantization & Efficient Inference](#quantization--efficient-inference)

---

## LLM Architectures & Foundations

### Q: What is the core difference between Encoder-only (BERT), Decoder-only (GPT/Llama), and Encoder-Decoder (T5) architectures?
<details>
<summary><b>💡 Show Answer</b></summary>

- **Encoder-Only (BERT)**: Uses bidirectional self-attention. Ideal for classification, NER, and sentence embeddings.
- **Decoder-Only (GPT/Llama/Mistral)**: Uses causal masked self-attention to predict the next token. Ideal for auto-regressive text generation.
- **Encoder-Decoder (T5/BART)**: Uses bidirectional encoder + cross-attention decoder. Ideal for sequence-to-sequence tasks like translation and summarization.
</details>

---

## Fine-Tuning Techniques (LoRA, QLoRA, PEFT)

### Q: Explain Low-Rank Adaptation (LoRA) and how it achieves parameter-efficient fine-tuning.
<details>
<summary><b>💡 Show Answer</b></summary>

LoRA freezes pre-trained weight matrices $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank decomposition matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll \min(d, k)$:
$$ W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A) $$

This reduces trainable parameters by $>99\%$ while maintaining model accuracy.
</details>

---

[⬆️ Back to Top](#table-of-contents) | [🏠 Back to Main Index](./README.md)
