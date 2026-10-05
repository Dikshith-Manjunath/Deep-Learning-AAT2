<div align="center">

# 🦀 Rust Sentence Transformer (from Scratch)

**A bare-metal, pure-Rust implementation of an Encoder-only Transformer for Semantic Search & Document Deduplication.**

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg?style=for-the-badge&logo=rust)](https://www.rust-lang.org)
[![Architecture](https://img.shields.io/badge/Architecture-MiniLM--Style%20Encoder-blue.svg?style=for-the-badge)](TRANSFORMER_ARCHITECTURE.md)
[![Zero Dependency](https://img.shields.io/badge/Frameworks-Zero%20(Pure%20Rust)-green.svg?style=for-the-badge)](#zero-framework-philosophy)
[![License](https://img.shields.io/badge/License-MIT-purple.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <a href="#-key-features">Key Features</a> •
  <a href="#-architecture--mathematics">Architecture & Math</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-benchmarks--similarity-evaluation">Benchmarks</a> •
  <a href="#-file-structure">File Structure</a> •
  <a href="TRANSFORMER_ARCHITECTURE.md">Full Theory Docs</a>
</p>

---

</div>

## 📌 Overview

This repository contains a lightweight, educational **Sentence Transformer (MiniLM-style)** built **entirely from scratch in safe Rust** without relying on PyTorch, LibTorch, ONNX, or Hugging Face.

It transforms variable-length sentences into fixed $d$-dimensional dense numerical vectors (embeddings) positioned on a **Unit Hypersphere** ($\mathbb{S}^{d-1}$). Sentences with similar semantic meanings are placed closely together in vector space, enabling **sub-millisecond Semantic Search, Document Deduplication, and Retrieval-Augmented Generation (RAG)**.

```
Input: "Deep learning models use neural networks"
          │
          ▼ [Tokenizer + Sinusoidal Positional Encoding]
  Sequence of Token Embeddings (L x d_model)
          │
          ▼ [Stacked Transformer Encoder Layers]
  Multi-Head Self-Attention + Add & LayerNorm + GELU FFN
          │
          ▼ [Mean Pooling (masked over active tokens)]
  Dense Sentence Vector (1 x d_model)
          │
          ▼ [L2 Normalization]
  Unit Hypersphere Embedding: [-0.1180, 0.0488, 0.2052, ..., 0.0173]  (||e||_2 = 1.0)
```

---

## ✨ Key Features

- **🛡️ Zero Heavyweight ML Frameworks**: Bare-metal linear algebra and matrix operations written in pure, idiomatic Rust.
- **⚡ High Performance**: 200 epochs of contrastive training execute in **~3.0 seconds** in release mode.
- **🧠 Multi-Head Self-Attention (MHA)**: Implements Scaled Dot-Product Attention across multiple orthogonal representation heads ($Q, K, V$).
- **📐 Mathematical Precision**:
  - Sinusoidal Positional Encodings (Vaswani et al.)
  - Gaussian Error Linear Unit (**GELU**) activation
  - Feature-wise **Layer Normalization** (pre/post-residual)
  - Attention-masked **Mean Pooling** + **$L_2$ Unit Sphere Normalization**
- **📊 Presentation & Demo Ready**:
  - Interactive terminal mode + CLI argument parser
  - Automated similarity benchmark across paraphrase & non-paraphrase pairs
  - Dual export: Formatted Markdown report ([`embeddings_output.md`](embeddings_output.md)) and structured JSON ([`embeddings_output.json`](embeddings_output.json))

---

## 🧮 Architecture & Mathematics

For a comprehensive theoretical derivation and defense reference, see [**`TRANSFORMER_ARCHITECTURE.md`**](TRANSFORMER_ARCHITECTURE.md).

### 1. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
* Scaling by $\frac{1}{\sqrt{d_k}}$ preserves unit variance of logits, preventing vanishing gradients in the softmax tails.
* Padding mask $M_{i,j} = -\infty$ for `[PAD]` positions eliminates influence from dummy tokens.

### 2. Multi-Head Attention (MHA)
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O + b_O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 3. Layer Normalization & GELU FFN
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

### 4. Mean Pooling & Unit Sphere Normalization
$$\mathbf{u} = \frac{\sum_{i=1}^L m_i \mathbf{h}_i}{\sum_{i=1}^L m_i}, \qquad \mathbf{e} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2}$$
Because $\|\mathbf{e}\|_2 = 1$, cosine similarity simplifies to the Euclidean inner product:
$$\cos(\mathbf{e}_A, \mathbf{e}_B) = \mathbf{e}_A^\top \mathbf{e}_B$$

---

## 🚀 Quickstart

### Prerequisites
- [Rust & Cargo](https://rustup.rs/) (version 1.70 or newer)

### 1. Clone & Build
```bash
git clone https://github.com/your-username/sentence-transformer-rust.git
cd sentence-transformer-rust
cargo build --release
```

### 2. Encode a Single Sentence via CLI
```bash
cargo run --release -- "Semantic search retrieves documents by meaning rather than keywords"
```

### 3. Run the Interactive Mode & Benchmark
```bash
cargo run --release
```

---

## 📈 Benchmarks & Similarity Evaluation

During the automated benchmark on representative sentence pairs, the model produces clear, separated cosine similarities:

| Sentence A | Sentence B | Cosine Similarity | Semantic Interpretation |
|---|---|:---:|---|
| *"Deep learning models use neural networks to learn representations"* | *"Artificial intelligence leverages neural architectures for representation learning"* | **`0.8709`** | 🟢 **High Match (Paraphrase)** |
| *"Deep learning models use neural networks to learn representations"* | *"Vector databases store embeddings for high dimensional nearest neighbor search"* | **`0.5497`** | 🟡 **Moderate Match (Related Domain)** |
| *"Deep learning models use neural networks to learn representations"* | *"The cat is resting on the living room rug"* | **`0.2965`** | 🔴 **Low Match (Unrelated Domain)** |

### 200 Epoch Training Convergence Log:
```text
=======================================================
 [TRAINING] Starting POC Contrastive Training Loop
=======================================================
 • Architecture: 2 Transformer Layer(s), 4 Attention Heads, d_model = 64
 • Training objective: Cosine Similarity MSE Loss
 • Number of Epochs: 200 | Learning Rate: 0.05
-------------------------------------------------------
 Epoch   1/200 | Avg MSE Loss: 0.195324 | Elapsed: 79.06ms
 Epoch  40/200 | Avg MSE Loss: 0.095627 | Elapsed: 703.81ms
 Epoch  80/200 | Avg MSE Loss: 0.052797 | Elapsed: 1.30s
 Epoch 120/200 | Avg MSE Loss: 0.032572 | Elapsed: 1.89s
 Epoch 160/200 | Avg MSE Loss: 0.021820 | Elapsed: 2.47s
 Epoch 200/200 | Avg MSE Loss: 0.015676 | Elapsed: 3.05s
-------------------------------------------------------
 Training completed in 3.05s! Ready for inference.
=======================================================
```

---

## 📂 File Structure

```text
sentence_transformer_poc/
├── Cargo.toml                     # Cargo configuration & dependencies
├── README.md                      # Project overview and GitHub documentation
├── TRANSFORMER_ARCHITECTURE.md    # Detailed mathematical breakdown for panel defense
├── embeddings_output.md           # Formatted Markdown report of generated embeddings
├── embeddings_output.json         # Raw JSON export of token IDs and vector arrays
└── src/
    ├── main.rs                    # Entry point, CLI dispatcher, interactive loop
    ├── model.rs                   # SentenceTransformer struct & pipeline orchestration
    ├── attention.rs               # MultiHeadAttention & Scaled Dot-Product Attention
    ├── encoder.rs                 # TransformerEncoderLayer, FFN, Sinusoidal Positional Encoding
    ├── tensor.rs                  # Matrix math, MatMul, LayerNorm, GELU, Cosine Similarity
    ├── tokenizer.rs               # Word/Subword tokenizer with special tokens ([CLS], [SEP], [PAD])
    └── trainer.rs                 # 200-epoch contrastive loss training routine
```

---

## ⚙️ Model Hyperparameters

| Hyperparameter | Value | Description |
|---|:---:|---|
| **Embedding Dimension ($d_{\text{model}}$)** | `64` | Dimensionality of token and sentence representations |
| **Attention Heads ($h$)** | `4` | Number of parallel attention heads |
| **Head Dimension ($d_k = d_v$)** | `16` | Subspace projection dimension ($64 / 4 = 16$) |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | `128` | Intermediate expansion in pointwise FFN |
| **Encoder Layers ($N$)** | `2` | Number of stacked Transformer encoder layers |
| **Max Sequence Length ($L_{\text{max}}$)** | `32` | Max sequence length (padded with `[PAD]`) |
| **Activation Function** | `GELU` | Gaussian Error Linear Unit |
| **Pooling Strategy** | `Mean Pooling` | Attention-weighted average over active tokens |
| **Normalization** | `$L_2$` | Unit Hypersphere projection ($\|\mathbf{e}\|_2 = 1.0$) |

---

## 📚 References

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
2. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
3. Wang, W., et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. NeurIPS.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
