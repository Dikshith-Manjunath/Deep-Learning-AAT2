# Architecture & Mathematical Foundations: Sentence Transformer from Scratch

> **Presentation & Defense Reference for AIML Panel**  
> **Topic**: Semantic Search & Document Deduplication via Dense Sentence Embeddings  
> **Implementation**: Pure Rust Bare-Metal Transformer Engine (No PyTorch/HuggingFace abstraction)

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Challenge of Lexical Search & Deduplication
Traditional text retrieval and deduplication systems rely on lexical matching (e.g., TF-IDF, BM25, Jaccard similarity of n-grams). These methods suffer from two critical fundamental flaws:
1. **Vocabulary Mismatch (Synonymy)**: *"deep learning architectures"* and *"neural network models"* share zero lexical tokens despite identical semantic meaning.
2. **Polysemy & Context Blindness**: The word *"bank"* in *"river bank"* vs. *"investment bank"* has identical lexical representation.

### 1.2 The Sentence Transformer Solution
A **Sentence Transformer** projects variable-length natural language sequences $\mathcal{S}$ into a fixed-dimensional dense vector space:
$$f_\theta: \mathcal{S} \to \mathbb{R}^{d_{\text{model}}} \quad \text{such that } \quad \cos(f_\theta(s_1), f_\theta(s_2)) \propto \text{SemanticSimilarity}(s_1, s_2)$$

This dense vector space powers:
- **Vector Databases & RAG**: Real-time approximate nearest neighbor (ANN / HNSW) semantic search.
- **Document Deduplication**: Near-duplicate detection by thresholding cosine similarity ($\ge \tau$) without pairwise lexical string comparisons.

```
+-----------------------------------------------------------------------------------+
|                            END-TO-END SYSTEM PIPELINE                             |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Input Text: "Deep learning models use neural networks"                           |
|       │                                                                           |
|       ▼                                                                           |
|  [Tokenizer] ──► [CLS], deep, learning, models, use, neural, networks, [SEP]      |
|       │                                                                           |
|       ▼                                                                           |
|  [Embedding Lookup] Token Embeddings + Sinusoidal Positional Encoding             |
|       │                                                                           |
|       ▼                                                                           |
|  +-----------------------------------------------------------------------------+  |
|  | Transformer Encoder Layer 1                                                 |  |
|  |   ├── Multi-Head Self-Attention (Q, K, V projections + Scaled Dot-Product)  |  |
|  |   ├── Add & Layer Normalization                                             |  |
|  |   ├── Position-Wise Feed-Forward Network (Linear ──► GELU ──► Linear)       |  |
|  |   └── Add & Layer Normalization                                             |  |
|  +-----------------------------------------------------------------------------+  |
|       │                                                                           |
|       ▼                                                                           |
|  +-----------------------------------------------------------------------------+  |
|  | Transformer Encoder Layer 2 (Repeated contextual refinement)               |  |
|  +-----------------------------------------------------------------------------+  |
|       │                                                                           |
|       ▼ (Sequence of Contextual Token Vectors: L x d_model)                       |
|  [Mean Pooling with Attention Mask] ──► Aggregates active token states            |
|       │                                                                           |
|       ▼ (Single Sentence Vector: 1 x d_model)                                     |
|  [L2 Normalization] ──► Projects onto Unit Hypersphere (||e||_2 = 1.0)           |
|       │                                                                           |
|       ▼                                                                           |
|  Dense 64-D Embedding: [-0.0339, 0.1152, -0.0517, ..., 0.0980]                   |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Mathematical Foundations of Transformer Architecture

### 2.1 Input Representation & Positional Encoding
Since self-attention contains no inherent recurrence or convolution, the model is permutation-equivariant. Positional information must be explicitly injected.

For token index $t$ at sequence position $pos$ and hidden dimension index $i \in [0, d_{\text{model}}-1]$:
$$X_{\text{input}}[pos, :] = E_{\text{token}}[t] \cdot \sqrt{d_{\text{model}}} + PE[pos, :]$$

where sinusoidal positional encoding is computed as:
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

This allows the model to attend based on relative positions because for any fixed offset $k$, $PE(pos+k)$ can be expressed as a linear function of $PE(pos)$.

---

### 2.2 The Scaled Dot-Product Attention Mechanism

Given an input matrix $H \in \mathbb{R}^{L \times d_{\text{model}}}$ (where $L$ is sequence length):
Linear projections map $H$ into Query ($Q$), Key ($K$), and Value ($V$) spaces:
$$Q = H W_Q + b_Q, \quad K = H W_K + b_K, \quad V = H W_V + b_V$$
where $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

```
                    Query (Q)       Key (K)
                        │              │
                        └───► [MatMul] ◄── Transpose (K^T)
                                 │
                                 ▼  (Raw Dot-Product Scores)
                           [Scale by 1 / sqrt(d_k)]
                                 │
                                 ▼
                           [Mask (Padding = -inf)]
                                 │
                                 ▼
                             [Softmax] ──► (Attention Probability Matrix A)
                                 │
                                 ├───► Value (V)
                                 ▼
                              [MatMul]
                                 │
                                 ▼
                           Context Vectors (L x d_v)
```

#### Attention Formula:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

#### Mathematical Breakdown:
1. **Compatibility Scoring ($Q K^T$)**: Computes pairwise alignment between every query token $i$ and key token $j$.
2. **Scaling Factor ($\frac{1}{\sqrt{d_k}}$)**:
   - For large $d_k$, the dot product $\sum_{m=1}^{d_k} q_m k_m$ has variance $\text{Var}(q \cdot k) = d_k$ (assuming zero mean, unit variance).
   - Large magnitude logits push the $\text{Softmax}$ function into regions with near-zero gradients ($\frac{\partial \text{Softmax}}{\partial z_i} \to 0$).
   - Scaling by $\frac{1}{\sqrt{d_k}}$ preserves unit variance and prevents vanishing gradients.
3. **Padding Mask ($M$)**:
   $$M_{i,j} = \begin{cases} 0 & \text{if token } j \text{ is valid} \\ -\infty & \text{if token } j \text{ is [PAD]} \end{cases}$$
   Since $e^{-\infty} = 0$, padded positions receive zero attention weight.
4. **Context Aggregation**: Multiplies the attention distribution matrix by $V$ to compute the contextualized representation.

---

### 2.3 Multi-Head Attention (MHA)

Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, Multi-Head Attention projects queries, keys, and values $h$ times with distinct learnable parameter matrices:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W_O + b_O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

```
                        Input Hidden States (L x d_model)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
       [Head 1]                     [Head 2]                   [Head 4]
   (Q1, K1, V1 in R^16)        (Q2, K2, V2 in R^16)       (Q4, K4, V4 in R^16)
            │                          │                          │
   [Scaled Dot-Product]        [Scaled Dot-Product]       [Scaled Dot-Product]
            │                          │                          │
      Context Out 1              Context Out 2              Context Out 4
            └──────────────────────────┬──────────────────────────┘
                                       ▼
                       [Concatenate Heads: L x 64]
                                       │
                                       ▼
                         [Linear Projection: W_O]
                                       │
                                       ▼
                         Output States: (L x d_model)
```

#### Why Multi-Head Attention is Superior:
- **Subspace Specialization**: Head 1 can attend to immediate syntactic neighbors (subject-verb agreement), Head 2 to long-range dependencies, Head 3 to semantic synonyms, and Head 4 to punctuation/delimiters.
- **Ensemble Representation**: Avoids averaging out diverse relationships into a single attention map.

---

### 2.4 Feed-Forward Network (FFN) & Layer Normalization

Each encoder layer contains a fully connected Feed-Forward Network applied independently to each position:
$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$
where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ and $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.

#### GELU (Gaussian Error Linear Unit):
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
Approximated in our Rust engine as:
$$\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3\right)\right)\right)$$

#### Residual Connection & Layer Normalization:
$$\text{Output}_1 = \text{LayerNorm}(x + \text{MultiHead}(x))$$
$$\text{Output}_2 = \text{LayerNorm}(\text{Output}_1 + \text{FFN}(\text{Output}_1))$$

Layer Normalization computes statistics across the feature dimension $d_{\text{model}}$ for each token independently:
$$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

---

## 3. Sentence Pooling & Vector Normalization

Standard BERT-style models produce sequence outputs $H \in \mathbb{R}^{L \times d_{\text{model}}}$. For sentence embeddings, this sequence must be collapsed into a single vector $\mathbf{u} \in \mathbb{R}^{d_{\text{model}}}$.

### 3.1 Mean Pooling vs. [CLS] Token Pooling

| Pooling Strategy | Formulation | Why Mean Pooling is Preferred |
|---|---|---|
| **[CLS] Pooling** | $\mathbf{u} = H[0, :]$ | Captures classification bias; often requires massive pre-training to generalize across unconstrained sentence similarity. |
| **Mean Pooling** | $\mathbf{u} = \frac{\sum_{i=1}^L m_i H[i, :]}{\sum_{i=1}^L m_i}$ | Uniformly weights contextual representations of all active informative tokens, significantly outperforming `[CLS]` on semantic benchmarks. |

### 3.2 L2 Normalization onto the Unit Hypersphere
$$\mathbf{e} = \frac{\mathbf{u}}{\|\mathbf{u}\|_2} = \frac{\mathbf{u}}{\sqrt{\sum_{k=1}^{d} u_k^2}}$$

#### Key Properties of Unit Embeddings:
1. All vectors lie on $\mathbb{S}^{d-1}$ (the unit sphere).
2. **Cosine Similarity equals Euclidean Dot Product**:
   $$\cos(\mathbf{e}_A, \mathbf{e}_B) = \frac{\mathbf{e}_A \cdot \mathbf{e}_B}{\|\mathbf{e}_A\|_2 \|\mathbf{e}_B\|_2} = \mathbf{e}_A^\top \mathbf{e}_B$$
3. Fast vectorized search: Similarity reduces to a single matrix multiplication without division.

---

## 4. Rust POC Architecture & Code Organization

The proof-of-concept is implemented in pure, safe Rust without heavy framework overhead:

```
src/
├── tensor.rs       # Matrix math, MatMul, Transpose, GELU, Masked Softmax, LayerNorm, L2Norm
├── tokenizer.rs    # Whitespace/punctuation tokenization, special tokens ([CLS], [SEP], [PAD])
├── attention.rs    # MultiHeadAttention struct with Scaled Dot-Product & Q, K, V projections
├── encoder.rs      # TransformerEncoderLayer (MHA + FFN + Residuals + LayerNorm) & Sinusoidal PE
├── model.rs        # SentenceTransformer combining Tokenizer + Encoders + Mean Pooling + L2 Normalization
├── trainer.rs      # Rapid contrastive training routine (MSE on Cosine Similarity)
└── main.rs         # CLI prompt, demo benchmark evaluation, and formatted file exporter
```

### Hyperparameters Configured in POC:
- **Embedding Dimension ($d_{\text{model}}$)**: 64
- **Attention Heads ($h$)**: 4 ($d_k = d_v = 16$)
- **Feed-Forward Dimension ($d_{\text{ff}}$)**: 128
- **Transformer Encoder Layers ($N$)**: 2
- **Max Sequence Length ($L_{\text{max}}$)**: 32 tokens
- **Activation Function**: GELU

---

## 5. Defense Guide: Q&A for AIML Professors

### Q1: Why not just average Word2Vec or GloVe embeddings for a sentence?
> **Answer**: Static embeddings (Word2Vec, GloVe) assign a single static vector to each word regardless of context. They cannot resolve polysemy (*"apple stock"* vs *"green apple"*), and they completely ignore syntax and word order. Sentence Transformers use self-attention to generate **context-dependent** token representations where every word's vector is dynamically modulated by every other word in the sequence.

### Q2: Why is the scaling factor $\frac{1}{\sqrt{d_k}}$ used in dot-product attention instead of $\frac{1}{d_k}$?
> **Answer**: If components of $q$ and $k$ are independent random variables with mean 0 and variance 1, their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $d_k$. To scale the variance back to 1 (standard normal distribution), we divide by the standard deviation, which is $\sqrt{d_k}$. Dividing by $d_k$ would over-dampen the logits, causing softmax to approach a uniform distribution.

### Q3: Why is Layer Normalization used instead of Batch Normalization in Transformers?
> **Answer**: Batch Normalization computes mean and variance across the batch dimension, which suffers with variable sequence lengths and small batch sizes during inference. Layer Normalization computes statistics across the feature dimension independently for each sample/token, making it completely independent of batch size and padding lengths.

### Q4: How does this model scale to real-world Retrieval-Augmented Generation (RAG)?
> **Answer**: In production RAG systems (using models like `all-MiniLM-L6-v2` or `text-embedding-3-small`), documents are chunked, passed through this encoder once at indexing time, and stored in a vector index (e.g., Faiss, Qdrant, Milvus). At query time, the user prompt is encoded via the same Transformer, and approximate nearest neighbor search finds the top-$k$ documents in sub-millisecond time.
