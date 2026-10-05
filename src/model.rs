use serde::Serialize;
use crate::encoder::{get_sinusoidal_positional_encoding, TransformerEncoderLayer};
use crate::tensor::{cosine_similarity, l2_normalize, mean_pooling, Matrix};
use crate::tokenizer::SimpleTokenizer;

#[derive(Debug, Clone, Serialize)]
pub struct SentenceEmbeddingResult {
    pub text: String,
    pub tokens: Vec<String>,
    pub token_ids: Vec<usize>,
    pub attention_mask: Vec<f32>,
    pub d_model: usize,
    pub embedding_sample: Vec<f32>, // First 8 dimensions for quick preview
    pub embedding_full: Vec<f32>,   // Complete dense vector (e.g. 64-dim)
    pub l2_norm: f32,
}

#[derive(Debug, Clone)]
pub struct SentenceTransformer {
    pub tokenizer: SimpleTokenizer,
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub token_embedding: Matrix, // (vocab_size, d_model)
    pub pos_embedding: Matrix,   // (max_seq_len, d_model)
    pub layers: Vec<TransformerEncoderLayer>,
}

impl SentenceTransformer {
    pub fn new(
        tokenizer: SimpleTokenizer,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        num_layers: usize,
        max_seq_len: usize,
    ) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let scale = (1.0 / d_model as f32).sqrt();

        let token_embedding = Matrix::random(vocab_size, d_model, scale);
        let pos_embedding = get_sinusoidal_positional_encoding(max_seq_len, d_model);

        let layers = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, num_heads, d_ff))
            .collect();

        Self {
            tokenizer,
            vocab_size,
            d_model,
            num_heads,
            num_layers,
            max_seq_len,
            token_embedding,
            pos_embedding,
            layers,
        }
    }

    /// Resize/reallocate token embeddings when new tokens are added to vocabulary
    pub fn update_vocab_size(&mut self) {
        let new_vocab_size = self.tokenizer.vocab_size();
        if new_vocab_size > self.token_embedding.rows {
            let scale = (1.0 / self.d_model as f32).sqrt();
            let mut new_emb = Matrix::random(new_vocab_size, self.d_model, scale);
            // Copy existing embeddings
            for r in 0..self.token_embedding.rows {
                for c in 0..self.d_model {
                    new_emb.set(r, c, self.token_embedding.get(r, c));
                }
            }
            self.token_embedding = new_emb;
            self.vocab_size = new_vocab_size;
        }
    }

    /// Forward pass through token embedding + positional encoding + N Transformer layers
    pub fn forward(
        &self,
        token_ids: &[usize],
        attention_mask: &[f32],
    ) -> (Matrix, Vec<Vec<Matrix>>) {
        let seq_len = token_ids.len();
        let mut x = Matrix::zeros(seq_len, self.d_model);

        let emb_scale = (self.d_model as f32).sqrt();

        // 1. Look up token embeddings (scaled) + sinusoidal positional encoding
        for i in 0..seq_len {
            let id = token_ids[i].min(self.token_embedding.rows - 1);
            for j in 0..self.d_model {
                let tok_val = self.token_embedding.get(id, j) * emb_scale;
                let pos_val = if i < self.pos_embedding.rows {
                    self.pos_embedding.get(i, j) * 0.5
                } else {
                    0.0
                };
                x.set(i, j, tok_val + pos_val);
            }
        }

        // 2. Pass through stacked Transformer Encoder Layers
        let mut all_attn_maps = Vec::new();
        let mut current_h = x;

        for layer in &self.layers {
            let (next_h, attn_maps) = layer.forward(&current_h, Some(attention_mask));
            current_h = next_h;
            all_attn_maps.push(attn_maps);
        }

        (current_h, all_attn_maps)
    }

    /// Encode a single raw sentence text into a dense L2-normalized sentence embedding
    pub fn encode(&self, text: &str) -> SentenceEmbeddingResult {
        let (token_ids, attention_mask, tokens) = self.tokenizer.encode(text, self.max_seq_len);
        let (hidden_states, _attn_maps) = self.forward(&token_ids, &attention_mask);

        // Mean Pooling across non-padded tokens
        let raw_pooled = mean_pooling(&hidden_states, &attention_mask);

        // L2 normalize vector onto unit hypersphere
        let normalized = l2_normalize(&raw_pooled);

        let norm_val = normalized.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let preview_len = 8.min(normalized.len());
        let sample = normalized[0..preview_len].to_vec();

        SentenceEmbeddingResult {
            text: text.to_string(),
            tokens,
            token_ids,
            attention_mask,
            d_model: self.d_model,
            embedding_sample: sample,
            embedding_full: normalized,
            l2_norm: norm_val,
        }
    }

    /// Compute semantic cosine similarity between two sentences
    pub fn compare(&self, sent1: &str, sent2: &str) -> (f32, SentenceEmbeddingResult, SentenceEmbeddingResult) {
        let emb1 = self.encode(sent1);
        let emb2 = self.encode(sent2);
        let sim = cosine_similarity(&emb1.embedding_full, &emb2.embedding_full);
        (sim, emb1, emb2)
    }
}
