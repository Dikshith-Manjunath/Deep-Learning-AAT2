use crate::attention::MultiHeadAttention;
use crate::tensor::Matrix;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FeedForwardNetwork {
    pub d_model: usize,
    pub d_ff: usize,
    pub w1: Matrix,
    pub b1: Vec<f32>,
    pub w2: Matrix,
    pub b2: Vec<f32>,
}

impl FeedForwardNetwork {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let scale1 = (2.0 / (d_model + d_ff) as f32).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f32).sqrt();

        Self {
            d_model,
            d_ff,
            w1: Matrix::random(d_model, d_ff, scale1),
            b1: vec![0.0; d_ff],
            w2: Matrix::random(d_ff, d_model, scale2),
            b2: vec![0.0; d_model],
        }
    }

    pub fn forward(&self, x: &Matrix) -> Matrix {
        // x: (seq_len, d_model)
        // Hidden: (seq_len, d_ff) with GELU
        let hidden = x.matmul(&self.w1).add_bias(&self.b1).gelu();
        // Out: (seq_len, d_model)
        hidden.matmul(&self.w2).add_bias(&self.b2)
    }
}

#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer {
    pub attention: MultiHeadAttention,
    pub ffn: FeedForwardNetwork,
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
    pub eps: f32,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, num_heads),
            ffn: FeedForwardNetwork::new(d_model, d_ff),
            ln1_gamma: vec![1.0; d_model],
            ln1_beta: vec![0.0; d_model],
            ln2_gamma: vec![1.0; d_model],
            ln2_beta: vec![0.0; d_model],
            eps: 1e-5,
        }
    }

    /// Forward pass of one encoder block:
    /// x1 = LayerNorm(x + Attention(x))
    /// x2 = LayerNorm(x1 + FFN(x1))
    pub fn forward(
        &self,
        x: &Matrix,
        attention_mask: Option<&[f32]>,
    ) -> (Matrix, Vec<Matrix>) {
        // 1. Multi-head attention sublayer
        let (attn_out, attn_maps) = self.attention.forward(x, attention_mask);

        // 2. Residual + LayerNorm
        let x_res1 = x.add(&attn_out);
        let x1 = x_res1.layer_norm(&self.ln1_gamma, &self.ln1_beta, self.eps);

        // 3. FFN sublayer
        let ffn_out = self.ffn.forward(&x1);

        // 4. Residual + LayerNorm
        let x_res2 = x1.add(&ffn_out);
        let x2 = x_res2.layer_norm(&self.ln2_gamma, &self.ln2_beta, self.eps);

        (x2, attn_maps)
    }
}

/// Sinusoidal Positional Encoding generator (Vaswani et al.)
pub fn get_sinusoidal_positional_encoding(max_len: usize, d_model: usize) -> Matrix {
    let mut pe = Matrix::zeros(max_len, d_model);
    for pos in 0..max_len {
        for i in 0..d_model {
            if i % 2 == 0 {
                let div_term = (10000.0f32).powf((i as f32) / (d_model as f32));
                pe.set(pos, i, ((pos as f32) / div_term).sin());
            } else {
                let div_term = (10000.0f32).powf(((i - 1) as f32) / (d_model as f32));
                pe.set(pos, i, ((pos as f32) / div_term).cos());
            }
        }
    }
    pe
}
