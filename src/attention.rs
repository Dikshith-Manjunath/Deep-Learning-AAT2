use crate::tensor::Matrix;

#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub d_k: usize,
    pub w_q: Matrix,
    pub b_q: Vec<f32>,
    pub w_k: Matrix,
    pub b_k: Vec<f32>,
    pub w_v: Matrix,
    pub b_v: Vec<f32>,
    pub w_o: Matrix,
    pub b_o: Vec<f32>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );
        let d_k = d_model / num_heads;
        let scale = (2.0 / (d_model + d_model) as f32).sqrt(); // Xavier initialization

        Self {
            d_model,
            num_heads,
            d_k,
            w_q: Matrix::random(d_model, d_model, scale),
            b_q: vec![0.0; d_model],
            w_k: Matrix::random(d_model, d_model, scale),
            b_k: vec![0.0; d_model],
            w_v: Matrix::random(d_model, d_model, scale),
            b_v: vec![0.0; d_model],
            w_o: Matrix::random(d_model, d_model, scale),
            b_o: vec![0.0; d_model],
        }
    }

    /// Forward pass through multi-head self-attention
    /// x: (seq_len, d_model)
    /// attention_mask: optional slice of length seq_len (1.0 for valid tokens, 0.0 for padding)
    /// returns: (output: (seq_len, d_model), attention_maps: Vec<Matrix>)
    pub fn forward(
        &self,
        x: &Matrix,
        attention_mask: Option<&[f32]>,
    ) -> (Matrix, Vec<Matrix>) {
        let seq_len = x.rows;
        assert_eq!(x.cols, self.d_model);

        // 1. Linear projections for Q, K, V
        let q_proj = x.matmul(&self.w_q).add_bias(&self.b_q); // (seq_len, d_model)
        let k_proj = x.matmul(&self.w_k).add_bias(&self.b_k); // (seq_len, d_model)
        let v_proj = x.matmul(&self.w_v).add_bias(&self.b_v); // (seq_len, d_model)

        let mut head_outputs: Vec<Matrix> = Vec::with_capacity(self.num_heads);
        let mut attention_maps: Vec<Matrix> = Vec::with_capacity(self.num_heads);
        let scale = 1.0 / (self.d_k as f32).sqrt();

        // 2. Scaled Dot-Product Attention per head
        for h in 0..self.num_heads {
            let offset = h * self.d_k;

            // Extract head slice: shape (seq_len, d_k)
            let mut q_h = Matrix::zeros(seq_len, self.d_k);
            let mut k_h = Matrix::zeros(seq_len, self.d_k);
            let mut v_h = Matrix::zeros(seq_len, self.d_k);

            for r in 0..seq_len {
                for c in 0..self.d_k {
                    q_h.set(r, c, q_proj.get(r, offset + c));
                    k_h.set(r, c, k_proj.get(r, offset + c));
                    v_h.set(r, c, v_proj.get(r, offset + c));
                }
            }

            // Raw attention scores: S_h = (Q_h * K_h^T) / sqrt(d_k) -> (seq_len, seq_len)
            let k_h_t = k_h.transpose();
            let mut scores = q_h.matmul(&k_h_t);
            for i in 0..scores.data.len() {
                scores.data[i] *= scale;
            }

            // Softmax with masking
            let attn_weights = scores.softmax_masked(attention_mask);
            attention_maps.push(attn_weights.clone());

            // Context output for head: O_h = Attn_h * V_h -> (seq_len, d_k)
            let out_h = attn_weights.matmul(&v_h);
            head_outputs.push(out_h);
        }

        // 3. Concatenate all heads -> (seq_len, d_model)
        let mut concat = Matrix::zeros(seq_len, self.d_model);
        for r in 0..seq_len {
            for h in 0..self.num_heads {
                let offset = h * self.d_k;
                let head_out = &head_outputs[h];
                for c in 0..self.d_k {
                    concat.set(r, offset + c, head_out.get(r, c));
                }
            }
        }

        // 4. Final output projection: Concat * W_o + b_o
        let output = concat.matmul(&self.w_o).add_bias(&self.b_o);

        (output, attention_maps)
    }
}
