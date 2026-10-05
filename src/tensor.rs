use std::f32::consts::PI;

/// A simple 2D dense float matrix for neural network operations.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            rows * cols,
            data.len(),
            "Data length {} does not match dimensions {}x{}",
            data.len(),
            rows,
            cols
        );
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize, std_dev: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1: f32 = rng.gen::<f32>().max(1e-7);
                let u2: f32 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                z * std_dev
            })
            .collect();
        Self { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f32) {
        self.data[r * self.cols + c] = val;
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[allow(dead_code)]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Matrix multiplication: C = A (M x K) * B (K x N) -> (M x N)
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Cannot multiply matrix {}x{} with {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for p in 0..k {
                let a_ip = self.data[i * k + p];
                for j in 0..n {
                    result[i * n + j] += a_ip * other.data[p * n + j];
                }
            }
        }
        Matrix::new(m, n, result)
    }

    /// Transpose matrix: (M x N) -> (N x M)
    pub fn transpose(&self) -> Matrix {
        let mut result = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[j * self.rows + i] = self.get(i, j);
            }
        }
        Matrix::new(self.cols, self.rows, result)
    }

    /// Elementwise addition of two identically shaped matrices
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Matrix::new(self.rows, self.cols, data)
    }

    /// Broadcast add a bias vector of shape (1 x cols) across all rows
    pub fn add_bias(&self, bias: &[f32]) -> Matrix {
        assert_eq!(self.cols, bias.len());
        let mut out = self.clone();
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(r, c, out.get(r, c) + bias[c]);
            }
        }
        out
    }

    /// Layer Normalization across the last dimension (cols)
    /// LN(x) = ((x - mean) / sqrt(var + eps)) * gamma + beta
    pub fn layer_norm(&self, gamma: &[f32], beta: &[f32], eps: f32) -> Matrix {
        let mut out = Matrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            let row = self.row(r);
            let mean: f32 = row.iter().sum::<f32>() / self.cols as f32;
            let var: f32 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / self.cols as f32;
            let std_inv = 1.0 / (var + eps).sqrt();

            for c in 0..self.cols {
                let normalized = (row[c] - mean) * std_inv;
                let g = if c < gamma.len() { gamma[c] } else { 1.0 };
                let b = if c < beta.len() { beta[c] } else { 0.0 };
                out.set(r, c, normalized * g + b);
            }
        }
        out
    }

    /// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Matrix {
        let sqrt_2_over_pi = (2.0 / PI).sqrt();
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        Matrix::new(self.rows, self.cols, data)
    }

    /// Softmax along each row with optional attention mask (mask=0 means masked out / -inf)
    pub fn softmax_masked(&self, mask: Option<&[f32]>) -> Matrix {
        let mut out = Matrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            let mut max_val = f32::NEG_INFINITY;
            for c in 0..self.cols {
                let is_masked = if let Some(m) = mask {
                    m[c] == 0.0
                } else {
                    false
                };
                let val = if is_masked { -1e9 } else { self.get(r, c) };
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum_exp = 0.0;
            let mut exp_vals = vec![0.0; self.cols];
            for c in 0..self.cols {
                let is_masked = if let Some(m) = mask {
                    m[c] == 0.0
                } else {
                    false
                };
                if is_masked {
                    exp_vals[c] = 0.0;
                } else {
                    let e = (self.get(r, c) - max_val).exp();
                    exp_vals[c] = e;
                    sum_exp += e;
                }
            }

            let safe_sum = if sum_exp > 0.0 { sum_exp } else { 1.0 };
            for c in 0..self.cols {
                out.set(r, c, exp_vals[c] / safe_sum);
            }
        }
        out
    }
}

/// Mean pooling over sequence tokens taking attention mask into account
pub fn mean_pooling(token_embeddings: &Matrix, attention_mask: &[f32]) -> Vec<f32> {
    let d_model = token_embeddings.cols;
    let seq_len = token_embeddings.rows;
    let mut sum_vec = vec![0.0; d_model];
    let mut mask_sum = 0.0;

    for i in 0..seq_len {
        let mask_val = if i < attention_mask.len() {
            attention_mask[i]
        } else {
            1.0
        };
        if mask_val > 0.0 {
            mask_sum += mask_val;
            for j in 0..d_model {
                sum_vec[j] += token_embeddings.get(i, j) * mask_val;
            }
        }
    }

    let divisor = if mask_sum > 0.0 { mask_sum } else { 1.0 };
    sum_vec.iter().map(|&x| x / divisor).collect()
}

/// Normalize vector to unit length (L2 norm)
pub fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let safe_norm = if norm > 1e-12 { norm } else { 1.0 };
    vec.iter().map(|&x| x / safe_norm).collect()
}

/// Compute cosine similarity between two unit vectors (or arbitrary vectors)
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have equal length");
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot / (norm_a * norm_b)
}
