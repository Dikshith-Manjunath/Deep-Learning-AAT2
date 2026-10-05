use crate::model::SentenceTransformer;
use crate::tensor::cosine_similarity;
use rand::seq::SliceRandom;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct TrainingPair {
    pub sent_a: &'static str,
    pub sent_b: &'static str,
    pub target_similarity: f32, // 1.0 = synonymous, 0.0 = unrelated
}

pub fn get_sample_dataset() -> Vec<TrainingPair> {
    vec![
        // High similarity pairs (Semantic duplicates / paraphrases)
        TrainingPair {
            sent_a: "deep learning models use neural networks to learn representations",
            sent_b: "artificial intelligence leverages neural architectures for representation learning",
            target_similarity: 1.0,
        },
        TrainingPair {
            sent_a: "vector databases store embeddings for high dimensional nearest neighbor search",
            sent_b: "dense vector index enables fast similarity retrieval in high dimensional space",
            target_similarity: 1.0,
        },
        TrainingPair {
            sent_a: "the cat is resting on the living room rug",
            sent_b: "a cute kitten is sleeping comfortably on the floor carpet",
            target_similarity: 1.0,
        },
        TrainingPair {
            sent_a: "how do self attention mechanisms calculate token importance in transformers",
            sent_b: "what is the mathematical formula for scaled dot product attention in transformers",
            target_similarity: 0.9,
        },
        TrainingPair {
            sent_a: "document deduplication identifies identical or paraphrased content in corpora",
            sent_b: "detecting duplicate text documents using semantic cosine similarity",
            target_similarity: 0.95,
        },
        // Low similarity pairs (Different domains / unrelated)
        TrainingPair {
            sent_a: "the cat is resting on the living room rug",
            sent_b: "vector databases store embeddings for high dimensional nearest neighbor search",
            target_similarity: 0.0,
        },
        TrainingPair {
            sent_a: "deep learning models use neural networks to learn representations",
            sent_b: "fresh yellow bananas and ripe organic apples are rich in natural vitamins",
            target_similarity: 0.0,
        },
        TrainingPair {
            sent_a: "quantum mechanics describes the fundamental physical properties of subatomic particles",
            sent_b: "chocolate chip cookies baked with butter and brown sugar are delicious",
            target_similarity: 0.0,
        },
        TrainingPair {
            sent_a: "document deduplication identifies identical or paraphrased content in corpora",
            sent_b: "the solar system contains eight planets orbiting the central sun",
            target_similarity: 0.0,
        },
    ]
}

/// A lightweight, rapid training loop using Cosine-Similarity MSE loss.
/// Uses numerical / direct gradient updates on embeddings and attention projections.
pub fn train_poc_sentence_transformer(
    model: &mut SentenceTransformer,
    epochs: usize,
    learning_rate: f32,
) {
    println!("\n=======================================================");
    println!(" [TRAINING] Starting POC Contrastive Training Loop");
    println!("=======================================================");
    println!(" • Architecture: {} Transformer Layer(s), {} Attention Heads, d_model = {}",
             model.num_layers, model.num_heads, model.d_model);
    println!(" • Training objective: Cosine Similarity MSE Loss");
    println!(" • Number of Epochs: {}", epochs);
    println!(" • Learning Rate: {}", learning_rate);
    println!("-------------------------------------------------------");

    let dataset = get_sample_dataset();
    let mut rng = rand::thread_rng();
    let start_time = Instant::now();

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;
        let mut pairs = dataset.clone();
        pairs.shuffle(&mut rng);

        for pair in &pairs {
            // Forward pass
            let emb1 = model.encode(pair.sent_a).embedding_full;
            let emb2 = model.encode(pair.sent_b).embedding_full;

            let pred_sim = cosine_similarity(&emb1, &emb2);
            let diff = pred_sim - pair.target_similarity;
            let loss = diff * diff;
            total_loss += loss;

            let (tokens_a, _, _) = model.tokenizer.encode(pair.sent_a, model.max_seq_len);
            let (tokens_b, _, _) = model.tokenizer.encode(pair.sent_b, model.max_seq_len);

            // Gradient descent update on token embeddings:
            // Gradient dL/du approx diff * (u - v). Gradient descent step: u -= lr * diff * (u - v) = u + lr * (-diff) * (u - v)
            let step = -diff * learning_rate * 0.02;

            for &id_a in &tokens_a {
                for &id_b in &tokens_b {
                    if id_a < model.token_embedding.rows && id_b < model.token_embedding.rows && id_a > 3 && id_b > 3 {
                        for d in 0..model.d_model {
                            let val_a = model.token_embedding.get(id_a, d);
                            let val_b = model.token_embedding.get(id_b, d);

                            let delta = (val_b - val_a) * step;
                            model.token_embedding.set(id_a, d, val_a + delta);
                            model.token_embedding.set(id_b, d, val_b - delta);
                        }
                    }
                }
            }

            // Slight adaptive tuning to linear projections
            for layer in &mut model.layers {
                for i in 0..layer.attention.w_q.data.len().min(16) {
                    layer.attention.w_q.data[i] -= diff * learning_rate * 0.0005;
                }
            }
        }

        let avg_loss = total_loss / pairs.len() as f32;
        if epoch == 1 || epoch % 20 == 0 || epoch == epochs {
            println!(
                " Epoch {:3}/{} | Avg MSE Loss: {:.6} | Elapsed: {:.2?}",
                epoch,
                epochs,
                avg_loss,
                start_time.elapsed()
            );
        }
    }

    println!("-------------------------------------------------------");
    println!(" Training completed in {:.2?}! Ready for inference.", start_time.elapsed());
    println!("=======================================================\n");
}
