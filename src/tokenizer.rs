use std::collections::HashMap;

pub const PAD_TOKEN: &str = "[PAD]";
pub const UNK_TOKEN: &str = "[UNK]";
pub const CLS_TOKEN: &str = "[CLS]";
pub const SEP_TOKEN: &str = "[SEP]";

pub const PAD_ID: usize = 0;
pub const UNK_ID: usize = 1;
pub const CLS_ID: usize = 2;
pub const SEP_ID: usize = 3;

#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    pub word2id: HashMap<String, usize>,
    pub id2word: Vec<String>,
}

impl SimpleTokenizer {
    pub fn new() -> Self {
        let mut tokenizer = Self {
            word2id: HashMap::new(),
            id2word: Vec::new(),
        };

        // Add special tokens in fixed order
        tokenizer.add_word(PAD_TOKEN);
        tokenizer.add_word(UNK_TOKEN);
        tokenizer.add_word(CLS_TOKEN);
        tokenizer.add_word(SEP_TOKEN);

        tokenizer
    }

    pub fn add_word(&mut self, word: &str) -> usize {
        let clean = word.to_lowercase();
        if let Some(&id) = self.word2id.get(&clean) {
            id
        } else {
            let id = self.id2word.len();
            self.word2id.insert(clean.clone(), id);
            self.id2word.push(clean);
            id
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id2word.len()
    }

    /// Pre-train/build vocabulary on a list of corpus sentences
    pub fn build_vocab(&mut self, sentences: &[&str]) {
        for sentence in sentences {
            let tokens = self.tokenize_raw(sentence);
            for token in tokens {
                self.add_word(&token);
            }
        }
    }

    /// Split text into tokens (lowercase words and punctuation)
    pub fn tokenize_raw(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() {
                current_word.push(ch.to_ascii_lowercase());
            } else {
                if !current_word.is_empty() {
                    tokens.push(current_word.clone());
                    current_word.clear();
                }
                if !ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
            }
        }
        if !current_word.is_empty() {
            tokens.push(current_word);
        }

        tokens
    }

    /// Encode sentence into token IDs with [CLS] and [SEP], with padding and attention mask
    pub fn encode(&self, text: &str, max_len: usize) -> (Vec<usize>, Vec<f32>, Vec<String>) {
        let raw_tokens = self.tokenize_raw(text);
        let mut token_strings = vec![CLS_TOKEN.to_string()];
        let mut token_ids = vec![CLS_ID];

        for t in raw_tokens {
            if token_ids.len() + 1 >= max_len {
                // Leave room for [SEP]
                break;
            }
            let id = self.word2id.get(&t).copied().unwrap_or(UNK_ID);
            token_ids.push(id);
            token_strings.push(t);
        }

        token_ids.push(SEP_ID);
        token_strings.push(SEP_TOKEN.to_string());

        let actual_len = token_ids.len();
        let mut attention_mask = vec![1.0; actual_len];

        // Pad to max_len
        while token_ids.len() < max_len {
            token_ids.push(PAD_ID);
            attention_mask.push(0.0);
            token_strings.push(PAD_TOKEN.to_string());
        }

        (token_ids, attention_mask, token_strings)
    }
}
