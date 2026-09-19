use std::collections::{HashMap, HashSet};

use linalg::Matrix;

pub struct Dataset {
    pub token_to_idx: HashMap<String, usize>,
    pub idx_to_token: HashMap<usize, String>,
    pub vocab_size: usize,
    pub corpus: String,
    pub token_type: TokenType,
}

pub enum TokenType {
    Char,
    Word,
}

impl Dataset {
    pub fn new(corpus: String, token_type: TokenType) -> Self {
        let (token_to_idx, idx_to_token) = match token_type {
            TokenType::Word => get_word_vocab(&corpus),
            TokenType::Char => get_char_vocab(&corpus),
        };
        Self {
            vocab_size: token_to_idx.len(),
            token_to_idx,
            corpus,
            idx_to_token,
            token_type,
        }
    }

    pub fn one_hot(&self, token: &str) -> Matrix {
        let mut m = Matrix::zeros(self.vocab_size, 1);
        let i = self
            .token_to_idx
            .get(token)
            .expect(&format!("Invalid token '{}'", token));
        m.data[*i] = 1.;

        m
    }

    fn get_word_tokens(&self) -> Vec<String> {
        let tokens: Vec<String> = self
            .corpus
            .split_whitespace()
            .map(|x| x.to_string())
            .collect();
        tokens
    }

    fn get_char_tokens(&self) -> Vec<String> {
        let tokens: Vec<String> = self.corpus.chars().map(|c| c.to_string()).collect();
        tokens
    }
    pub fn parse_data(&self) -> Vec<(Matrix, Matrix)> {
        let mut parsed: Vec<(Matrix, Matrix)> = Vec::new();

        let tokens = match self.token_type {
            TokenType::Char => self.get_char_tokens(),
            TokenType::Word => self.get_word_tokens(),
        };

        for i in 0..tokens.len() - 1 {
            let x_c = &tokens[i];
            let y_c = &tokens[i + 1];

            let x = self.one_hot(&x_c);

            let y = self.one_hot(&y_c);

            parsed.push((x, y));
        }

        parsed
    }
}

fn get_char_vocab(corpus: &str) -> (HashMap<String, usize>, HashMap<usize, String>) {
    let tokens: HashSet<String> = HashSet::from(
        corpus
            .chars()
            .map(|x| x.to_string())
            .collect::<HashSet<String>>(),
    );

    let token_to_idx: HashMap<String, usize> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t, i))
        .collect();

    let idx_to_token: HashMap<usize, String> = token_to_idx
        .clone()
        .into_iter()
        .map(|(t, i)| (i, t))
        .collect();

    (token_to_idx, idx_to_token)
}
fn get_word_vocab(corpus: &str) -> (HashMap<String, usize>, HashMap<usize, String>) {
    let tokens: HashSet<String> = HashSet::from(
        corpus
            .split_whitespace()
            .map(|x| x.to_string())
            .collect::<HashSet<String>>(),
    );

    let token_to_idx: HashMap<String, usize> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t, i))
        .collect();

    let idx_to_token: HashMap<usize, String> = token_to_idx
        .clone()
        .into_iter()
        .map(|(t, i)| (i, t))
        .collect();

    (token_to_idx, idx_to_token)
}
