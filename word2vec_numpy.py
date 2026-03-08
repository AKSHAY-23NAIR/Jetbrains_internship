import re
import math
import random
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np


# ---------------------------------------------------
# Text preprocessing
# ---------------------------------------------------

def tokenize(text: str) -> List[str]:
    
    text = text.lower()
    return re.findall(r"[a-z']+", text)


def build_vocab(tokens: List[str], min_count: int = 2):
   
    word_counts = Counter(tokens)

    vocab_words = [w for w, c in word_counts.items() if c >= min_count]
    vocab_words.sort()

    word_to_id = {w: i for i, w in enumerate(vocab_words)}
    id_to_word = {i: w for w, i in word_to_id.items()}

    token_ids = [word_to_id[w] for w in tokens if w in word_to_id]
    vocab_counts = Counter(token_ids)

    return word_to_id, id_to_word, token_ids, vocab_counts


def generate_skipgram_pairs(token_ids: List[int], window_size: int) -> List[Tuple[int, int]]:
    
    pairs = []
    n = len(token_ids)

    for i, center_id in enumerate(token_ids):
        left = max(0, i - window_size)
        right = min(n, i + window_size + 1)

        for j in range(left, right):
            if i == j:
                continue
            context_id = token_ids[j]
            pairs.append((center_id, context_id))

    return pairs


# ---------------------------------------------------
# Math helpers
# ---------------------------------------------------

def sigmoid(x):
   
    x = np.clip(x, -15, 15)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------
# Negative sampler
# ---------------------------------------------------

class NegativeSampler:
   

    def __init__(self, vocab_counts: Counter):
        self.vocab_size = len(vocab_counts)

        freqs = np.zeros(self.vocab_size, dtype=np.float64)
        for idx, count in vocab_counts.items():
            freqs[idx] = count

        probs = np.power(freqs, 0.75)
        probs /= probs.sum()
        self.probs = probs

    def sample(self, k: int, forbidden: int = None) -> np.ndarray:
        
        result = []
        while len(result) < k:
            candidates = np.random.choice(self.vocab_size, size=k, p=self.probs)
            for c in candidates:
                if forbidden is not None and c == forbidden:
                    continue
                result.append(int(c))
                if len(result) == k:
                    break
        return np.array(result, dtype=np.int64)


# ---------------------------------------------------
# Word2Vec Skip-Gram with Negative Sampling
# ---------------------------------------------------

class Word2VecSGNS:
    def __init__(self, vocab_size: int, embedding_dim: int = 50, seed: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        rng = np.random.default_rng(seed)

        # Input embeddings: center word vectors
        self.W_in = rng.normal(0, 0.01, size=(vocab_size, embedding_dim)).astype(np.float64)

        # Output embeddings: context word vectors
        self.W_out = np.zeros((vocab_size, embedding_dim), dtype=np.float64)

    def train_one_pair(self, center_id: int, context_id: int, negative_ids: np.ndarray, lr: float) -> float:
        

        # ---- Embedding lookup ----
        v_c = self.W_in[center_id].copy()         # shape: (D,)
        u_o = self.W_out[context_id].copy()       # shape: (D,)
        u_neg = self.W_out[negative_ids].copy()   # shape: (K, D)

        # ---- Forward pass ----
        pos_score = np.dot(u_o, v_c)              # scalar
        neg_scores = np.dot(u_neg, v_c)           # shape: (K,)

        pos_sig = sigmoid(pos_score)
        neg_sig = sigmoid(neg_scores)

        # ---- Loss ----
        eps = 1e-10
        pos_loss = -math.log(pos_sig + eps)
        neg_loss = -np.sum(np.log(sigmoid(-neg_scores) + eps))
        loss = pos_loss + neg_loss

        # ---- Gradients ----
        # d/ds [-log(sigmoid(s))] = sigmoid(s) - 1
        grad_pos_score = pos_sig - 1.0  # scalar

        # d/ds [-log(sigmoid(-s))] = sigmoid(s)
        grad_neg_scores = neg_sig       # shape: (K,)

        # Gradient wrt center embedding
        grad_v_c = grad_pos_score * u_o + np.sum(grad_neg_scores[:, None] * u_neg, axis=0)

        # Gradient wrt positive output embedding
        grad_u_o = grad_pos_score * v_c

        # Gradient wrt negative output embeddings
        grad_u_neg = grad_neg_scores[:, None] * v_c[None, :]

        # ---- Parameter update ----
        self.W_in[center_id] -= lr * grad_v_c
        self.W_out[context_id] -= lr * grad_u_o

        # Update negatives one by one (duplicates may appear)
        for i, neg_id in enumerate(negative_ids):
            self.W_out[neg_id] -= lr * grad_u_neg[i]

        return float(loss)

    def fit(
        self,
        pairs: List[Tuple[int, int]],
        sampler: NegativeSampler,
        epochs: int = 3,
        negative_k: int = 5,
        lr: float = 0.025,
        shuffle: bool = True,
    ):
        
        #Main training loop.
        
        for epoch in range(epochs):
            if shuffle:
                random.shuffle(pairs)

            total_loss = 0.0

            for center_id, context_id in pairs:
                negative_ids = sampler.sample(negative_k, forbidden=context_id)
                loss = self.train_one_pair(center_id, context_id, negative_ids, lr)
                total_loss += loss

            avg_loss = total_loss / len(pairs)
            print(f"Epoch {epoch + 1}/{epochs} | avg loss = {avg_loss:.4f}")

    def get_embeddings(self) -> np.ndarray:
       
        return self.W_in

    def most_similar(self, word: str, word_to_id: Dict[str, int], id_to_word: Dict[int, str], top_k: int = 5):
        
        #Find nearest neighbors by cosine similarity.
        
        if word not in word_to_id:
            raise ValueError(f"Word '{word}' not in vocabulary.")

        W = self.get_embeddings()
        idx = word_to_id[word]

        target = W[idx]
        target_norm = np.linalg.norm(target) + 1e-10
        all_norms = np.linalg.norm(W, axis=1) + 1e-10

        sims = (W @ target) / (all_norms * target_norm)
        sims[idx] = -np.inf

        best = np.argsort(-sims)[:top_k]
        return [(id_to_word[i], float(sims[i])) for i in best]
