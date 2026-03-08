import numpy as np
from datasets import load_dataset

from word2vec_numpy import (
    tokenize,
    build_vocab,
    generate_skipgram_pairs,
    NegativeSampler,
    Word2VecSGNS,
)


def row_to_text(example: dict) -> str:
    
    parts = [
        str(example.get("Question", "")),
        str(example.get("Knowledge", "")),
        str(example.get("Ground Truth", "")),
    ]
    return " ".join(p for p in parts if p.strip())


def build_corpus(dataset_split):
    """
    Convert all rows into a list of text documents.
    """
    docs = []
    for example in dataset_split:
        text = row_to_text(example)
        if text.strip():
            docs.append(text)
    return docs


def main():
    

    ds = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_artificial")
    train_ds = ds["train"].select(range(300))

    print(train_ds)

    
    documents = build_corpus(train_ds)
    print("Number of documents:", len(documents))

    
    all_tokens = []
    for doc in documents:
        all_tokens.extend(tokenize(doc))

    print("Number of raw tokens:", len(all_tokens))

    
    word_to_id, id_to_word, token_ids, vocab_counts = build_vocab(all_tokens, min_count=2)

    print("Vocabulary size:", len(word_to_id))
    print("Number of token ids after filtering:", len(token_ids))

    
    pairs = generate_skipgram_pairs(token_ids, window_size=2)
    print("Number of skip-gram pairs:", len(pairs))

    
    sampler = NegativeSampler(vocab_counts)

    
    model = Word2VecSGNS(
        vocab_size=len(word_to_id),
        embedding_dim=50,
        seed=42
    )

    
    model.fit(
        pairs=pairs,
        sampler=sampler,
        epochs=3,
        negative_k=5,
        lr=0.025,
        shuffle=True
    )

    print('saving files')
    np.save("word_embeddings.npy", model.get_embeddings())

   
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for i in range(len(id_to_word)):
            f.write(f"{i}\t{id_to_word[i]}\n")

    print('files saved')
    test_words = ["question", "knowledge", "phospholipases", "mitochondria"]

    print("\nNearest neighbors:")
    for word in test_words:
        if word in word_to_id:
            print(f"\nMost similar to '{word}':")
            for neighbor, score in model.most_similar(word, word_to_id, id_to_word, top_k=5):
                print(f"  {neighbor:15s} {score:.4f}")
        else:
            print(f"\nWord '{word}' not in vocabulary.")


if __name__ == "__main__":
    main()
