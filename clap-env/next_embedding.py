import numpy as np
import os

np.random.seed(42) 
temp_embedding = np.random.randn(512)
PREVIOUS_EMBEDDING_CONSTANT = temp_embedding / np.linalg.norm(temp_embedding)

DATASET_FILE = "/home/lucaa/audio_data/unc/clap-env/dataset.npz"

def load_dataset(dataset_path: str = DATASET_FILE):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    embeddings = data["embeddings"]
    paths = data["paths"]
    classified = data["classified"].astype(bool)
    return embeddings, paths, classified


def next_embedding(previous_embedding, mode):

    all_embeddings, all_paths, classified = load_dataset()

    if mode not in {"similar", "diverse", "cluster"}:
        raise ValueError(f"Invalid mode: {mode}. Must be 'similar', 'diverse', or 'cluster'")

    unclassified_mask = ~classified
    if not np.any(unclassified_mask):
        raise ValueError("No unclassified embeddings remaining")

    unclassified_embeddings = all_embeddings[unclassified_mask]
    unclassified_paths = all_paths[unclassified_mask]

    classified_embeddings = all_embeddings[classified]
    classified_paths = all_paths[classified]

    if mode == "similar":
        similarities_array = np.dot(unclassified_embeddings, previous_embedding)
        max_similarity_index = np.argmax(similarities_array)
        return unclassified_paths[max_similarity_index]

    if mode == "diverse":
        mean_embedding = classified_embeddings.mean(axis=0)
        similarities = np.dot(unclassified_embeddings, mean_embedding)
        min_similarity_index = np.argmin(similarities)
        return unclassified_paths[min_similarity_index]
    if mode == "cluster":
        return None ## cluster mode not implemented