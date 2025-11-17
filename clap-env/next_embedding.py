
import numpy as np
import os

np.random.seed(42) 
temp_embedding = np.random.randn(512)
PREVIOUS_EMBEDDING_CONSTANT = temp_embedding / np.linalg.norm(temp_embedding)

DATASET_FILE = "/home/lucaa/audio_data/unc/clap-env/dataset.npz"
TEXT_DATA_FILE = "/home/lucaa/audio_data/unc/clap-env/text_data.npz"
TARGET_SEED_COUNT = 100 
RADIUS_THRESHOLD = 0.7  

def load_dataset(dataset_path: str = DATASET_FILE):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    audio_embeddings = data["embeddings"]
    paths = data["paths"]
    classified = data["classified"].astype(bool)
    return audio_embeddings, paths, classified

def load_text_data(text_data_path: str = TEXT_DATA_FILE):
    if not os.path.exists(text_data_path):
        raise FileNotFoundError(f"Text data file not found: {text_data_path}")
    data = np.load(text_data_path, allow_pickle=True)
    class_names = data["class_names"]
    text_embeddings = data["embeddings"]
    return class_names, text_embeddings

def find_adaptive_threshold(similarities, target_count):
    sorted_sims = sorted(similarities, reverse=True)
    idx = min(target_count, len(sorted_sims))
    threshold = sorted_sims[idx - 1]
    return threshold


def next_embedding(previous_embedding, mode, cluster_name):

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

    if mode == "similar": ##does it improve speed if we do classify the same things, find paper
        similarities_array = np.dot(unclassified_embeddings, previous_embedding)
        max_similarity_index = np.argmax(similarities_array)
        return unclassified_paths[max_similarity_index]

    if mode == "diverse":
        mean_embedding = classified_embeddings.mean(axis=0)
        similarities = np.dot(unclassified_embeddings, mean_embedding)
        min_similarity_index = np.argmin(similarities)
        return unclassified_paths[min_similarity_index]
    
    if mode == "cluster":
        if cluster_name is None:
            raise ValueError("cluster_name must be provided for cluster mode")
        
        # Load text data and find the text embedding for cluster_name
        class_names, text_embeddings = load_text_data()
        matching_indices = np.where(class_names == cluster_name)[0]
        if len(matching_indices) == 0:
            raise ValueError(f"Class '{cluster_name}' not found in text_data.npz")
        text_embedding = text_embeddings[matching_indices[0]]
        
        # Find top 100 similar audio embeddings and compute centroid
        all_similarities = np.dot(all_embeddings, text_embedding)
        threshold = find_adaptive_threshold(all_similarities, TARGET_SEED_COUNT)
        similar_indices = np.where(all_similarities >= threshold)[0]
        centroid = all_embeddings[similar_indices].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        
        # Compute similarities of unclassified embeddings to centroid
        unclassified_similarities = np.dot(unclassified_embeddings, centroid)
        
        # Filter by RADIUS_THRESHOLD
        above_threshold_mask = unclassified_similarities >= RADIUS_THRESHOLD
        
        if not np.any(above_threshold_mask):
            raise ValueError("No unclassified embeddings meet the similarity threshold")

        filtered_similarities = unclassified_similarities[above_threshold_mask]
        filtered_paths = unclassified_paths[above_threshold_mask]
        best_idx = np.argmax(filtered_similarities)
        return filtered_paths[best_idx]