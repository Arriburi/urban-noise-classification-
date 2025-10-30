import numpy as np
import pandas as pd
import json
import os

# Config
TEXT_CLASS = "Speech"  
RADIUS_THRESHOLD = 0.7  
PRINT_LIMIT = 5  
TARGET_SEED_COUNT = 100  # target number of initial candidates

# File paths
CLASSES_FILE = "/home/lucaa/audio_data/unc/clap-env/classes.json"
TEXT_EMBEDDINGS_FILE = "/home/lucaa/audio_data/unc/clap-env/clap_text_embeddings.npy"
AUDIO_EMBEDDINGS_FILE = "/home/lucaa/audio_data/unc/clap-env/clap_embeddings.npy"
AUDIO_PATHS_FILE = "/home/lucaa/audio_data/unc/clap-env/clap_paths.txt"

# Output
PROTOTYPE_OUTPUT = f"/home/lucaa/audio_data/unc/clap-env/prototype_{TEXT_CLASS.lower()}_clap.npz"
RESULTS_OUTPUT = f"/home/lucaa/audio_data/unc/clap-env/results_{TEXT_CLASS.lower()}_clap.csv"

def load_data():
    with open(CLASSES_FILE, 'r') as f:
        classes = json.load(f)
    text_embeddings = np.load(TEXT_EMBEDDINGS_FILE)
    #print(f"Text embeddings shape: {text_embeddings.shape}")
    audio_embeddings = np.load(AUDIO_EMBEDDINGS_FILE)
    
    # Load audio paths
    with open(AUDIO_PATHS_FILE, 'r', encoding='utf-8') as f:
        audio_paths = [line.strip() for line in f if line.strip()]
    
    return classes, text_embeddings, audio_embeddings, audio_paths

def find_text_embedding(classes, text_embeddings, target_class):
    idx = classes.index(target_class)
    return text_embeddings[idx]

def find_adaptive_threshold(similarities, target_count=1000):
    sorted_sims = sorted(similarities, reverse=True)
    idx = min(target_count, len(sorted_sims))
    threshold = sorted_sims[idx - 1]
    return threshold

def find_similar_audio_files(text_embedding, audio_embeddings, audio_paths, target_count):
    print(f"\n=== CREATE SIMILIAR AUDIO ARRAY ===")
    similarities = np.dot(audio_embeddings, text_embedding)
    threshold = find_adaptive_threshold(similarities, target_count)
    print(f"Using threshold: {threshold}")
    similar_indices = np.where(similarities >= threshold)[0]
    
    #print(f"Found {len(similar_indices)} audio files above threshold {threshold}")
    #print(f"Similarity range: {similarities.min():.4f} to {similarities.max():.4f}")
    
    if len(similar_indices) > 0:
        print(f"\nTop {min(PRINT_LIMIT, len(similar_indices))} most similar files:")
        top_indices = similar_indices[np.argsort(similarities[similar_indices])[::-1]][:PRINT_LIMIT]
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {similarities[idx]:.4f} - {audio_paths[idx]}") ## debug print
    
    return similar_indices, similarities

def compute_centroid(similar_indices, audio_embeddings):
    print(f"\n=== COMPUTING CENTROID ===")

    centroid = audio_embeddings[similar_indices].mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    
    print(f"Computed mean embedding from {len(similar_indices)} similar files")
    
    return centroid

def radius_search(centroid, audio_embeddings, audio_paths, threshold):
    print(f"\n=== RADIUS SEARCH ===")
    print(f"Using radius threshold: {threshold}")
    
    similarities = np.dot(audio_embeddings, centroid)
    radius_indices = np.where(similarities >= threshold)[0]
    
    print(f"Found {len(radius_indices)} files within radius")
    print(f"Similarity range: {similarities.min():.4f} to {similarities.max():.4f}")
    
    results_df = pd.DataFrame({
        'filename': audio_paths,
        'similarity': similarities
    }).sort_values('similarity', ascending=False)
    
    def print_rows(indices, label):
        print(f"\n{label} {len(indices)} files within radius:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {similarities[idx]:.4f} - {audio_paths[idx]}")

    n_to_show = min(PRINT_LIMIT, len(radius_indices))
    if n_to_show > 0:
        # Top N 
        top_indices = radius_indices[np.argsort(similarities[radius_indices])[::-1][:n_to_show]]
        print_rows(top_indices, "Top")

        # Bottom N (just above threshold)
        bottom_indices = radius_indices[np.argsort(similarities[radius_indices])[:n_to_show]]
        print_rows(bottom_indices, "Bottom threshold")
    
    return results_df, radius_indices

def main():
    print(f"=== TEXT-TO-AUDIO PROTOTYPE PIPELINE ===")
    print(f"Target class: {TEXT_CLASS}")
    print(f"Radius search threshold: {RADIUS_THRESHOLD}")
    
    classes, text_embeddings, audio_embeddings, audio_paths = load_data()
    
    # Find text embedding for target class
    text_embedding = find_text_embedding(classes, text_embeddings, TEXT_CLASS)
    if text_embedding is None:
        return
    
    # Find similar audio files using text embedding and compute centroid
    similar_indices, _ = find_similar_audio_files(
        text_embedding, audio_embeddings, audio_paths, TARGET_SEED_COUNT
    )
    centroid = compute_centroid(similar_indices, audio_embeddings)
    if centroid is None:
        return
    
    # Radius search using mean embedding
    results_df, radius_indices = radius_search(
        centroid, audio_embeddings, audio_paths, RADIUS_THRESHOLD
    )

if __name__ == "__main__":
    main()
