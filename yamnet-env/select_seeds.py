import numpy as np
import pandas as pd

INPUT_CSV = "/home/lucaa/audio_data/unc/yamnet-env/yamnet_predictions_table.csv"
CLASS_NAME = "Water"
SEED_THRESHOLD = 0.80
SIMILARITY_THRESHOLD = 0.7  # cosine similarity cutoff for reporting
PRINT_SEED_LIMIT = 10       # just for printing

# YAMNet 
YAMNET_MEMMAP_PATH = "/home/lucaa/audio_data/unc/yamnet-env/yamnet_embeddings_memmap.dat"
YAMNET_PATHS_TXT = "/home/lucaa/audio_data/unc/yamnet-env/yamnet_paths.txt"

EMBEDDING_TYPE = "YAMNet"
EMB_DIM = 1024

CENTROID_OUT = f"/home/lucaa/audio_data/unc/yamnet-env/centroid_{CLASS_NAME.lower()}_{EMBEDDING_TYPE.lower()}.npz"

def main():
    
    # Create seeds
    df = pd.read_csv(INPUT_CSV)
    score1 = df["score_1"].where(df["class_1"] == CLASS_NAME, other=np.nan)
    score2 = df["score_2"].where(df["class_2"] == CLASS_NAME, other=np.nan)
    score3 = df["score_3"].where(df["class_3"] == CLASS_NAME, other=np.nan)
    best_score = pd.concat([score1, score2, score3], axis=1).max(axis=1)

    f = df.loc[best_score >= SEED_THRESHOLD, ["filename"]].copy()
    f["score"] = best_score.loc[f.index].values
    f = f.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    seeds = f["filename"].tolist()
    print("=== STEP 1: SEED SELECTION ===")
    print(f"Using {EMBEDDING_TYPE} embeddings")
    print(f"Seeds selected: {len(seeds)} for class '{CLASS_NAME}' (threshold >= {SEED_THRESHOLD}).")

    print("\n=== STEP 2: CENTROID COMPUTATION ===")
    
    if seeds:
        with open(YAMNET_PATHS_TXT, 'r', encoding='utf-8') as fh:
            file_paths = [line.strip() for line in fh if line.strip()]
        path_to_idx = {p: i for i, p in enumerate(file_paths)}

        seed_indices = []
        unmapped = []
        for p in seeds:
            idx = path_to_idx.get(p)
            if idx is None:
                unmapped.append(p)
            else:
                seed_indices.append(idx)

        if seed_indices:
            normalized_embeddings = np.memmap(YAMNET_MEMMAP_PATH, 
                                             dtype="float32", mode="r", shape=(len(file_paths), EMB_DIM))
            
            seed_vecs = np.array(normalized_embeddings[seed_indices])
            centroid = seed_vecs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            print(f"Mapped seeds: {len(seed_indices)} | Unmapped: {len(unmapped)}")
            
            print("\n=== STEP 3: SIMILARITY COMPUTATION ===")
            # Find similar files using the computed centroid
            find_similar_files(centroid, seed_indices, normalized_embeddings, file_paths)
        else:
            print(f"No seeds mapped to YAMNet embeddings; centroid not computed.")


def find_similar_files(centroid, seed_indices, normalized_embeddings, file_paths):
    print(f"Finding similar files using centroid (based on {len(seed_indices)} seeds)")
    
    # Compute cosine similarities
    similarities = np.dot(normalized_embeddings, centroid)
    
    # Create DataFrame (sorted by similarity, highest first)
    import pandas as pd
    results_df = pd.DataFrame({
        'filename': file_paths,
        'similarity': similarities
    })
    results_df = results_df.sort_values('similarity', ascending=False)
    
    print(f"Computed similarities for {len(similarities)} files")
    print(f"Similarity range: {similarities.min():.4f} to {similarities.max():.4f}")
    
    # Use configured similarity threshold
    threshold = SIMILARITY_THRESHOLD
    print(f"\n=== THRESHOLD RESULTS ===")
    print(f"Threshold: {threshold:.4f}")
    print(f"Files above threshold: {(similarities >= threshold).sum()}")
    # Bottom 5 among those above threshold
    above_idx = np.where(similarities >= threshold)[0]
    if above_idx.size > 0:
        order_asc = np.argsort(similarities[above_idx])
        bottom_n = min(5, above_idx.size)
        print(f"\n=== BOTTOM {bottom_n} ABOVE THRESHOLD ===")
        for rank in range(bottom_n):
            idx = above_idx[order_asc[rank]]
            print(f"{rank+1}. {similarities[idx]:.4f} - {file_paths[idx]}")
    
    # Debug: Show a limited number of seed file similarities (but all seeds were used)
    #print(f"\n=== SEED FILE SIMILARITIES (showing up to {PRINT_SEED_LIMIT}) ===")
    #for i, seed_idx in enumerate(seed_indices[:PRINT_SEED_LIMIT]):
    #    print(f"Seed {i+1}: {similarities[seed_idx]:.4f} - {file_paths[seed_idx]}")
    
    # Show top 5 and bottom 5 files
    print(f"\n=== TOP 5 MOST SIMILAR FILES ===")
    for i, (_, row) in enumerate(results_df.head(10).iterrows()):
        print(f"{i+1}. {row['similarity']:.4f} - {row['filename']}")
    
    return results_df


if __name__ == "__main__":
    main()


