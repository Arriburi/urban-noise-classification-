import pandas as pd
import numpy as np
import os


def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-12)

def get_next_similiar(previous_embedding, df):
    unclassified_mask = df["is_classified"] == False
    unclassified_df = df[unclassified_mask]
    
    if unclassified_df.empty:
        return None

    unclassified_embeddings = np.stack(unclassified_df["embedding"].values)
    
    unclassified_embeddings = normalize(unclassified_embeddings)
    previous_embedding_norm = normalize(previous_embedding)
    

    similarities_array = np.dot(unclassified_embeddings, previous_embedding_norm.T)
    

    max_similarity_index = np.argmax(similarities_array)
    
    return unclassified_df.index[max_similarity_index]

def get_next_max_min(df):

    unclassified_mask = df["is_classified"] == False
    unclassified_df = df[unclassified_mask]
    
    if unclassified_df.empty:
        return None


    classified_mask = df["is_classified"] == True
    classified_df = df[classified_mask]
    
    if classified_df.empty:
        return unclassified_df.sample(1).index[0]
        
 
    U = np.stack(unclassified_df["embedding"].values)
    C = np.stack(classified_df["embedding"].values)
    
    # normalize embeddings
    U = normalize(U)
    C = normalize(C)
    
    # dot product of unclassified and classified embeddings
    similarities_matrix = np.dot(U, C.T)
    
    # find the max similarity for each unclassified item (nearest neighbor)
    max_sims = np.max(similarities_matrix, axis=1)
    
    # find the item with the lowest max similarity (furthest from any neighbor)
    best_candidate_idx = np.argmin(max_sims)
    
    return unclassified_df.index[best_candidate_idx]

def get_next_cluster():
    
    pass
