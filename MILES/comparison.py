from scipy.spatial.distance import cosine
from fastdtw import fastdtw

def compare_actions(emb_seq_a, emb_seq_b):
    """
    emb_seq_a/b: Numpy arrays of shape (num_windows, 128)
    """
    
    # --- 1. Global Embedding Similarity ---
    # We take the "centroid" (average) of the entire session
    avg_a = np.mean(emb_seq_a, axis=0)
    avg_b = np.mean(emb_seq_b, axis=0)
    
    # Cosine similarity is 1 - cosine distance
    global_sim = 1 - cosine(avg_a, avg_b)
    
    # --- 2. Temporal Embedding DTW ---
    # Measures how well the 'rhythm' and 'sequence' matches
    distance, path = fastdtw(emb_seq_a, emb_seq_b, dist=cosine)
    
    # Normalize DTW distance by path length to make it comparable across different lengths
    normalized_dtw_score = distance / len(path)
    # Convert to a 0-1 scale where 1 is perfect consistency
    temporal_consistency = 1 / (1 + normalized_dtw_score)

    return global_sim, temporal_consistency

# Example usage:
# sim, consistency = compare_actions(session1_embs, session2_embs)
print(f"Global Form Similarity: {sim:.2%}")
print(f"Temporal Rhythm Consistency: {consistency:.2%}")