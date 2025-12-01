import pandas as pd
from py_salt import event_mapping


e = event_mapping.EventExplorer() #taxonomy in memory


def hierarchical_match(clap_std, audio_std, max_hops: int | None = None):

    if clap_std is None or audio_std is None:
        return False, None

    # exact match
    if clap_std == audio_std:
        return True, clap_std

    # get ALL ancestors in set/list form
    clap_ancestors = e.get_coarse_labels_for_std_label(clap_std)
    audio_ancestors = e.get_coarse_labels_for_std_label(audio_std)

    if clap_std in audio_ancestors:
        return True, clap_std
    if audio_std in clap_ancestors:
        return True, audio_std

    # get paths to label returns a list of paths (list of lists)
    clap_paths = e.get_paths_to_label(clap_std)
    audio_paths = e.get_paths_to_label(audio_std)
    
    # ASSUMPTION: We are using AudioSet labels, which typically form a single tree.
    # We take the first path found which is also the only path found.
    # NOTE: If supporting full poly-hierarchy (e.g. FSD50K mixed with AudioSet), 
    # we should iterate over ALL pairs (p1 in clap_paths, p2 in audio_paths)
    # and find the pair with the minimum hop distance to avoid overestimating distance.
    clap_path = clap_paths[0]
    audio_path = audio_paths[0]

    # Lowest common ancestor
    lca_idx = -1
    lca_label = None
    min_len = min(len(clap_path), len(audio_path))
    
    for i in range(min_len):
        if clap_path[i] == audio_path[i]:
            lca_idx = i
            lca_label = clap_path[i]
        else:
            break
            
    if lca_idx == -1:
        return False, None # No common root

    # Hops = distance from LCA to node1 + distance from LCA to node2
    # dist(LCA, node) = len(path) - 1 - lca_idx
    
    dist_clap = (len(clap_path) - 1) - lca_idx
    dist_audio = (len(audio_path) - 1) - lca_idx
    
    total_hops = dist_clap + dist_audio
    
    return total_hops <= max_hops, lca_label


def compare_labels(clap_label, audioset_list, max_hops: int | None = None):
    clap_std = e.get_std_label_from_dataset_label(clap_label.strip())

    results = []
    match_found = False

    for a_label in audioset_list:

        audio_std = e.get_std_label_from_dataset_label(a_label.strip())
        
        matched, lca = hierarchical_match(clap_std, audio_std, max_hops=max_hops)

        if matched:
            match_found = True

        results.append({
            "audioset_original": a_label,
            "audioset_std": audio_std,
            "clap_std": clap_std,
            "matched": matched,
            "lca": lca
        })

    return match_found, results


# ---------------------------------------------------------------
# 4. Test Cases AI generated
# ---------------------------------------------------------------
if __name__ == "__main__":
    
    # Strict Hops Configuration
    HOPS = 2

    test_data = [
        {
            "clap": "Speech",
            "audioset": ["Speech", "Silence"]
        },
        {
            "clap": "Music",
            "audioset": ["Happy music"]
        },
        {
            "clap": "Cat",
            "audioset": ["Dog"] 
        },
        {
            "clap": "Guitar",
            "audioset": ["Piano"]
        },
        {
            "clap": "Explosion",
            "audioset": ["Lullaby"]
        },
        {
            "clap": "Super Fake Sound 3000",
            "audioset": ["Music"]
        },
        {
            "clap": "Female speech, woman speaking",
            "audioset": ["Female speech, woman speaking"]
        },
        {
            "clap": "Human voice",
            "audioset": ["Shout"]
        },
        {
            "clap": "Narration, monologue",
            "audioset": ["Conversation"]
        },
        {
            "clap": "Male speech, man speaking",
            "audioset": ["Shout"]
        },
        {
            "clap": "Female speech, woman speaking",
            "audioset": ["Music"]
        }
    ]

    print(f"\nRunning SALT comparison with FIXED HOPS = {HOPS}...\n")
    
    output_rows = []
    for item in test_data:
        lca_name = "N/A"
        try:
            match, details = compare_labels(item["clap"], item["audioset"], max_hops=HOPS)
            status = "TRUE" if match else "FALSE"
            
            # Get LCA from details
            target = next((d for d in details if d['matched']), details[0] if details else None)
            if target:
                lca_name = target.get('lca') or "N/A"

        except ValueError:
            status = "ERROR"
            
        output_rows.append({
            "clap": item["clap"],
            "audioset": item["audioset"],
            "result": status,
            "LCA": lca_name
        })

    result_df = pd.DataFrame(output_rows)
    
    # Pretty print
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 40)
    
    print(result_df)
