import os
import sys

import pandas as pd

SYS_SALT_ROOT = "/home/lucaa/audio_data/unc/clap-env"
if SYS_SALT_ROOT not in sys.path:
    sys.path.append(SYS_SALT_ROOT)

from salt import compare_labels  


SALT_MAX_HOPS = 2


def compute_salt_result(clap_label, human_labels):
    try:
        match, results = compare_labels(clap_label, human_labels, max_hops=SALT_MAX_HOPS)

        target = next((r for r in results if r["matched"]), None)
        lca = target.get("lca") if target else None
        return (bool(match), lca)

    except ValueError:
        return (None, None)
    
    except Exception as e:
        print(f"Unexpected error for CLAP: '{clap_label}': {e}")
        return (None, None)


def add_salt_columns(input_path, output_path=None):
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}", flush=True)

    cache = {}
    salt_match = []
    salt_lca = []

    for i, (idx, row) in enumerate(df.iterrows()):
        if i > 0 and i % 100 == 0:
            print(f"Progress: {i}/{len(df)} rows processed", flush=True)

        clap_label = row["clap_labels"]
        human_labels = row["human_labels"]
        key = (clap_label, tuple(human_labels))
        
        if key not in cache:
            cache[key] = compute_salt_result(clap_label, human_labels)
        
        match, lca = cache[key]
        salt_match.append(match)
        salt_lca.append(lca)

    df["salt_match"] = salt_match
    df["salt_lca"] = salt_lca

    print(f"\nUnique (clap_label, human_labels) combinations: {len(cache)}")

    # Save
    if output_path is None:
        output_path = "/home/lucaa/audio_data/unc/clap-env/simulation/salt_matches.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nWrote SALT-augmented results to {output_path}")
    
    return df


def main():
    input_path = "/home/lucaa/audio_data/unc/clap-env/simulation/simulation_results_similar_5000.parquet"
    add_salt_columns(input_path)


if __name__ == "__main__":
    main()


