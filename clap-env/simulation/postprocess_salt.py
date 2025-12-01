import os
import sys

import pandas as pd

SYS_SALT_ROOT = "/home/lucaa/audio_data/unc/clap-env"
if SYS_SALT_ROOT not in sys.path:
    sys.path.append(SYS_SALT_ROOT)

from salt import compare_labels  


SALT_MAX_HOPS = 2  


def compute_salt_for_row(clap_label, human_labels, cache):
    key = (clap_label, tuple(human_labels)) #cache retrieval
    if key in cache:
        return cache[key]

    match, results = compare_labels(clap_label, human_labels, max_hops=SALT_MAX_HOPS)

    target = next((r for r in results if r["matched"]), None)
    lca = target.get("lca") if target else None

    result = (bool(match), lca)
    cache[key] = result
    return result


def add_salt_columns(input_path, output_path=None):
    """
    Load a simulation_results parquet, compute SALT match + LCA per row,
    and write out a new parquet with added columns:
      - salt_match (bool)
      - salt_lca (str or None)
    """
    df = pd.read_parquet(input_path)

    cache = {}
    salt_match = []
    salt_lca = []

    for _, row in df.iterrows():
        clap_label = row["clap_labels"]
        human_labels = row["human_labels"]

        match, lca = compute_salt_for_row(clap_label, human_labels, cache)
        salt_match.append(match)
        salt_lca.append(lca)

    df["salt_match"] = salt_match
    df["salt_lca"] = salt_lca

    output_path = "/home/lucaa/audio_data/unc/clap-env/simulation/salt_matches.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Wrote SALT-augmented results to {output_path}")


def main():
    input_path = "/home/lucaa/audio_data/unc/clap-env/simulation/simulation_results_similar_10.parquet"
    add_salt_columns(input_path)


if __name__ == "__main__":
    main()


