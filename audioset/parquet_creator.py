from pathlib import Path

import pandas as pd

CSV_PATH = Path("/home/lucaa/audio_data/unc/audioset/eval_segments.csv")
FLAC_DIR = Path("/home/lucaa/audio_data/unc/audioset/eval_set_flac")
OUTPUT_PATH = Path("/home/lucaa/audio_data/unc/audioset/audioset_eval.parquet")


def main() -> None:
    df = pd.read_csv(
        CSV_PATH,
        comment="#",
        header=None,
        names=["video_id", "start_seconds", "end_seconds", "positive_labels"],
        engine="python",
        skipinitialspace=True,
    )

    available = {p.stem.split("_", 1)[1]: p for p in FLAC_DIR.glob("*.flac")}
    df = df[df["video_id"].isin(available)].copy()
    df["file_path"] = df["video_id"].map(lambda vid: available[vid].as_posix())
    df["human_labels"] = (
        df["positive_labels"]
        .str.strip()
        .str.strip('"')
        .str.split(",")
        .apply(lambda labels: [label.strip() for label in labels if label])
    )
    df["embedding"] = [None] * len(df)
    df["clap_labels"] = [[] for _ in range(len(df))]
    df["clap_score"] = [None] * len(df)
    df["is_classified"] = False
    df = df[
        [
            "video_id",
            "file_path",
            "embedding",
            "human_labels",
            "clap_labels",
            "clap_score",
            "is_classified",
        ]
    ]
    df.to_parquet(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()

