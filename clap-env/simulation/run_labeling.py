from strategies import get_next_similiar, get_next_max_min, get_next_cluster

import os
import pandas as pd

PARQUET_PATH = "/home/lucaa/audio_data/unc/audioset/eval_set_flac.parquet"
def load_parquet(path=PARQUET_PATH):
  return pd.read_parquet(path)
