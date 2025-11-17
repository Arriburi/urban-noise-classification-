import pandas as pd

df = pd.read_parquet("/home/lucaa/audio_data/unc/audioset/audioset_eval.parquet")
print(df.head())    

# print the number of rows and columns
print(df.shape)
# print the column names
print(df.columns)

# print the data types of the columns
print(df.dtypes)

pd.set_option("display.max_colwidth", None)
print(df.file_path.head())