import pandas as pd
df = pd.read_parquet('StarWars_Raw_Sentences.parquet')
df.to_csv('StarWars_Raw_Sentences.csv')