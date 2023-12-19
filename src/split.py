import pandas as pd
import numpy as np


df = pd.read_csv('training.csv', encoding='gbk')
df = df.sample(frac=1, random_state=42)


train_size = int(0.8 * len(df))
val_size = (len(df) - train_size) // 2

train_df = df[:train_size]
val_df = df[train_size: train_size + val_size]
test_df = df[train_size + val_size:]

with open('../model/training.txt', 'w', encoding='utf-8') as f:
    for index, row in train_df.iterrows():
        dummy_value = 1 if pd.notna(row['dummy']) and row['dummy'] == '1' else 0
        f.write(f"{dummy_value}  {row['content']}\n")

with open('../model/validation.txt', 'w', encoding='utf-8') as f:
    for index, row in val_df.iterrows():
        dummy_value = 1 if pd.notna(row['dummy']) and row['dummy'] == '1' else 0
        f.write(f"{dummy_value}  {row['content']}\n")

with open('../model/test.txt', 'w', encoding='utf-8') as f:
    for index, row in test_df.iterrows():
        dummy_value = 1 if pd.notna(row['dummy']) and row['dummy'] == '1' else 0
        f.write(f"{dummy_value}  {row['content']}\n")
