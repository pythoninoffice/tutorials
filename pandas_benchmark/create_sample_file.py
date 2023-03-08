import pandas as pd
import numpy as np


def create_dataset(n_rows, n_cols, f_name):
    df = pd.DataFrame(np.random.rand(n_rows, n_cols-3))
    df['integer_id'] = np.random.randint(1, 10000, size=n_rows)
    df['text_id'] = df['integer_id'].astype(str) + '_ID' 
    df['unique_id_text'] = df.index.astype(str) + '_ID'
    df = df.sample(frac =1)
    df.to_csv(f_name)
    
    
    

rows = 100_000_000
cols = 10

create_dataset(rows,cols,'100_million_2.csv')