import numpy as np
import pandas as pd

total_data = r"data\all_data.csv"
total_data_df = pd.read_csv(total_data)
print(total_data_df.head(5))