import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

with open ('student_depression_dataset.csv', 'r') as infile:
    df = pd.read_csv(infile)
    print(df.head(5))