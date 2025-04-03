import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

with open ('student_depression_dataset.csv', 'r') as infile:
    df = pd.read_csv(infile)
    print(df.head(5))

# lets first find out which of the degree has the most suicides
# this gives us the percentage of Yes or No from each degree.
suicide_deg = df.groupby('Degree')['Have you ever had suicidal thoughts ?'].value_counts(normalize=True)
suicide_deg.loc[:, 'Yes'] # locate only with Yes answers
print(suicide_deg.sort_values(ascending=False))

# Age vs suicidal thoughts
suicide_age = df.groupby('Age')['Have you ever had suicidal thoughts ?'].value_counts()
suicide_age_yes = suicide_age.loc[:, 'Yes']
plt.plot(suicide_age_yes.index, suicide_age_yes.values)
plt.xlabel('Age')
plt.ylabel('Said Yes to Suicidal Thoughts Count')
plt.show()

