import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

with open ('student_depression_dataset.csv', 'r') as infile:
    df = pd.read_csv(infile)
    
# pre-cod: makes sure that df != null csv
def train():
        # sklearn expects a 2D structure so use double bracker
    X = df[['Degree', 'Age']]

    # removes multicollinearity 
    X = pd.get_dummies(X, drop_first=True)
    y = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    model = DecisionTreeClassifier()

    # train the model
    model.fit(X_train, y_train)

    # test accuracy score for the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)   
    print(accuracy)
""" 
 post-cod: returns a percentage prediction accuracy using two features
 degree and age to see if you are likely to think about suicide 
"""





print(train())