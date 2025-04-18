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


# sklearn expects a 2D structure so use double bracker
X = df[['Degree', 'Age']]

# put categorical feature to one hot encoding
# drop_first removes multicollinearity
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


def predict(degree, age):
    # turn our input to dataframe
    df = pd.DataFrame([{'Degree': degree, 'Age': age}])

    # put the dataframe into the one hot encoding  
    X = pd.get_dummies(df)
    # makes sure that every other missing feature in this new data
    # will have be filled with 0
    X = X.reindex(columns=X_train.columns, fill_value=0)
    y_pred = model.predict(X)

    if y_pred[0] == 1: 
        return 'Likely to have thoughts about suicide'
    
    return 'Not likely to have thoughts about suicide'

print(predict('B.Pharm', 20))