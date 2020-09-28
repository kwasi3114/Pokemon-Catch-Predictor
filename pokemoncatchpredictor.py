import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



df = pd.read_csv('file:///Users/kwasidebrah/Documents/pokemon_catch_stats.csv')

df['Captured'] = df['Captured?'] == 'Yes'
print(df)

X = df[['Catch Rate', 'Level']].values
y = df['Captured']


model = LogisticRegression()
model.fit(X,y)
print(model.coef_, model.intercept_)

poke_predictor = model.predict([[45,50]])
print(poke_predictor)

y_pred = model.predict(X)
print(confusion_matrix(y,y_pred))
