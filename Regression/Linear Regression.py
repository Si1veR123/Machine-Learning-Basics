import sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.datasets as datasets

"""
- CRIM     per capita crime rate by town
- ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS    proportion of non-retail business acres per town
- CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX      nitric oxides concentration (parts per 10 million)
- RM       average number of rooms per dwelling
- AGE      proportion of owner-occupied units built prior to 1940
- DIS      weighted distances to five Boston employment centres
- RAD      index of accessibility to radial highways
- TAX      full-value property-tax rate per $10,000
- PTRATIO  pupil-teacher ratio by town
- B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT    % lower status of the population
"""

data = datasets.load_boston()
predict_val = 'CRIM'
predict_pos = np.where(data.feature_names == predict_val)[0]

X = []
for row in data.data:
    row = np.delete(row, predict_pos)
    X.append(row)

y = [row[predict_pos] for row in data.data]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.8)

model = LinearRegression(fit_intercept=True)
model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)

for prediction, result in zip(model.predict(x_test), y_test):
    print(f'Prediction: {prediction}\tResult: {result}')
