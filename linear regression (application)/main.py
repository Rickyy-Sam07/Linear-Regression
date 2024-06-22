import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv(r'C:\Users\DELL\Desktop\projects\linear regression\linear regression (application)\cars.csv')
y = df['Price']
df = pd.get_dummies(df, columns=['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Owner_Type'])
x = df.drop(columns=['Price', 'Car_ID'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

Lr = LinearRegression()
Lr.fit(x_train, y_train)

ytrain_pred = Lr.predict(x_train)
ytest_pred = Lr.predict(x_test)

Lr_train_mse = mean_squared_error(y_train, ytrain_pred)
Lr_train_r2 = r2_score(y_train, ytrain_pred)
Lr_test_mse = mean_squared_error(y_test, ytest_pred)
Lr_test_r2 = r2_score(y_test, ytest_pred)

Lr_results = pd.DataFrame({'Model': ['Linear regression'],
                            'Train MSE': [Lr_train_mse],
                            'Test MSE': [Lr_test_mse],
                            'Train R2': [Lr_train_r2],
                            'Test R2': [Lr_test_r2]})

r2_train = r2_score(y_train, ytrain_pred)
r2_test = r2_score(y_test, ytest_pred)

plt.figure(figsize=(5, 5))
plt.scatter(x=y_train, y=ytrain_pred, c="#7CAE00", alpha=0.3)
z = np.polyfit(y_train, ytrain_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), '#F8766D')
plt.show()
