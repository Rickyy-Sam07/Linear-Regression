import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy

df = pd.read_csv(r'C:\Users\DELL\Desktop\projects\linear regression\Linear Regression using SKlearn\data.csv')

x_train = df['population'].values.reshape(-1, 1)
y_train = df['profit'].values

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb_i = w * x[i] + b
        cost = (f_wb_i - y[i])**2
        total_cost += cost
       
    total_cost = total_cost / (2 * m) 
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb_i = w * x[i] + b
        dj_dw += (f_wb_i - y[i]) * x[i]
        dj_db += f_wb_i - y[i]
        
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)  

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        if i % (num_iters // 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
        
    return w, b, J_history, w_history

iterations = 1500
alpha = 0.01

initial_w = 0
initial_b = 0

w, b, J_history, w_history = gradient_descent(x_train_scaled, y_train, initial_w, initial_b, 
                                              compute_cost, compute_gradient, alpha, iterations)

print(f"Updated weight (w): {w}")
print(f"Updated bias (b): {b}")

plt.plot(range(iterations), J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function history')
plt.show()

plt.scatter(x_train, y_train, color='blue', label='Training data')
plt.plot(x_train, w * x_train_scaled + b, color='red', label='Linear regression')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

def predict(population, w, b, scaler):
    population_scaled = scaler.transform(np.array([[population]]))
    profit = w * population_scaled + b
    return profit[0]

populations = [30000, 70000]
for population in populations:
    population_normalized = population / 10000
    profit_prediction = predict(population_normalized, w, b, scaler)
    print(f"Predicted profit for population {population}: {profit_prediction}")
