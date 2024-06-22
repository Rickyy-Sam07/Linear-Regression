# libraries we need
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import math


df = pd.read_csv(r'C:\Users\DELL\Desktop\projects\linear regression\Linear Regression without SKlearn\data.csv')


x_train = df['population'].to_numpy()
y_train = df['profit'].to_numpy()

# normalization usins zscore 
def zscore_normalize_features(X):
    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Normalize the features
x_train_norm, mu, sigma = zscore_normalize_features(x_train)

# Function to compute cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb_i = w * x[i] + b
        cost = (f_wb_i - y[i])**2
        total_cost += cost
    total_cost = total_cost / (2 * m)
    return total_cost

initial_w = 2
initial_b = 1

cost = compute_cost(x_train_norm, y_train, initial_w, initial_b)
print('Cost is', cost)
print(type(cost))

from public_tests import compute_cost_test
compute_cost_test(compute_cost)
# Function to compute gradients
def compute_gradient(x, y, w, b):
    m = len(x)
    dj_db = 0
    dj_dw = 0
    for i in range(m):
        f_wb_i = w * x[i] + b
        dj_db_i = f_wb_i - y[i]
        dj_dw_i = (f_wb_i - y[i]) * x[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

initial_w = 0
initial_b = 0

temp_dj_dw, temp_dj_db = compute_gradient(x_train_norm, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', temp_dj_dw, temp_dj_db)

from public_tests import compute_gradient_test
compute_gradient_test(compute_gradient)
test_w=0.82
test_b=0.82
temp_dj_dw,temp_dj_db = compute_gradient(x_train,y_train,test_w,test_b)
print('Gradient at test w, b:', temp_dj_dw, temp_dj_db)

#  now find the optimal parameters of a linear regression model by using BATCH GRADIENT DECENT.
 #  Recall batch refers to running all the examples in one iteration.
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    w = w_in
    b = b_in
    m = len(x)
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Compute cost and store it for visualization
        if i % 100 == 0:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
            print(f"Iteration {i}: Cost {cost}")

    return w, b, J_history

# Initial parameters for gradient descent
initial_w = 0.   # Initial slope
initial_b = 0.   # Initial intercept
iterations = 1500
alpha = 0.01  # Learning rate

# Run gradient descent with normalized features
w_final, b_final, J_history = gradient_descent(x_train_norm, y_train, initial_w, initial_b,
                                               compute_cost, compute_gradient, alpha, iterations)

# Print optimized parameters
print(f"Optimized parameters: w = {w_final}, b = {b_final}")

# Plotting the cost over iterations
plt.figure(figsize=(10, 6))
plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.grid(True)
plt.show()

# Make predictions and visualize the linear fit
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, marker='x', c='r', label='Training Data')
plt.plot(x_train, w_final * x_train_norm + b_final, c='b', label='Linear Fit')

# Set the title and labels
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.legend()

# Example predictions
populations = [3.5, 7.0]
for population in populations:
    population_norm = (population - mu) / sigma  # Normalize the population for prediction
    profit_prediction = w_final * population_norm + b_final
    print(f'For population = {population * 10000} people, we predict a profit of {profit_prediction :.2f}')

plt.show()
