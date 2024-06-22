import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd 
import copy
import math

df=pd.read_csv(r'C:\Users\DELL\Desktop\projects\linear regression\data.csv')
print(df)
y=df['profit']
print(y)
x = df['population']
print(x)
x_train =df['population']
y_train=df['profit']
print(x_train)
print(y_train)
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))
m=len(x_train)
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()

def compute_cost(x,y,w,b):
    m=x.shape[0]
    total_cost=0
    for i in range(m):
        f_wb_i=w*x[i] + b
        cost=(f_wb_i - y[i])**2
        total_cost=total_cost+cost
       
    total_cost=(total_cost)/(2*m)

    return total_cost


initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print('cost is',cost)
print(type(cost))

from public_tests import compute_cost_test
compute_cost_test(compute_cost)
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

intital_w=0
initial_b=0

temp_dj_dw,temp_dj_db = compute_gradient (x_train,y_train,initial_w,initial_b)
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
    
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient repeat until convergence
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing

initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha =  1e-7

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

# plot linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')

predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))

# Feature scaling
x_train_scaled = (x_train - np.mean(x_train)) / np.std(x_train)
y_train_scaled = (y_train - np.mean(y_train)) / np.std(y_train)

print("Scaled x_train:", x_train_scaled)
print("Scaled y_train:", y_train_scaled)