import numpy as np
import math, copy
import matplotlib.pyplot as plt


# Load the CSV file using numpy
data = np.genfromtxt('Data/CarPrice_Assignment.csv', delimiter=',', skip_header=1)

# Extract the columns into arrays
x_train = data[:, 0]  # Assuming the first column is at index 0
y_train = data[:, 1]  # Assuming the second column is at index 1

x_train /= 1000
y_train /= 1000

m = x_train.shape[0]

print (f"Number of training examples is: {m}")


# plot of x and y (feature and target)
plt.scatter(x_train,y_train, marker='x', c='r')
plt.title("Car prices")
plt.ylabel("Price of a car")
plt.xlabel("MPG of a car")
plt.show()

# function to calculate the cost. We are using mean squared error 
def compute_cost(x,y,w,b):
    m = x.shape[0]    
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1/(2*m) * cost

    return total_cost


def compute_gradient(x,y,w,b):

    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db




def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function,gradient_function):
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):

        dj_dw, dj_db = gradient_function(x,y,w,b)

        # update parameters until you reach the minima (w and b stops changing much)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing
       
#initialize parameters    
w_init = 0
b_init = 0

#some gradient descent settings
iterations = 500000
tmp_alpha = 0.1

# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")



