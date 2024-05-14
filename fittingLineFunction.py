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


# compute the predicted y
def compute_model_output(x, w, b):
    
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


f_wb = compute_model_output(x_train, -737.6627, 31.8801)

# Plot our model prediction
plt.plot(x_train, f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('MPG (1000 MPG)')
plt.legend()
plt.show()

def make_prediction(x):
    f_wb = (-737.6627) * x + 31.8801

    return f_wb

feature = 0.024

predicted_price = make_prediction(feature)


print(f"MPG: {feature} Predicted price: {predicted_price}")