import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler  #optional

# 1. Load and Explore the Data
housing = fetch_california_housing()
X = housing.data[:, 0].reshape(-1, 1)  # only use the medinc feature 
y = housing.target  #median house value (scaled to $100,000s)

#scatter plot of medinc vs. medhouseval 
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.3, label='Data points')
plt.xlabel("Median Income (in $10,000s)")
plt.ylabel("Median House Value (in $100,000s)")
plt.title("California Housing: Median Income vs. House Value")
plt.legend()
plt.show()

#calculate summary statistics
print("Summary Statistics for Median Income (MedInc):")
print("Mean:", np.mean(X))
print("Median:", np.median(X))
print("Standard Deviation:", np.std(X))
print("\nSummary Statistics for Median House Value (MedHouseVal):")
print("Mean:", np.mean(y))
print("Median:", np.median(y))
print("Standard Deviation:", np.std(y))

# 2. Preprocess the Data
#split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Optional: Standardize the feature using StandardScaler (note: linear regression coefficients are interpretable without standardization).
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# 3. Build a Linear Regression Model
#batch gradient descent
def batch_gradient_descent(X, y, lr=0.01, n_iters=1000):
    """
    Performs batch gradient descent for linear regression.
    Returns the slope (w) and intercept (b).
    """
    m = len(y)
    w = 0.0
    b = 0.0

    for i in range(n_iters):
        #predicted values
        y_pred = w * X.flatten() + b
        #compute gradients
        dw = (1/m) * np.sum((y_pred - y) * X.flatten())
        db = (1/m) * np.sum(y_pred - y)
        #update parameters
        w = w - lr * dw
        b = b - lr * db
        
        #print loss every 100 iterations
        if i % 100 == 0:
            loss = (1/(2*m)) * np.sum((y_pred - y)**2)
            print(f"Batch GD Iteration {i}: Loss = {loss:.4f}")
    
    return w, b

#stochastic gradient descent 
def stochastic_gradient_descent(X, y, lr=0.01, n_epochs=50):
    """
    Performs stochastic gradient descent for linear regression.
    Returns the slope (w) and intercept (b).
    """
    m = len(y)
    w = 0.0
    b = 0.0

    for epoch in range(n_epochs):
        # shuffle data at the beginning of each epoch
        indices = np.arange(m)
        np.random.shuffle(indices)
        for i in indices:
            xi = X[i, 0]
            yi = y[i]
            #compute prediction and error for sample
            y_pred_i = w * xi + b
            error = y_pred_i - yi
            #update parameters using the gradient computed from single sample
            w = w - lr * error * xi
            b = b - lr * error
        #print loss at each epoch
        y_pred = w * X.flatten() + b
        loss = (1/(2*m)) * np.sum((y_pred - y)**2)
        print(f"SGD Epoch {epoch+1}: Loss = {loss:.4f}")
    
    return w, b

#train using batch gradient descent
print("\nTraining using Batch Gradient Descent:")
w_batch, b_batch = batch_gradient_descent(X_train, y_train, lr=0.05, n_iters=1000)
print(f"Batch GD parameters: w = {w_batch:.4f}, b = {b_batch:.4f}")

#train using stochastic gradient descent
print("\nTraining using Stochastic Gradient Descent:")
w_sgd, b_sgd = stochastic_gradient_descent(X_train, y_train, lr=0.01, n_epochs=50)
print(f"SGD parameters: w = {w_sgd:.4f}, b = {b_sgd:.4f}")

# 4. Make Predictions
#batch gradient descent model to predict test set values
y_pred_batch = w_batch * X_test.flatten() + b_batch
#predict house value for a district with median income of $80,000
income_value = 8.0
predicted_value_batch = w_batch * income_value + b_batch
print(f"\nPredicted median house value (Batch GD) for MedInc = {income_value}: {predicted_value_batch:.4f} (in $100,000s)")
#stochastic gradient descent  to predict test set values
y_pred_sgd = w_sgd * X_test.flatten() + b_sgd
predicted_value_sgd = w_sgd * income_value + b_sgd
print(f"Predicted median house value (SGD) for MedInc = {income_value}: {predicted_value_sgd:.4f} (in $100,000s)")

# 5. Visualize the Results
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, alpha=0.3, label='Test Data')
#create sequence of values from min to max medinc in test set to plot regression lines
x_line = np.linspace(X_test.min(), X_test.max(), 100)
#regression line from batch gradient descent
y_line_batch = w_batch * x_line + b_batch
plt.plot(x_line, y_line_batch, color='red', linewidth=2, label='Regression Line (Batch GD)')
#regression line from stochastic gradient descent
y_line_sgd = w_sgd * x_line + b_sgd
plt.plot(x_line, y_line_sgd, color='green', linewidth=2, label='Regression Line (SGD)')

plt.xlabel("Median Income (in $10,000s)")
plt.ylabel("Median House Value (in $100,000s)")
plt.title("Linear Regression: Median Income vs. House Value (Test Set)")
plt.legend()
plt.show()

