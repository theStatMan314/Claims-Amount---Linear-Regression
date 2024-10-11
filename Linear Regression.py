# Packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# importing data
Raw = pd.read_csv('C:/data/swedish_insurance.csv')

df = Raw.rename(columns={'X': 'No_Of_Claims', 'Y': 'Claims_Total'})
print(df.head())

# Test correlation
correlation = df['No_Of_Claims'].corr(df['Claims_Total'])
print("Correlation:", correlation)

# Plot the scatter plot for the two variables
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='No_Of_Claims', y='Claims_Total')
plt.title('Scatter Plot of No_Of_Claims vs Claims_Total')
plt.xlabel('No_Of_Claims')
plt.ylabel('Claims_Total')

print("This graph represents the the relationship between the number of claims vs the total claims paid out. As shown the two have a positive linear relationship")

# Boxplot of Variables
df.describe()
plt.figure(figsize=(8, 6))  # Set the figure size
sns.boxplot(data=df[['No_Of_Claims', 'Claims_Total']])  # Create the boxplot
plt.title('Boxplot of No_Of_Claims and Claims_Total')  # Add a title
plt.ylabel('Values')   # Add a y-axis label
plt.xticks(ticks=[0, 1], labels=['No_Of_Claims', 'Claims_Total'])  # Set the x-axis labels
plt.show()  # Display the plot

# hisplot of columns
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
sns.histplot(df['No_Of_Claims'],kde=True)
plt.subplot(2,1,2)
sns.histplot(df['Claims_Total'], kde=True)
plt.tight_layout()
plt.show()

# Linear Modelling 

# split data into X and Y
X= df[['No_Of_Claims']] # NOTE The input to LinearRegression.fit() requires X to be a 2D array
Y= df['Claims_Total']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

# Fit the model with the training data
model.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using R^2 score
r2_score = model.score(X_test, Y_test)
print(f'R^2 Score: {r2_score * 100:.2f}%')

# calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
mae = mean_absolute_error(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Generate predictions for the entire dataset based on 'No_Of_Claims'
y_pred_full = model.predict(df[['No_Of_Claims']])

# Create a scatter plot of the original data and the predicted line
plt.figure(figsize=(5, 4))
sns.scatterplot(x='No_Of_Claims', y='Claims_Total', data=df)

# Plot the predicted line based on the predicted values
plt.plot(df['No_Of_Claims'], y_pred_full, color='red')

# Add a legend and show the plot
plt.legend(['Regression Line', 'Original Data'])
plt.title('Original Data vs Predicted Line')
plt.show()

slope = model.coef_[0]
intercept = model.intercept_

# Print the slope and intercept
print(f"Slope of the regression line: {slope}")
print(f"Intercept of the regression line: {intercept}")

# Predict y for a random x value, say x = 35
random_x = [[35]]  # Input must be 2D for sklearn's predict method
predicted_y = model.predict(random_x)

print(f"Predicted Claims_Total for No_Of_Claims = 35: {predicted_y[0]}")