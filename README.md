
## Linear Regression Project


# Simple Linear Regression

This project focuses on building a linear regression model to predict insurance claim totals based on the number of claims, using Swedish insurance data. The following explains each step of the code.

### Importing Packages
We first import the necessary libraries:

- pandas for data manipulation.
- seaborn and matplotlib for visualizations.
- numpy for numerical operations.
- scikit-learn tools for linear regression, data splitting, and model evaluation.
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

```

### Importing and Preprocessing the Data
We load the dataset into a pandas DataFrame and rename the columns to more meaningful names:


```python
Raw = pd.read_csv('C:/VS Code/data/swedish_insurance.csv')
df = Raw.rename(columns={'X': 'No_Of_Claims', 'Y': 'Claims_Total'})
print(df.head())
```
![image](https://github.com/user-attachments/assets/45cc6617-6910-4fe1-b7d0-26f0107b9d24)


### Testing Correlation Between Variables


```python
correlation = df['No_Of_Claims'].corr(df['Claims_Total'])
print("Correlation:", correlation)

```
Correlation: 0.9128782350234075 

This calculates the Pearson correlation coefficient between the two variables. Represents the the relationship between the number of claims vs the total claims paid out. As shown the two have a positive linear relationship

### Scatter Plot of No_Of_Claims vs Claims_Total


```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='No_Of_Claims', y='Claims_Total')
plt.title('Scatter Plot of No_Of_Claims vs Claims_Total')
plt.xlabel('No_Of_Claims')
plt.ylabel('Claims_Total')
```
![image](https://github.com/user-attachments/assets/1e9fef6b-5dc8-45ff-85b7-6132cd40cf11)

We plot a scatter graph to visually inspect the relationship between No_Of_Claims and Claims_Total. The positive trend suggests that these two variables are likely linearly related.

###  Boxplot for No_Of_Claims and Claims_Total


```python
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['No_Of_Claims', 'Claims_Total']])
plt.title('Boxplot of No_Of_Claims and Claims_Total')
plt.ylabel('Values')
plt.xticks(ticks=[0, 1], labels=['No_Of_Claims', 'Claims_Total'])
plt.show()

```
![image](https://github.com/user-attachments/assets/237223d4-2955-47ad-9237-f5c35372040a)

We create boxplots for both variables to observe their distribution and detect any outliers.

### Histogram for Distribution of Columns


```python
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
sns.histplot(df['No_Of_Claims'], kde=True)
plt.subplot(2,1,2)
sns.histplot(df['Claims_Total'], kde=True)
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/684eb36b-5cff-4e58-b91e-36b474bbd4a6)

Histograms are created to visualize the distribution of the No_Of_Claims and Claims_Total

## Building the Model

### Splitting Data for Linear Regression
We split the dataset into training and test sets. The train_test_split method from sklearn helps to split 80% of the data for training and 20% for testing.
```python
X= df[['No_Of_Claims']]
Y= df['Claims_Total']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

### Training the Linear Regression Model
We create and train the linear regression model using the training data.

```python
model = LinearRegression()
model.fit(X_train, Y_train)
```
### Evaluating the Model
We predict on the test data and calculate the R² score, which measures how well the model explains the variance in the data.

```python
y_pred = model.predict(X_test)
r2_score = model.score(X_test, Y_test)
print(f'R^2 Score: {r2_score * 100:.2f}%')

```
![image](https://github.com/user-attachments/assets/a0d6e0a4-0d54-4a2e-baf6-a05a7ff9e893)

This means that 89.51% of the variability in the outcome can be explained by our predictors in the model. In other words, the model is quite good at explaining the variation in the data, as most of the changes in the outcome are captured by the independent variables.

### Calculating MAE and RMSE
We also compute the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to further evaluate the model's accuracy.
```python
mae = mean_absolute_error(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
```
![image](https://github.com/user-attachments/assets/8c7b24ae-26b4-401f-9fbc-1e50e0bb3b7d)

Given that our MAE is 26.41, this indicates that, on average, the absolute error between our predicted values and actual values is around 26.41. This suggests that while there is some error in our predictions, it is within a reasonable range considering the nature of the data and the magnitude of the target values.
Furthermore, our RMSE is approximately 29.58. Since RMSE squares the errors, it penalizes larger deviations more heavily, making it more sensitive to outliers. The fact that our RMSE is only slightly higher than our MAE suggests that there are no significant outliers disproportionately affecting the model’s performance

### Visualising the Regression Line
Plot the original data points alongside the regression line, which is derived from the model’s predictions.

```python
y_pred_full = model.predict(df[['No_Of_Claims']])
plt.figure(figsize=(5, 4))
sns.scatterplot(x='No_Of_Claims', y='Claims_Total', data=df)
plt.plot(df['No_Of_Claims'], y_pred_full, color='red')
plt.legend(['Regression Line', 'Original Data'])
plt.title('Original Data vs Predicted Line')
plt.show()

```
![image](https://github.com/user-attachments/assets/3f9c5652-aff7-44d3-b228-0aabba2f95be)

### Predicting Claims Total (£) for a Specific Number of Claims
Using our model to predict outcomes:

```python
slope = model.coef_[0]
intercept = model.intercept_

# Print the slope and intercept
print(f"Slope of the regression line: {slope}")
print(f"Intercept of the regression line: {intercept}")

# Predict y for a random x value, say x = 35
random_x = [[35]]  # Input must be 2D for sklearn's predict method
predicted_y = model.predict(random_x)

print(f"Predicted Claims_Total for No_Of_Claims = 35: {predicted_y[0]}")

```
![image](https://github.com/user-attachments/assets/41d40a37-f199-4e05-9b90-e3ca05576235)

Using the equation 
𝑌_predict=16.75+3.43×𝑋1
=16.75+3.43×X1
​where the intercept is approximately 16.75 and the coefficient for the number of claims is 3.43, we can predict the total claims amount. When we input 35 for 𝑋1 (No_Of_Claims), we get an estimated Claims_Total of around £136.80.

## Conclusion

The linear regression model developed to predict total claims based on the number of claims shows a strong positive relationship between the two variables. With an R² score of approximately 89.51%, the model explains a significant proportion of the variance in the total claims, indicating that the number of claims is a key predictor of the total claims amount.

The Mean Absolute Error (MAE) of 26.41 and Root Mean Squared Error (RMSE) of 29.58 indicate that, on average, our predictions deviate from the actual values by around 26.41, with no significant impact from outliers. This suggests that the model is robust and provides reasonably accurate predictions, especially for this type of insurance data.

Additionally, the model’s equation 𝑌_predict=16.75+3.43×(No_Of_Claims) allows for practical predictions, as demonstrated by an estimate of approximately £136.80 for 35 claims. Overall, this model serves as a valuable tool for predicting total claims based on the number of claims, though further refinements or the inclusion of additional variables may improve its performance.

