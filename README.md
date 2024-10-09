
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
INSERT PICTURE OF TABLE HEAD


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
INSERT SCATTER PLOT!!! \
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
INSERT BOX PLOT!!! \
\ We create boxplots for both variables to observe their distribution and detect any outliers.

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
INSERT Histogram PLOT!!! n\
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
SHOW R2 VALUE AND DISCRIBE WHAT IT MEANS

### Calculating MAE and RMSE
We also compute the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to further evaluate the model's accuracy.
```python
mae = mean_absolute_error(Y_test, y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
```
SHOW MAE MSE VALUE AND DISCRIBE WHAT IT MEANS

### Visualizing the Regression Line
We plot the original data points alongside the regression line, which is derived from the model’s predictions.

```python
y_pred_full = model.predict(df[['No_Of_Claims']])
plt.figure(figsize=(5, 4))
sns.scatterplot(x='No_Of_Claims', y='Claims_Total', data=df)
plt.plot(df['No_Of_Claims'], y_pred_full, color='red')
plt.legend(['Regression Line', 'Original Data'])
plt.title('Original Data vs Predicted Line')
plt.show()

```

### Predicting Claims for a Specific Number of Claims


```python
random_x_value = float(input("Enter a value for No_Of_Claims: "))
random_x = [[35]]  # Example input
predicted_y = model.predict(random_x)
print(f"Predicted Claims_Total for No_Of_Claims = 35: {predicted_y[0]}")

```




## Conclusion

