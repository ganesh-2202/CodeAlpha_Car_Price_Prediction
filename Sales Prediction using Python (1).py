# STEP 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# STEP 2: Load the dataset
df = pd.read_csv(r"C:\Users\vishn\Downloads\Advertising.csv")

# STEP 3: View dataset
print(df.head())
# STEP 4: Dataset information
print(df.info())
# STEP 5: Statistical summary
print(df.describe())
# STEP 6: Check for missing values
print(df.isnull().sum())
# STEP 7: Data visualization
sns.pairplot(df)
plt.show()
# STEP 8: Correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
# STEP 9: Feature selection
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
# STEP 10: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# STEP 11: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# STEP 12: Make predictions
y_pred = model.predict(X_test)
# STEP 13: Model evaluation
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
# STEP 14: Model coefficients (Advertising impact)
coeff_df = pd.DataFrame({
    "Advertising Medium": X.columns,
    "Coefficient": model.coef_
})
print(coeff_df)
# STEP 15: Actual vs Predicted comparison
comparison = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": y_pred
})
print(comparison.head())
