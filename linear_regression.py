import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 2/cleaned_cardekho.csv'
data = pd.read_csv(file_path)

# Feature Selection
features = ['mileage(km/ltr/kg)', 'engine', 'max_power', 'age', 'seller_type_Individual', 'owner_Second Owner', 'transmission_Manual']
target = 'selling_price'

X = data[features]
y = data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Feature Importance (Coefficients)
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save feature importance as a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig('C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 2/linear_regression_visualiation/feature_importance.png')

# Plot and save Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.title("Actual vs Predicted Selling Prices")
plt.xlabel("Actual Selling Prices")
plt.ylabel("Predicted Selling Prices")
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 2/linear_regression_visualiation/actual_vs_predicted.png')

print("Plots saved successfully!")