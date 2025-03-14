import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 2/cleaned_cardekho.csv'
data = pd.read_csv(file_path)

# Random Forest Classifier (Binary Classification)
# Calculate median selling price
median_price = data['selling_price'].median()
data['price_category'] = (data['selling_price'] >= median_price).astype(int)

# Features and target
features = ['mileage(km/ltr/kg)', 'engine', 'max_power', 'age',
            'seller_type_Individual', 'owner_Second Owner', 'transmission_Manual']
X = data[features]
y = data['price_category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5)
clf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Price', 'High Price'],
            yticklabels=['Low Price', 'High Price'])
plt.title('Confusion Matrix (Random Forest Classifier)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_rf_classifier.png')
plt.clf()

# Feature Importance (Classifier)
importance_clf = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_})
importance_clf = importance_clf.sort_values(by='Importance', ascending=False)
print("Feature Importance (Classifier):")
print(importance_clf)

# Random Forest Regressor (Regression Task)
# Target and features
X = data[features]
y = data['selling_price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
reg = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5)
reg.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = reg.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Feature Importance (Regressor)
importance_reg = pd.DataFrame({'Feature': features, 'Importance': reg.feature_importances_})
importance_reg = importance_reg.sort_values(by='Importance', ascending=False)
print("Feature Importance (Regressor):")
print(importance_reg)

# Save results
plt.figure(figsize=(10, 6))
plt.barh(importance_clf['Feature'], importance_clf['Importance'], color='skyblue')
plt.title('Feature Importance (Random Forest Classifier)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance_rf_classifier.png')
plt.clf()

plt.figure(figsize=(10, 6))
plt.barh(importance_reg['Feature'], importance_reg['Importance'], color='skyblue')
plt.title('Feature Importance (Random Forest Regressor)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance_rf_regressor.png')
plt.clf()