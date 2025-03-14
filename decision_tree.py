import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 2/cleaned_cardekho.csv'
data = pd.read_csv(file_path)

# Decision Tree Classifier (Binary Classification)
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

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=10, min_samples_leaf=5)
clf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Price', 'High Price'],
            yticklabels=['Low Price', 'High Price'])
plt.title('Confusion Matrix (Decision Tree Classifier)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_classifier.png')
plt.clf()

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=features, class_names=['Low Price', 'High Price'], filled=True, rounded=True)
plt.title('Decision Tree Visualization (Classifier)')
plt.savefig('decision_tree_classifier.png')
plt.clf()

# Feature Importance (Classifier)
importance_clf = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_})
importance_clf = importance_clf.sort_values(by='Importance', ascending=False)
print("Feature Importance (Classifier):")
print(importance_clf)

# Decision Tree Regressor (Regression Task)
# Target and features
X = data[features]
y = data['selling_price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Regressor
reg = DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_split=10, min_samples_leaf=5)
reg.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = reg.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(reg, feature_names=features, filled=True, rounded=True)
plt.title('Decision Tree Visualization (Regressor)')
plt.savefig('decision_tree_regressor.png')
plt.clf()

# Feature Importance (Regressor)
importance_reg = pd.DataFrame({'Feature': features, 'Importance': reg.feature_importances_})
importance_reg = importance_reg.sort_values(by='Importance', ascending=False)
print("Feature Importance (Regressor):")
print(importance_reg)