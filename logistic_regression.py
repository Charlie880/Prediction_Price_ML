import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 2/cleaned_cardekho.csv'
data = pd.read_csv(file_path)

# Calculate median selling price
median_price = data['selling_price'].median()

# Create binary target column
# 1: High Price, 0: Low Price
data['price_category'] = (data['selling_price'] >= median_price).astype(int)

# Features and target
features = ['mileage(km/ltr/kg)', 'engine', 'max_power', 'age',
            'seller_type_Individual', 'owner_Second Owner', 'transmission_Manual']
X = data[features]
y = data['price_category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Price', 'High Price'],
            yticklabels=['Low Price', 'High Price'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')

# ROC-AUC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('roc_curve.png')

# Coefficients analysis
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print("Feature Importance:")
print(coefficients)