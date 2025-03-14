import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def evaluate_regression(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

def evaluate_classification(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Load preprocessed dataset
file_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 2/cleaned_cardekho.csv'
data = pd.read_csv(file_path)

# Define features and target for regression and classification
features = ['mileage(km/ltr/kg)', 'engine', 'max_power', 'age',
            'seller_type_Individual', 'owner_Second Owner', 'transmission_Manual']

# Classification Task Evaluation
median_price = data['selling_price'].median()
data['price_category'] = (data['selling_price'] >= median_price).astype(int)

X_classification = data[features]
y_classification = data['price_category']

# Regression Task Evaluation
X_regression = data[features]
y_regression = data['selling_price']

# Split datasets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)
X_train_regr, X_test_regr, y_train_regr, y_test_regr = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Logistic Regression (for classification)
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_class, y_train_class)
y_pred_class_log = log_reg.predict(X_test_class)

# Linear Regression (for regression)
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_regr, y_train_regr)
y_pred_regr_linear = linear_regression_model.predict(X_test_regr)

# Decision Tree
# Classification
dt_class = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_class.fit(X_train_class, y_train_class)
y_pred_class_dt = dt_class.predict(X_test_class)

# Regression
dt_regr = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_regr.fit(X_train_regr, y_train_regr)
y_pred_regr_dt = dt_regr.predict(X_test_regr)

# Random Forest
# Classification
rf_class = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
rf_class.fit(X_train_class, y_train_class)
y_pred_class_rf = rf_class.predict(X_test_class)

# Regression
rf_regr = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42)
rf_regr.fit(X_train_regr, y_train_regr)
y_pred_regr_rf = rf_regr.predict(X_test_regr)

# Evaluation Functions
def print_classification_metrics(metrics):
    print("Classification Metrics:")
    for metric in metrics:
        print(f"{metric[0]}: Accuracy={metric[1]:.2f}, Precision={metric[2]:.2f}, Recall={metric[3]:.2f}, F1 Score={metric[4]:.2f}")

def print_regression_metrics(metrics):
    print("Regression Metrics:")
    for metric in metrics:
        print(f"{metric[0]}: MSE={metric[1]:.2f}, R²={metric[2]:.2f}, MAE={metric[3]:.2f}")

# Classification metrics
classification_metrics = [
    ("Logistic Regression", *evaluate_classification(y_test_class, y_pred_class_log)),
    ("Decision Tree", *evaluate_classification(y_test_class, y_pred_class_dt)),
    ("Random Forest", *evaluate_classification(y_test_class, y_pred_class_rf)),
]

# Regression metrics
regression_metrics = [
    ("Linear Regression", *evaluate_regression(y_test_regr, y_pred_regr_linear)),
    ("Decision Tree", *evaluate_regression(y_test_regr, y_pred_regr_dt)),
    ("Random Forest", *evaluate_regression(y_test_regr, y_pred_regr_rf)),
]

# Visualization
# Regression metrics bar chart
regression_results = pd.DataFrame(regression_metrics, columns=["Model", "MSE", "R2", "MAE"])
regression_results.set_index("Model").plot(kind='bar', figsize=(10, 8), title="Regression Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('regression_metrics_comparison.png')
plt.close()

# Classification metrics bar chart
classification_results = pd.DataFrame(classification_metrics, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
classification_results.set_index("Model").plot(kind='bar', figsize=(10, 8), title="Classification Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('classification_metrics_comparison.png')
plt.close()

# Residual plot for Linear Regression
residuals = y_test_regr - y_pred_regr_linear
plt.scatter(y_pred_regr_linear, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Linear Regression)")
plt.tight_layout()
plt.savefig('residual_plot_linear_regr.png')
plt.close()

# Feature importance for Random Forest
feature_importance_rf = pd.DataFrame({
    "Feature": features,
    "Importance": rf_regr.feature_importances_
}).sort_values(by="Importance", ascending=False)
feature_importance_rf.plot(kind='bar', x='Feature', y='Importance', legend=False, title="Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig('feature_importance_rf_regr.png')
plt.close()

# Decision Tree visualization
plt.figure(figsize=(15, 10))
plot_tree(dt_regr, feature_names=features, filled=True, rounded=True, max_depth=2)
plt.title("Decision Tree Regressor Visualization (Max Depth=3)")
plt.tight_layout()
plt.savefig('decision_tree_regr.png')
plt.close()

# Confusion Matrix for Random Forest Classifier
ConfusionMatrixDisplay.from_estimator(rf_class, X_test_class, y_test_class)
plt.title("Confusion Matrix (Random Forest)")
plt.tight_layout()
plt.savefig('confusion_matrix_rf_class.png')
plt.close()

# Print evaluation results
print_classification_metrics(classification_metrics)
print()
print_regression_metrics(regression_metrics)