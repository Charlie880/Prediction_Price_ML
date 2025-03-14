import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

# Split the data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)
X_train_regr, X_test_regr, y_train_regr, y_test_regr = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Evaluation Functions
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

# Linear Regression Model Evaluation
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_regr, y_train_regr)

# Predictions
y_pred_linear = linear_regression_model.predict(X_test_regr)

# Evaluate Linear Regression
mse_linear, r2_linear, mae_linear = evaluate_regression(y_test_regr, y_pred_linear)

# Document Linear Regression Results
print("Linear Regression Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse_linear:.2f}")
print(f"R² Score: {r2_linear:.2f}")
print(f"Mean Absolute Error (MAE): {mae_linear:.2f}")

# Coefficients
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': linear_regression_model.coef_
})
print("\nLinear Regression Coefficients:")
print(coefficients)

# Hyperparameter Tuning for Decision Tree and Random Forest
# Decision Tree Classifier Grid Search
param_grid_dt_class = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
dt_classifier = DecisionTreeClassifier(random_state=42)
grid_search_dt_class = GridSearchCV(dt_classifier, param_grid_dt_class, cv=5, scoring='accuracy')
grid_search_dt_class.fit(X_train_class, y_train_class)
print("Best Parameters for Decision Tree Classifier:", grid_search_dt_class.best_params_)

# Decision Tree Regressor Grid Search
param_grid_dt_regr = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
dt_regressor = DecisionTreeRegressor(random_state=42)
grid_search_dt_regr = GridSearchCV(dt_regressor, param_grid_dt_regr, cv=5, scoring='neg_mean_squared_error')
grid_search_dt_regr.fit(X_train_regr, y_train_regr)
print("Best Parameters for Decision Tree Regressor:", grid_search_dt_regr.best_params_)

# Random Forest Classifier Grid Search
param_grid_rf_class = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
rf_classifier = RandomForestClassifier(random_state=42)
grid_search_rf_class = GridSearchCV(rf_classifier, param_grid_rf_class, cv=5, scoring='accuracy')
grid_search_rf_class.fit(X_train_class, y_train_class)
print("Best Parameters for Random Forest Classifier:", grid_search_rf_class.best_params_)

# Random Forest Regressor Grid Search
param_grid_rf_regr = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
rf_regressor = RandomForestRegressor(random_state=42)
grid_search_rf_regr = GridSearchCV(rf_regressor, param_grid_rf_regr, cv=5, scoring='neg_mean_squared_error')
grid_search_rf_regr.fit(X_train_regr, y_train_regr)
print("Best Parameters for Random Forest Regressor:", grid_search_rf_regr.best_params_)

# k-Fold Cross-Validation for Random Forest Classifier
rf_classifier_best = grid_search_rf_class.best_estimator_
cross_val_scores_rf_class = cross_val_score(rf_classifier_best, X_train_class, y_train_class, cv=5, scoring='accuracy')
print("Cross-Validation Scores for Random Forest Classifier:", cross_val_scores_rf_class)
print("Mean Accuracy:", cross_val_scores_rf_class.mean())

# k-Fold Cross-Validation for Random Forest Regressor
rf_regressor_best = grid_search_rf_regr.best_estimator_
cross_val_scores_rf_regr = cross_val_score(rf_regressor_best, X_train_regr, y_train_regr, cv=5, scoring='neg_mean_squared_error')
print("Cross-Validation Scores for Random Forest Regressor:", cross_val_scores_rf_regr)
print("Mean MSE:", -cross_val_scores_rf_regr.mean())