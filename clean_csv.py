import pandas as pd

# Load the dataset
file_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 1/cardekho.csv'
car_data = pd.read_csv(file_path)

# Check for missing values
print("Missing Values:")
print(car_data.isnull().sum())

# Analyze the target variable 'selling_price'
print("\nSelling Price Analysis:")
print(car_data['selling_price'].describe())

# Inspect 'max_power' column
print("\nUnique values in 'max_power' column (Top 10):")
print(car_data['max_power'].value_counts().head(10))

# Remove leading and trailing whitespaces in string columns
car_data = car_data.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

# Retain all rows, no duplicate removal
print(f"\nNumber of rows retained (no duplicates removed): {len(car_data)}")

# Clean and convert 'max_power' to numeric
# Extract numerical part of 'max_power' and leave existing values unchanged
car_data['max_power'] = car_data['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)

# Check if 'year' column exists
if 'year' in car_data.columns:
    # Calculate the age of the car
    current_year = 2025  # Replace with the current year if needed
    car_data['age'] = current_year - car_data['year']
    
    # Save the updated dataset
    updated_file_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 2/cleaned_cardekho_with_age.csv'
    car_data.to_csv(updated_file_path, index=False)
    print(f"Age column added successfully and dataset saved to {updated_file_path}")
else:
    print("The 'year' column does not exist in the dataset. Please verify the input file.")

# Fill missing values
car_data['mileage(km/ltr/kg)'] = car_data['mileage(km/ltr/kg)'].fillna(car_data['mileage(km/ltr/kg)'].mean())
car_data['engine'] = car_data['engine'].fillna(car_data['engine'].mean())
car_data['seats'] = car_data['seats'].fillna(car_data['seats'].mode()[0])
car_data['max_power'] = car_data['max_power'].fillna(car_data['max_power'].mean())

# Verify the changes
print("\nMissing Values After Cleaning:")
print(car_data.isnull().sum())

# Select categorical features
categorical_features = ['seller_type', 'owner', 'transmission']

# Analyze unique values in these columns
print("\nUnique values in selected categorical features:")
for col in categorical_features:
    print(f"{col}: {car_data[col].unique()}")

# One-hot encoding for categorical features
car_data_encoded = pd.get_dummies(car_data, columns=categorical_features, drop_first=True)

# Verify the shape of the dataset after encoding
print("\nEncoded Dataset Shape:", car_data_encoded.shape)

# Save the cleaned dataset to a new CSV file
output_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 2/cleaned_cardekho.csv'
car_data_encoded.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")