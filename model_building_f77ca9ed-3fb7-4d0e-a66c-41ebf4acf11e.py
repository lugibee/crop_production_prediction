import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Load the cleaned dataset
data_path = "outputs/cleaned_data.csv"
df = pd.read_csv(data_path)

# Filter out rows with missing or invalid values
df = df.dropna(subset=['Area_harvested', 'Yield', 'Production'])
df = df[(df['Area_harvested'] > 0) & (df['Yield'] > 0) & (df['Production'] > 0)]

# Features and target variable
X = df[['Area_harvested', 'Yield']]  # Features
y = df['Production']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "Support Vector Regressor (SVR)": SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

# Train and evaluate models
results = {}
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[model_name] = {
        "MSE": mse,
        "MAE": mae,
        "R²": r2
    }
    
    print(f"{model_name} Performance:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    print("-" * 30)

# Select the best model (based on R² score)
best_model_name = max(results, key=lambda x: results[x]['R²'])
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

# Save the best model
model_path = f"models/{best_model_name.replace(' ', '_').lower()}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"Best model saved to {model_path}")

# Save the results to a CSV file for comparison
results_df = pd.DataFrame(results).T  # Transpose for better readability
results_df.to_csv("outputs/model_comparison.csv", index=True)
print("Model comparison results saved to outputs/model_comparison.csv")
