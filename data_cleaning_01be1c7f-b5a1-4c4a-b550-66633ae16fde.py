import pandas as pd

# Load the dataset (Excel file)
data_path = "data/FAOSTAT_data.xlsx"  # Update the file path to point to your .xlsx file
df = pd.read_excel(data_path, engine='openpyxl')  # Use the openpyxl engine for .xlsx files

# Inspect the dataset
print(df.head())
print(df.info())

# Check unique values in the 'Element' column
print("Unique Elements:", df['Element'].unique())

# Filter only relevant rows (Area harvested, Yield, Production)
df = df[df['Element'].isin(['Area harvested', 'Yield', 'Production'])]

# Filter relevant columns
df = df[['Area', 'Year', 'Element', 'Item', 'Value']]

# Pivot the data to make it more usable
df_pivot = df.pivot_table(index=['Area', 'Year', 'Item'], columns='Element', values='Value').reset_index()

# Rename columns for clarity
df_pivot.columns = ['Area', 'Year', 'Item', 'Area_harvested', 'Yield', 'Production']

# Remove rows with missing values
df_pivot = df_pivot.dropna()

# Remove outliers using IQR (Interquartile Range)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply outlier removal to numerical columns
df_pivot = remove_outliers_iqr(df_pivot, 'Area_harvested')
df_pivot = remove_outliers_iqr(df_pivot, 'Yield')
df_pivot = remove_outliers_iqr(df_pivot, 'Production')

# Save the cleaned dataset
output_path = "outputs/cleaned_data.csv"
df_pivot.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")
