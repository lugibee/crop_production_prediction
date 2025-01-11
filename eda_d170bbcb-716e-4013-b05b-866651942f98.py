import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
data_path = "outputs/cleaned_data.csv"
df = pd.read_csv(data_path)

# Inspect the dataset
print("Dataset Overview:")
print(df.head())
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 1. Crop Distribution
print("\nCrop Distribution:")
crop_distribution = df['Item'].value_counts()
print(crop_distribution)

# Plot crop distribution (Top 20 crops)
plt.figure(figsize=(12, 6))
top_crops = crop_distribution.head(20)  # Show only the top 20 crops
sns.barplot(x=top_crops.index, y=top_crops.values, palette="viridis")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title("Top 20 Crops by Count")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/eda_plots/crop_distribution.png")
plt.show()

# 2. Geographical Distribution
print("\nTop 10 Regions by Total Production:")
top_regions = df.groupby('Area')['Production'].sum().sort_values(ascending=False).head(10)
print(top_regions)

# Plot top regions
plt.figure(figsize=(12, 6))
sns.barplot(x=top_regions.index, y=top_regions.values, palette="mako")
plt.title("Top 10 Regions by Total Production")
plt.xlabel("Region")
plt.ylabel("Total Production (tons)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/eda_plots/top_regions.png")
plt.show()

# 3. Temporal Analysis
# Yearly Trends
print("\nYearly Trends in Production:")
yearly_trends = df.groupby('Year')[['Area_harvested', 'Yield', 'Production']].sum()
print(yearly_trends)

# Plot yearly trends
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_trends, markers=True)
plt.title("Yearly Trends in Area Harvested, Yield, and Production")
plt.xlabel("Year")
plt.ylabel("Values")
plt.grid()
plt.tight_layout()
plt.savefig("outputs/eda_plots/yearly_trends.png")
plt.show()

# Growth Analysis (Trends for specific crops)
print("\nGrowth Analysis for Top 5 Crops:")
top_5_crops = df['Item'].value_counts().head(5).index
for crop in top_5_crops:
    crop_data = df[df['Item'] == crop].groupby('Year')['Production'].sum()
    plt.plot(crop_data.index, crop_data.values, marker='o', label=crop)

plt.title("Production Trends for Top 5 Crops")
plt.xlabel("Year")
plt.ylabel("Production (tons)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("outputs/eda_plots/top_crops_growth.png")
plt.show()

# 4. Input-Output Relationships
print("\nCorrelation Matrix:")
correlation = df[['Area_harvested', 'Yield', 'Production']].corr()
print(correlation)

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Area Harvested, Yield, and Production")
plt.tight_layout()
plt.savefig("outputs/eda_plots/correlation_heatmap.png")
plt.show()

# 5. Productivity Analysis
df['Productivity'] = df['Production'] / df['Area_harvested']
print("\nTop 10 Crops by Productivity:")
top_productivity = df.groupby('Item')['Productivity'].mean().sort_values(ascending=False).head(10)
print(top_productivity)


# 6. Outliers in Production (Fixing Overlapping Words)
# Limit to top 20 crops by total production
top_20_crops = df.groupby('Item')['Production'].sum().sort_values(ascending=False).head(20).index
filtered_df = df[df['Item'].isin(top_20_crops)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_df, x='Item', y='Production', palette="Set3")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.title("Outliers in Production for Top 20 Crops")
plt.xlabel("Crop")
plt.ylabel("Production (tons)")
plt.tight_layout()
plt.savefig("outputs/eda_plots/production_outliers_fixed.png")
plt.show()

# 5. Top Crops by Productivity (Fixing Empty Plot)
# Filter out rows where Area_harvested is 0 or NaN
df = df[df['Area_harvested'] > 0]

# Calculate productivity
df['Productivity'] = df['Production'] / df['Area_harvested']

# Get top 10 crops by average productivity
top_productivity = df.groupby('Item')['Productivity'].mean().sort_values(ascending=False).head(10)
print("\nTop 10 Crops by Productivity:")
print(top_productivity)

# Plot top crops by productivity
plt.figure(figsize=(12, 6))
sns.barplot(x=top_productivity.index, y=top_productivity.values, palette="coolwarm")
plt.title("Top 10 Crops by Productivity")
plt.xlabel("Crop")
plt.ylabel("Productivity (tons/ha)")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.savefig("outputs/eda_plots/top_crops_productivity_fixed.png")
plt.show()

# Save EDA insights to a text file
eda_insights_path = "outputs/eda_insights.txt"
with open(eda_insights_path, "w") as f:
    f.write("Crop Distribution:\n")
    f.write(str(crop_distribution))
    f.write("\n\nYearly Trends:\n")
    f.write(str(yearly_trends))
    f.write("\n\nCorrelation Matrix:\n")
    f.write(str(correlation))
    f.write("\n\nTop 10 Regions by Total Production:\n")
    f.write(str(top_regions))
    f.write("\n\nTop 10 Crops by Productivity:\n")
    f.write(str(top_productivity))
print(f"EDA insights saved to {eda_insights_path}")
