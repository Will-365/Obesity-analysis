# Install libraries (only run once in terminal if not already installed):
# pip install pandas matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
# # # --------------------------
# # # 1. DATA CLEANING
# # # --------------------------

# # # Load raw dataset
# df = pd.read_csv("obesity-prevalence-adults-who-gho.csv")

# # Rename last column to shorter name
# df.rename(columns={df.columns[-1]: "Obesity_Prevalence"}, inplace=True)

# # Drop 'Code' column (not needed for analysis)
# if 'Code' in df.columns:
#     df.drop(columns=['Code'], inplace=True)

# # Drop rows with missing prevalence
# df = df.dropna(subset=['Obesity_Prevalence'])

# # Convert Year to integer
# df['Year'] = df['Year'].astype(int)

# # Display info and basic statistics
# print("\nDataset Info After Cleaning:")
# print(df.info())
# print("\nDescriptive Statistics:")
# print(df.describe())

# # Save cleaned dataset
# df.to_csv("cleaned_obesity_data.csv", index=False)
# print("\nCleaned dataset saved as 'cleaned_obesity_data.csv'")

# # --------------------------
# # 2. EXPLORATORY DATA ANALYSIS
# # --------------------------

# #Load cleaned dataset
# df = pd.read_csv("cleaned_obesity_data.csv")

# # Global trend
# global_trend = df.groupby('Year')['Obesity_Prevalence'].mean()

# plt.figure(figsize=(10,5))
# plt.plot(global_trend.index, global_trend.values, marker='o', color='blue')
# plt.title("Global Average Obesity Trend (1990-2022)")
# plt.xlabel("Year")
# plt.ylabel("Average Obesity Prevalence (%)")
# plt.grid(True)
# plt.show()

# # Top 10 countries (latest year)
# latest_year = df['Year'].max()
# top10 = df[df['Year'] == latest_year].sort_values('Obesity_Prevalence', ascending=False).head(10)

# plt.figure(figsize=(10,5))
# sns.barplot(x='Obesity_Prevalence', y='Entity', data=top10, palette='Reds_r')
# plt.title(f"Top 10 Countries by Obesity Prevalence in {latest_year}")
# plt.xlabel("Obesity Prevalence (%)")
# plt.ylabel("Country")
# plt.show()
# # --------------------------
# # 3. MACHINE LEARNING (K-MEANS)
# # --------------------------

# # Load cleaned dataset
# df = pd.read_csv("cleaned_obesity_data.csv")

# # 1. Prepare average obesity prevalence per country
# country_avg = df.groupby('Entity')['Obesity_Prevalence'].mean().reset_index()

# # 2. Scale the data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(country_avg[['Obesity_Prevalence']])

# # 3. Apply KMeans clustering
# kmeans = KMeans(n_clusters=3, random_state=42)
# country_avg['Cluster'] = kmeans.fit_predict(scaled_data)

# # 4. Evaluate clustering
# score = silhouette_score(scaled_data, country_avg['Cluster'])
# print("\nSilhouette Score (measure of cluster quality):", score)

# # 5. Show clustered data sample
# print("\nClustered Data (first 10 rows):")
# print(country_avg.head(10))

# # 6. Visualize clusters
# plt.figure(figsize=(8,5))
# sns.scatterplot(
#     x=country_avg['Obesity_Prevalence'],
#     y=[0]*len(country_avg),
#     hue=country_avg['Cluster'],
#     palette='Set2'
# )
# plt.title("Clusters of Countries by Average Obesity Prevalence")
# plt.xlabel("Average Obesity Prevalence (%)")
# plt.yticks([])
# plt.show()

# # 7. Save clustered data for Power BI
# country_avg.to_csv("obesity_clusters.csv", index=False)
# print("\nClustered data saved as 'obesity_clusters.csv'")
# # --------------------------
# # 4. INNOVATION: FORECASTING
# # --------------------------
# # Load cleaned dataset
df = pd.read_csv("cleaned_obesity_data.csv")

# Prepare global average obesity per year
global_trend = df.groupby('Year')['Obesity_Prevalence'].mean().reset_index()

# Features (years) and target (obesity prevalence)
X = global_trend['Year'].values.reshape(-1,1)
y = global_trend['Obesity_Prevalence'].values

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict next 5 years
future_years = np.array([2023, 2024, 2025, 2026, 2027]).reshape(-1,1)
future_preds = model.predict(future_years)

# Combine past and future predictions
plt.figure(figsize=(10,5))
plt.plot(global_trend['Year'], y, label='Historical', color='blue', marker='o')
plt.plot(future_years.flatten(), future_preds, label='Forecast', color='orange', linestyle='--', marker='x')
plt.title("Forecast of Global Obesity Trend (2023-2027)")
plt.xlabel("Year")
plt.ylabel("Average Obesity Prevalence (%)")
plt.legend()
plt.grid(True)
plt.show()

# Show forecasted values
forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted_Obesity_Prevalence': future_preds})
print("\nForecasted Obesity Prevalence (2023-2027):")
print(forecast_df)

