# %% [markdown]
# # Analysis of Real Estate Prices in New York
# In this project we present a complete data analysis pipeline including cleaning, geographic segmentation,
# and statistical modeling for the New York real estate market.
#
# ### Objectives:
# 1. **Data Cleaning**: Identifying and removing noise (outliers) and redundant data.
# 2. **Exploratory Analysis**: Understanding the distribution of key variables.
# 3. **Geographic Segmentation**: Grouping properties by area using K-Means and identifying the most/least expensive zones.
# 4. **Predictive Modeling**: Building a regression model to estimate property prices.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.vq import kmeans2, whiten
import statsmodels.api as sm
import plotly.express as px
from IPython.display import display

# Set number format to avoid scientific notation
pd.options.display.float_format = '{:.2f}'.format
plt.style.use('ggplot')  # Cleaner visual style

# %% [markdown]
# ## 1. Loading and Initial Data Inspection
# We start by loading the dataset and examining the available information.

# %%
filename = "NY-House-Dataset.csv"
df = pd.read_csv(filename)

print(f"Total number of records: {df.shape[0]}")
print(f"Total number of variables: {df.shape[1]}")
display(df.describe())

# %% [markdown]
# ## 2. Data Cleaning
# In this step we remove columns that cannot be used in mathematical calculations (raw addresses),
# handle duplicates, and remove outliers.
# *Note: No null values were found in this dataset — they appear to have been pre-cleaned using mean imputation.
# We also see values like 2.37 for bathrooms, which happens to be exactly the mean, supporting this conclusion.*
# In the previous step we also observed max(price) = 2147483647, which is the upper limit of a 32-bit integer —
# clearly an error. These will be handled in the next step.

# %%
# Check for null values
print("Null values per column:\n", df.isnull().sum())

# Drop columns irrelevant to the analysis
df = df.drop(columns=['BROKERTITLE', 'MAIN_ADDRESS', 'FORMATTED_ADDRESS'])

# Remove duplicates (identical records that can distort statistics)
print(f"Duplicates found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Duplicates after cleaning: {df.duplicated().sum()}")


# %% [markdown]
# ## 3. Distribution Analysis and Outlier Removal
# Datasets often contain extreme values — either data entry errors or exceptional cases.
# We visualize the distribution to observe data skewness.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['BEDS'], bins=20, color='skyblue', kde=True, ax=axes[0])
axes[0].set_title('Bedroom Distribution')

sns.histplot(df['BATH'], bins=20, color='salmon', kde=True, ax=axes[1])
axes[1].set_title('Bathroom Distribution')

sns.histplot(df['PROPERTYSQFT'], bins=20, color='green', kde=True, ax=axes[2])
axes[2].set_title('Property Size Distribution (sqft)')

plt.tight_layout()
plt.show()

# Outlier filtering: keep data representative of the general market
# We remove properties exceeding extreme thresholds to avoid misleading the regression model.
df_clean = df[
    (df['PRICE'] < 500_000_000) &
    (df['PRICE'] > 20_000) &
    (df['BEDS'] < 13) &
    (df['BATH'] < 11) &
    (df['PROPERTYSQFT'] < 10001)
].copy()

print(f"Records remaining after cleaning: {len(df_clean)}")
display(df_clean.describe())
# Both the mean and standard deviation dropped substantially — the figures were artificially inflated by what appeared to be data entry errors.

# %% [markdown]
# ## 4. Correlation Analysis
# We check how strongly variables are related using Pearson correlation.
# A value close to 1.0 indicates a strong direct relationship.

# %%
numeric_cols = ['PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT']
corr_matrix = df_clean[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Numeric Variables)")
plt.show()

# %% [markdown]
# Interpretation: There is a strong correlation between BEDS and BATH (~0.7), which can lead to multicollinearity.
# We also observe that BATH correlates more strongly with PRICE than BEDS does.
# This is explained by the importance of comfort level: a 5-bedroom property is less valuable without a matching number of bathrooms.
# The positive correlation between PROPERTYSQFT and PRICE is expected.

# %% [markdown]
# ## 5. Geographic Clustering
# Location is the most important factor in real estate. Since neighborhood data is not normalized,
# we use **K-Means** on coordinates (Lat/Long) to create 75 "micro-zones".
# This avoids dealing with LOCALITY / SUBLOCALITY variables while still enabling map visualization.
# After clustering, a geocoding service like Google Places API could be used to label the zones,
# but that is outside the scope of this project. We do record each cluster's centroid for map identification.

# %%
# Whiten coordinates to prepare them for K-Means clustering
cluster_features = ['LATITUDE', 'LONGITUDE']
data_whitened = whiten(df_clean[cluster_features].values)

# Run K-Means to identify geographic zones
centroids, labels = kmeans2(data_whitened, k=75, minit='points', iter=20)
df_clean['Cluster'] = labels

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_clean, x='LONGITUDE', y='LATITUDE', hue='Cluster', palette='viridis', alpha=0.4, legend=False)
plt.title('Automatic Geographic Clustering (75 Clusters)')
plt.show()

cluster_summary = df_clean.groupby('Cluster').agg({
    'LATITUDE': 'mean',
    'LONGITUDE': 'mean',
    'PRICE': 'mean'
}).reset_index()
cluster_summary.columns = ['Cluster', 'Center_Lat', 'Center_Lng', 'Avg_Cluster_Price']

cluster_summary = cluster_summary.sort_values(by='Avg_Cluster_Price', ascending=False)

print("Summary of 75 clusters (sorted by average price):")
display(cluster_summary)

# %% [markdown]
# ## 6. Statistical Inference (ANOVA)
# We use the ANOVA test to confirm whether prices differ significantly across localities.
# If p-value < 0.05, location has a statistically significant impact on price.

# %%
print("--- ANOVA Analysis for Clusters ---")
localities_list = [df_clean[df_clean['Cluster'] == loc]['PRICE'] for loc in df_clean['Cluster'].unique() if len(df_clean[df_clean['Cluster'] == loc]) > 10]

f_stat, p_val = stats.f_oneway(*localities_list)
print(f"F-Statistic: {f_stat:.2f}, P-value: {p_val:.4e}")
print("Interpretation: There is a statistically significant difference in prices across neighborhoods. Location matters!")
print("The standard threshold to confirm the hypothesis is p < 0.05. In this case, p is astronomically small.")

# %% [markdown]
# ## 7. Interactive Market Visualization
# We color zones by their average cluster price. Yellow/light green = expensive zones, purple = affordable zones.

# %%
cluster_prices = df_clean.groupby('Cluster')['PRICE'].mean().reset_index()
cluster_prices.columns = ['Cluster', 'Cluster_Avg_Price']
df_plot = df_clean.merge(cluster_prices, on='Cluster')

fig_clusters = px.scatter_map(
    df_plot,
    lat='LATITUDE', lon='LONGITUDE',
    color='Cluster_Avg_Price',
    zoom=10, height=800,
    title='Average Price Map by Zone (Cluster)',
    color_continuous_scale=px.colors.sequential.Viridis,
    map_style="open-street-map"
)
fig_clusters.update_layout(
    margin={"r": 0, "t": 50, "l": 0, "b": 0}
)
fig_clusters.show()

# %% [markdown]
# ## 8. Predictive Modeling (OLS Regression)
# We attempt to predict price using property features and location (clusters).
# Categorical variables are converted into dummy variables (0 or 1).

# %%
# Convert categorical variables to dummies
df_model = pd.get_dummies(df_clean, columns=['TYPE', 'Cluster'], drop_first=True)

X_cols = ['BEDS', 'BATH', 'PROPERTYSQFT'] + \
    [col for col in df_model.columns if 'TYPE_' in col] + \
    [col for col in df_model.columns if 'Cluster_' in col]

X = df_model[X_cols].astype(float)
y = df_model['PRICE'].astype(float)

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()

print(model.summary())

# %% [markdown]
# ## 9. Model Optimization: Log-Linear Model
# Real estate prices often grow exponentially, not linearly.
# 1. **Log(Price)**: NYC prices are highly skewed. Log-transforming the price reduces the effect of outliers
#    that cause anomalies like the negative BEDS coefficient seen in the first regression version.
#    After log transformation, coefficients become relative rather than absolute.
# 2. **BATH_PER_BED**: A new variable describing the comfort ratio, which removes multicollinearity
#    caused by the correlated BEDS and BATH variables. It also significantly increases R² and
#    gives the BEDS coefficient a correct, meaningful value.

# %%
# Create new engineered features
df_model['LOG_PRICE'] = np.log1p(df_model['PRICE'])
df_model['BATH_PER_BED'] = df_model['BATH'] / df_model['BEDS']
df_model['BATH_PER_BED'] = df_model['BATH_PER_BED'].replace([np.inf, -np.inf], 0).fillna(0)

# Select new variables, removing BATH to reduce multicollinearity
X_new_cols = ['BEDS', 'BATH_PER_BED', 'PROPERTYSQFT'] + \
    [col for col in df_model.columns if 'TYPE_' in col] + \
    [col for col in df_model.columns if 'Cluster_' in col]

X_new = df_model[X_new_cols].astype(float)
y_log = df_model['LOG_PRICE']

X_new_const = sm.add_constant(X_new)
model_optimized = sm.OLS(y_log, X_new_const).fit()

print(model_optimized.summary())

# %% [markdown]
# ## 10. Interpreting the Optimized Regression Model (LOG_PRICE)
# We evaluate model quality through key OLS indicators, focusing on predictive power and error validity.
#
# ### 10.1. Predictive Power and Global Significance
# * **R-squared (0.726):** The model explains ~72% of the variance in log-transformed prices.
#   In a volatile market like NYC, a score above 0.7 indicates strong predictive capacity and a well-calibrated model.
# * **Adj. R-squared (0.720):** Very close to R², confirming that the added variables (75 cluster dummies and property types)
#   are relevant and do not artificially inflate the score.
# * **F-statistic (131.5) & Prob(F) = 0.00:** Confirms global significance. The chosen independent variables
#   have a real impact on sale price.
#
# ### 10.2. Error Diagnostics (Validity Tests)
# * **Durbin-Watson (1.977):** Close to the ideal value of 2.0, indicating no autocorrelation in residuals.
#   Residuals are independent — an essential condition for OLS validity.
# * **Skew (0.046) & Kurtosis (6.651):** Near-zero skewness confirms the log transformation successfully
#   normalized the price distribution.
#   * The kurtosis, while reduced, remains above the ideal of 3, suggesting some residual outliers
#     (unique NYC properties) still produce larger-than-average prediction errors — but far fewer than before.
# * **Jarque-Bera (2510.015) & Omnibus (388.735), prob = 0:** Residuals do not follow a perfectly normal distribution.
#   In econometrics, this is common with thousands of observations and does not negate the model's practical usefulness —
#   it simply reflects real-world data complexity.
#
# ### 10.3. Multicollinearity and Data Matrix
# * **Cond. No. (1.89e+05):** A high condition number is expected when using 75 cluster dummy variables.
#   This numerical complexity is unavoidable in models with such fine geographic segmentation,
#   but is offset by the coefficient stability gained by removing the BEDS–BATH direct correlation.
#
# **Conclusion:** The log-price optimized model is significantly more robust than the initial version,
# providing a faithful picture of how size, comfort, and location interact to determine real estate value in NYC.

# %% [markdown]
# ## 11. Identifying Undervalued Properties (Investment Opportunities)
# To find undervalued properties, we compare the actual listed price with the model's predicted price.
# * **Large negative residual**: The actual price is well below what the model considers "fair" for those features and location.
# * **Note**: A property may appear undervalued due to hidden defects (e.g., structural issues) not captured in the data.

# %%
# 1. Get model predictions in log format and convert back to dollars
# We use expm1 to reverse the log1p transformation used during training
df_clean['PREDICTED_PRICE'] = np.expm1(model_optimized.predict(X_new_const))

# 2. Calculate absolute and percentage difference
# A negative number means the actual price is lower than the model's estimate
df_clean['PRICE_DIFF'] = df_clean['PRICE'] - df_clean['PREDICTED_PRICE']
df_clean['UNDERVALUATION_PCT'] = (df_clean['PRICE_DIFF'] / df_clean['PREDICTED_PRICE']) * 100

# 3. Filter properties
# We look for properties where the actual price is at least 40% below the model estimate
deals = df_clean[df_clean['UNDERVALUATION_PCT'] < -40].sort_values(by='UNDERVALUATION_PCT')

print(f"Found {len(deals)} potentially undervalued properties.")
display(deals[['TYPE', 'PRICE', 'PREDICTED_PRICE', 'UNDERVALUATION_PCT', 'BEDS', 'BATH']].head(10))

# 4. Visualize investment opportunities on an interactive map
fig_deals = px.scatter_map(
    deals,
    lat='LATITUDE', lon='LONGITUDE',
    color='UNDERVALUATION_PCT',
    size='PROPERTYSQFT',
    hover_data=['PRICE', 'PREDICTED_PRICE', 'TYPE'],
    title='Top Investment Opportunities: Undervalued Properties (Actual vs. Model Price)',
    color_continuous_scale='RdYlGn_r',  # Green = undervalued
    zoom=10, height=800,
    map_style="open-street-map"
)
fig_deals.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
fig_deals.show()
deals.to_csv("undervalued_properties.csv", index=False)

# %% [markdown]
# ### Interpretation
# If the model predicts $1,000,000 (PREDICTED_PRICE) and the property is listed at $600,000 (PRICE),
# there is a potential profit of $400,000 achievable by reselling at market value.
# Intense green dots represent areas where the asking price is well below the statistical value
# for that zone and its features.

# %% [markdown]
# ## 12. Specific Analysis: Geographic Extremes
# **Question: What are the "poles" of the NYC real estate market based on identified clusters?**
#
# ### Answer:
# We identify the clusters with the highest and lowest average prices to understand the zone hierarchy.

# %%
# Calculate cluster averages and sort
top_clusters = df_clean.groupby('Cluster')['PRICE'].mean().sort_values(ascending=False)

print("Top 5 Most Expensive Clusters (Luxury Zones):")
display(top_clusters.head(5))

print("\nTop 5 Least Expensive Clusters (Affordable Zones):")
display(top_clusters.tail(5))

# %% [markdown]
# **Interpretation:** The price difference between the most and least expensive cluster can exceed 14x.
# This confirms that in NYC, location has a stronger impact than property size.

# %% [markdown]
# ## 13. Specific Analysis: Cost Efficiency per Square Foot
# **Question: Which property type offers the best Price / Square Foot ratio?**
#
# ### Answer:
# We calculate a new variable `PRICE_PER_SQFT` to see how much a buyer pays per unit of area
# depending on property type.

# %%
df_typeclean = df_clean[df_clean["TYPE"] != "For sale"].copy()  # Remove generic "For sale" type — not meaningful here
df_typeclean['PRICE_PER_SQFT'] = df_typeclean['PRICE'] / df_typeclean['PROPERTYSQFT']
sqft_analysis = df_typeclean.groupby('TYPE')['PRICE_PER_SQFT'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sqft_analysis.plot(kind='bar', color='teal')
plt.title('Average Price per Square Foot by Property Type')
plt.ylabel('USD / Sqft')
plt.show()

# %% [markdown]
# **Interpretation:** Luxury types (e.g. Townhouses) have a much higher price per sqft.
# An investor seeking "volume" will target property types with a lower price per sqft,
# getting more space for the same amount of money.

# %% [markdown]
# ## 14. Specific Analysis: Price per Square Foot by Cluster
# **Question: Which zones are the most expensive relative to the space offered?**

# %%
# 1. Calculate price per square foot for each property
df_clean['PRICE_PER_SQFT'] = df_clean['PRICE'] / df_clean['PROPERTYSQFT']

# 2. Group by cluster and calculate average price per sqft
cluster_value = df_clean.groupby('Cluster').agg({
    'PRICE_PER_SQFT': 'mean',
    'LATITUDE': 'mean',
    'LONGITUDE': 'mean'
}).reset_index()

cluster_value.columns = ['Cluster', 'Avg_Price_Per_Sqft', 'Center_Lat', 'Center_Lng']
cluster_value = cluster_value.sort_values(by='Avg_Price_Per_Sqft', ascending=False)
cluster_value.to_csv("price_per_sqft_clusters.csv", index=False)

# 3. Visualize top 15 most expensive clusters (where space is most valuable)
top_expensive_sqft = cluster_value.sort_values(by='Avg_Price_Per_Sqft', ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_expensive_sqft,
    x='Cluster',
    y='Avg_Price_Per_Sqft',
    palette='magma',
    order=top_expensive_sqft['Cluster'],
    hue='Avg_Price_Per_Sqft',
)
plt.title('Top 15 Clusters: Most Expensive Zones per Unit Area (USD/Sqft)')
plt.xlabel('Cluster')
plt.ylabel('Average Price USD / Sqft')
plt.show()

# 4. Interactive map for value distribution per sqft
df_plot_val = df_clean.merge(cluster_value, on='Cluster')

fig_val = px.scatter_map(
    df_plot_val,
    lat='LATITUDE', lon='LONGITUDE',
    color='Avg_Price_Per_Sqft',
    size='PRICE_PER_SQFT',
    hover_data=['Cluster', 'Avg_Price_Per_Sqft', 'PRICE'],
    title='Real Estate Value Map: Price per Square Foot by Cluster',
    color_continuous_scale='Inferno',
    zoom=10, height=800,
    map_style="open-street-map"
)
fig_val.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
fig_val.show()

# %% [markdown]
# ### Interpretation:
# * **Differences between averages:** Huge discrepancies exist between clusters. While peripheral zones have low price/sqft,
#   luxury areas like Manhattan Island around Central Park command enormous premiums purely for location.
# * **Investment decision:** Clusters with a high total price but moderate price/sqft may represent safer investments
#   than zones where you pay a lot for very few square meters.
# * **Regression confirmation:** This analysis explains why SQFT has a stable coefficient in the regression,
#   while Cluster variables dominate price variation.
