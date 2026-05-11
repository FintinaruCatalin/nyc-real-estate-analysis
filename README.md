# nyc-real-estate-analysis
DData science project analyzing NYC real estate prices using Python, Pandas, clustering and regression modeling.
# 🏙️ New York Real Estate Analysis & Price Prediction

## 📌 Project Overview

This project presents a full data science pipeline applied to the New York real estate market.  
The goal is to understand how location, property characteristics, and market structure influence housing prices, and to build a statistical model capable of predicting property values.

The analysis includes data cleaning, exploratory data analysis, clustering, statistical testing, and regression modeling.

---

## 🎯 Objectives

- Clean and preprocess real estate data
- Perform exploratory data analysis (EDA)
- Detect and remove outliers
- Analyze correlations between variables
- Segment New York into geographic clusters using K-Means
- Perform statistical inference (ANOVA)
- Build predictive models for property pricing
- Identify undervalued investment opportunities

---

## 📊 Dataset

The dataset contains real estate listings in New York with features such as:

- Price
- Number of bedrooms and bathrooms
- Property size (sqft)
- Latitude & Longitude
- Property type
- Location-based information

---

## 🧠 Methodology

### 1. Data Exploration & Cleaning
- Removed duplicates and irrelevant columns
- Verified missing values (none significant)
- Handled extreme outliers in price and property size

### 2. Exploratory Data Analysis
- Distribution analysis of key variables
- Detection of skewed price distribution
- Identification of extreme values

### 3. Correlation Analysis
- Pearson correlation matrix
- Strong relationship between:
  - Bedrooms and bathrooms
  - Bathrooms and price
  - Property size and price

### 4. Geographical Clustering
- Applied K-Means clustering on latitude and longitude
- Created 75 spatial clusters
- Analyzed price differences between regions

### 5. Statistical Inference (ANOVA)
- Tested whether prices differ significantly across regions
- Result: location has a statistically significant impact on price

### 6. Regression Modeling
- Built OLS regression model
- Converted categorical variables into dummy variables
- Achieved strong explanatory power (R² ≈ 0.72)

### 7. Model Optimization
- Applied log transformation to price (LOG_PRICE)
- Created engineered features (e.g. bath-to-bed ratio)
- Reduced skewness and improved model stability

---

## 📈 Key Insights

- 📍 Location is the strongest predictor of housing price
- 🏡 Property type significantly impacts price (Townhouse > House > Apartment)
- 🚿 Comfort ratio (bathrooms per bedroom) is a major value driver
- 📐 Square footage matters, but less than location
- 📊 Price distribution is highly skewed → log transformation improves accuracy

---

## 🤖 Model Performance

- R² Score: ~0.72
- High statistical significance (p < 0.05 for most variables)
- Log-linear model significantly improves robustness
- Low autocorrelation of residuals (Durbin-Watson ≈ 2)

---

## 💡 Business Applications

- Identify undervalued properties (investment opportunities)
- Compare pricing across NYC regions
- Support real estate decision-making
- Understand market segmentation patterns

---

## 🗺️ Visualizations

The project includes:
- Heatmaps (correlation analysis)
- Distribution plots
- Cluster-based geographic maps
- Interactive Plotly maps for price distribution
- Investment opportunity mapping

---

## 🛠️ Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Plotly
- SciPy
- Statsmodels
- Scikit-learn (K-Means clustering)

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
