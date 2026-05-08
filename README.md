
# 🌍 WorldLens — Global Happiness EDA Dashboard

> End-to-end Data Science project — EDA · Machine Learning · Interactive Dashboard

<img width="1309" height="621" alt="image" src="https://github.com/user-attachments/assets/106e2828-4dc1-4b05-812e-5e2d98f309ee" />
---

## 📌 What is WorldLens?

WorldLens is a complete, portfolio-ready Data Science project built on the
World Happiness Report (2015–2022). It covers the full pipeline:
raw data → cleaning → EDA → ML modelling → interactive web dashboard.

---

## 🗂️ Project Structure

WorldLens_EDA_Dashboard/
├── data/                  ← Raw CSV files (2015–2022)
├── notebooks/
│   └── WorldLens_EDA_Dashboard.ipynb
├── src/
│   ├── eda.py             ← Data loading & cleaning pipeline
│   └── model.py           ← ML training, evaluation & model saving
├── dashboard/
│   └── app.py             ← Interactive Plotly Dash web app
├── models/
│   └── best_model.pkl     ← Best trained model (joblib)
├── outputs/               ← Saved charts (PNG)
└── requirements.txt

---

## 🚀 Quick Start

# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/WorldLens-Happiness-EDA.git
cd WorldLens-Happiness-EDA

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run EDA script
python src/eda.py

# 5. Train ML models
python src/model.py

# 6. Launch dashboard
python dashboard/app.py
# Open: http://127.0.0.1:8050

---

## 📊 Dataset

| Attribute  | Details                                                      |
|------------|--------------------------------------------------------------|
| Source     | World Happiness Report (Kaggle)                              |
| Years      | 2015 – 2022 (8 years)                                        |
| Countries  | 156 unique countries                                         |
| Total rows | 1,231 after merging                                          |
| Features   | GDP, Social Support, Health, Freedom, Corruption, Generosity |

---

## 📈 EDA — 11 Visualizations

| # | Chart | Key Insight |
|---|-------|-------------|
| 1 | Happiness Score Distribution | Slight negative skew |
| 2 | Pearson Correlation Heatmap | GDP has highest r=0.78 |
| 3 | Pairplot by Region | Western Europe clusters top-right |
| 4 | Violin Plot by Region | Sub-Saharan Africa widest spread |
| 5 | Year-over-Year Trend | Global avg stable ~5.4 |
| 6 | GDP vs Happiness Regression | Strong positive relationship |
| 7 | Top 10 Happiest Countries | Finland leads consistently |
| 8 | Bottom 10 Countries | Afghanistan at bottom |
| 9 | Feature Boxplots | Corruption most skewed |
| 10 | Region x Year Heatmap | ANZ region most stable |
| 11 | Skewness and Kurtosis | Corruption highest kurtosis |

---

## 🤖 Machine Learning

Models trained:
- Linear Regression (baseline)
- Random Forest Regressor (100 trees)
- Gradient Boosting Regressor (100 estimators)

Results: Random Forest and Gradient Boosting both achieved R² above 0.85.
GDP and Social Support are the top two most important features.

---

## 🖥️ Interactive Dashboard (Plotly Dash)

- 🌍 World choropleth map colored by happiness score
- 📊 Top 10 happiest countries bar chart
- 📉 Feature vs Happiness scatter with OLS trendline
- 📈 Year-over-year global trend line
- 🎻 Violin distribution by region
- 5 live KPI cards
- Year slider + Region + Feature dropdowns

---

## 🛠️ Tech Stack

| Category       | Libraries                          |
|----------------|------------------------------------|
| Data           | NumPy, Pandas                      |
| Visualization  | Matplotlib, Seaborn, Plotly        |
| Machine Learning | Scikit-learn, Joblib             |
| Dashboard      | Plotly Dash                        |
| Environment    | Python 3.11, venv                  |

---

## 👤 Author

Your Name
- LinkedIn: linkedin.com/in/yourprofile
- GitHub: github.com/yourusername
