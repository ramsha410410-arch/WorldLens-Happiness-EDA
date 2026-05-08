import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    dfs = []
    col_map = {
        'Country or region': 'Country', 'Country': 'Country',
        'Score': 'Happiness_Score', 'Happiness Score': 'Happiness_Score',
        'Happiness.Score': 'Happiness_Score',
        'GDP per capita': 'GDP', 'Economy (GDP per Capita)': 'GDP',
        'Economy..GDP.per.Capita.': 'GDP',
        'Social support': 'Social_Support', 'Family': 'Social_Support',
        'Healthy life expectancy': 'Health',
        'Health (Life Expectancy)': 'Health',
        'Health..Life.Expectancy.': 'Health',
        'Freedom to make life choices': 'Freedom', 'Freedom': 'Freedom',
        'Perceptions of corruption': 'Corruption',
        'Trust (Government Corruption)': 'Corruption',
        'Trust..Government.Corruption.': 'Corruption',
        'Generosity': 'Generosity', 'Region': 'Region',
    }
    keep = ['Country','Region','Happiness_Score','GDP','Social_Support',
            'Health','Freedom','Corruption','Generosity','Year']
    for year in range(2015, 2023):
        f = os.path.join(DATA_DIR, f'{year}.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df['Year'] = year
        df.columns = df.columns.str.strip()
        df = df.rename(columns=col_map)
        df = df[[c for c in keep if c in df.columns]]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data['Region'] = data['Region'].fillna('Unknown')
    return data

if __name__ == '__main__':
    data = load_data()
    print(f"✅ Loaded: {data.shape}")
    print(data.head())
