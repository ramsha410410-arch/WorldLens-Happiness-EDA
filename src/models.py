import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, '..', 'data')
MODELS = os.path.join(BASE, '..', 'models')
OUT    = os.path.join(BASE, '..', 'outputs')
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────
def load_data():
    dfs = []
    col_map = {
        'Country or region':'Country','Country':'Country',
        'Score':'Happiness_Score','Happiness Score':'Happiness_Score',
        'Happiness.Score':'Happiness_Score',
        'GDP per capita':'GDP','Economy (GDP per Capita)':'GDP',
        'Economy..GDP.per.Capita.':'GDP',
        'Social support':'Social_Support','Family':'Social_Support',
        'Healthy life expectancy':'Health',
        'Health (Life Expectancy)':'Health',
        'Health..Life.Expectancy.':'Health',
        'Freedom to make life choices':'Freedom','Freedom':'Freedom',
        'Perceptions of corruption':'Corruption',
        'Trust (Government Corruption)':'Corruption',
        'Trust..Government.Corruption.':'Corruption',
        'Generosity':'Generosity','Region':'Region',
    }
    keep = ['Country','Region','Happiness_Score','GDP','Social_Support',
            'Health','Freedom','Corruption','Generosity','Year']
    for year in range(2015, 2023):
        f = os.path.join(DATA, f'{year}.csv')
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

# ── Features & target ──────────────────────────────────
FEATURES = ['GDP','Social_Support','Health','Freedom','Corruption','Generosity']
TARGET   = 'Happiness_Score'

def prepare(data):
    df = data[FEATURES + [TARGET]].dropna()
    X  = df[FEATURES]
    y  = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train ──────────────────────────────────────────────
def train(X_train, y_train):
    models = {
        'Linear Regression':  LinearRegression(),
        'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        trained[name] = m
        print(f"✅ Trained: {name}")
    return trained

# ── Evaluate ───────────────────────────────────────────
def evaluate(trained, X_test, y_test):
    print("\n" + "="*55)
    print("  MODEL EVALUATION RESULTS")
    print("="*55)
    results = {}
    for name, m in trained.items():
        pred = m.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae  = mean_absolute_error(y_test, pred)
        r2   = r2_score(y_test, pred)
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'model': m}
        print(f"\n  {name}")
        print(f"    R²   : {r2:.4f}")
        print(f"    RMSE : {rmse:.4f}")
        print(f"    MAE  : {mae:.4f}")
    return results

# ── Plot 1: Actual vs Predicted ────────────────────────
def plot_actual_vs_predicted(results, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, r) in zip(axes, results.items()):
        pred = r['model'].predict(X_test)
        ax.scatter(y_test, pred, alpha=0.5, color='steelblue',
                   edgecolors='white', linewidth=0.3)
        mn, mx = y_test.min(), y_test.max()
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
        ax.set_title(f"{name}\nR² = {r['R2']:.4f}", fontsize=12)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
    plt.suptitle('Actual vs Predicted — All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'actual_vs_predicted.png'), dpi=150)
    plt.show()
    print("✅ Saved: outputs/actual_vs_predicted.png")

# ── Plot 2: Feature Importance ─────────────────────────
def plot_feature_importance(results):
    rf = results['Random Forest']['model']
    importance = pd.Series(rf.feature_importances_,
                           index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot(kind='barh', color='coral', edgecolor='white')
    plt.title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'feature_importance.png'), dpi=150)
    plt.show()
    print("✅ Saved: outputs/feature_importance.png")

# ── Plot 3: Model Comparison ───────────────────────────
def plot_model_comparison(results):
    names = list(results.keys())
    r2s   = [results[n]['R2']   for n in names]
    rmses = [results[n]['RMSE'] for n in names]
    colors = ['steelblue', 'coral', 'mediumseagreen']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(names, r2s, color=colors)
    axes[0].set_title('R² Score (higher = better)', fontweight='bold')
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

    axes[1].bar(names, rmses, color=colors)
    axes[1].set_title('RMSE (lower = better)', fontweight='bold')
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 0.001, f'{v:.3f}', ha='center', fontsize=10)

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'model_comparison.png'), dpi=150)
    plt.show()
    print("✅ Saved: outputs/model_comparison.png")

# ── Plot 4: Residuals ──────────────────────────────────
def plot_residuals(results, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, r) in zip(axes, results.items()):
        pred      = r['model'].predict(X_test)
        residuals = y_test - pred
        ax.scatter(pred, residuals, alpha=0.5, color='purple',
                   edgecolors='white', linewidth=0.3)
        ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
        ax.set_title(f"{name}\nResiduals", fontsize=12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
    plt.suptitle('Residual Plots — All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'residuals.png'), dpi=150)
    plt.show()
    print("✅ Saved: outputs/residuals.png")

# ── Cross Validation ───────────────────────────────────
def cross_validate(trained, X_train, y_train):
    print("\n" + "="*55)
    print("  CROSS VALIDATION (5-fold)")
    print("="*55)
    for name, m in trained.items():
        scores = cross_val_score(m, X_train, y_train,
                                 cv=5, scoring='r2')
        print(f"\n  {name}")
        print(f"    CV R² scores : {[round(s,3) for s in scores]}")
        print(f"    Mean R²      : {scores.mean():.4f}")
        print(f"    Std          : {scores.std():.4f}")

# ── Save best model ────────────────────────────────────
def save_best(results):
    best_name  = max(results, key=lambda n: results[n]['R2'])
    best_model = results[best_name]['model']
    path = os.path.join(MODELS, 'best_model.pkl')
    joblib.dump(best_model, path)
    print(f"\n✅ Best model : {best_name}")
    print(f"   R²         : {results[best_name]['R2']:.4f}")
    print(f"   Saved to   : models/best_model.pkl")
    return best_model

# ── Main ───────────────────────────────────────────────
if __name__ == '__main__':
    print("📦 Loading data...")
    data = load_data()
    print(f"   Shape: {data.shape}")

    print("\n🔧 Preparing features...")
    X_train, X_test, y_train, y_test = prepare(data)
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

    print("\n🤖 Training models...")
    trained = train(X_train, y_train)

    results = evaluate(trained, X_test, y_test)

    cross_validate(trained, X_train, y_train)

    print("\n📊 Generating plots...")
    plot_actual_vs_predicted(results, X_test, y_test)
    plot_feature_importance(results)
    plot_model_comparison(results)
    plot_residuals(results, X_test, y_test)

    save_best(results)

    print("\n🎉 Phase 3 complete!")
