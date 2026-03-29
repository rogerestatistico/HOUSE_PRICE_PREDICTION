# =============================================================================
# DESAFIO DATA SCIENCE — Seattle House Prices
# Script 02: Feature Engineering + Modelagem ML + Previsões
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
import json

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. CARREGAMENTO
# ---------------------------------------------------------------------------
house  = pd.read_csv(os.path.join(DATA_DIR, "kc_house_data.csv"))
demo   = pd.read_csv(os.path.join(DATA_DIR, "zipcode_demographics.csv"))
future = pd.read_csv(os.path.join(DATA_DIR, "future_unseen_examples.csv"))

house["date"]       = pd.to_datetime(house["date"], format="%Y%m%dT%H%M%S")
house["year_sold"]  = house["date"].dt.year
house["month_sold"] = house["date"].dt.month
house.drop(columns=["id", "date"], inplace=True)

demo["zipcode"]   = demo["zipcode"].astype(int)
house["zipcode"]  = house["zipcode"].astype(int)
future["zipcode"] = future["zipcode"].astype(int)

df = house.merge(demo, on="zipcode", how="left")

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(data, is_future=False):
    d = data.copy()

    # Idade e renovação
    ref_year = 2015
    d["age"]           = ref_year - d["yr_built"]
    d["was_renovated"] = (d["yr_renovated"] > 0).astype(int)
    d["years_since_reno"] = np.where(d["yr_renovated"] > 0,
                                      ref_year - d["yr_renovated"],
                                      ref_year - d["yr_built"])

    # Ratios de área
    d["sqft_ratio"]    = d["sqft_living"] / (d["sqft_lot"] + 1)
    d["basement_ratio"] = d["sqft_basement"] / (d["sqft_living"] + 1)
    d["above_ratio"]   = d["sqft_above"] / (d["sqft_living"] + 1)

    # Interações chave
    d["grade_sqft"]     = d["grade"] * d["sqft_living"]
    d["grade_age"]      = d["grade"] * d["age"]
    d["bath_bed_ratio"] = d["bathrooms"] / (d["bedrooms"] + 1)

    # Variável de sazonalidade (apenas para treino)
    if not is_future and "month_sold" in d.columns:
        d["high_season"] = d["month_sold"].isin([3,4,5,6,7]).astype(int)
    else:
        d["high_season"] = 0

    # Interação com demographics
    if "medn_hshld_incm_amt" in d.columns:
        d["income_grade"] = d["medn_hshld_incm_amt"] * d["grade"]

    return d

df_feat    = engineer_features(df)
future_dem = future.merge(demo, on="zipcode", how="left")
fut_feat   = engineer_features(future_dem, is_future=True)

# ---------------------------------------------------------------------------
# 3. SELEÇÃO DE FEATURES
# ---------------------------------------------------------------------------
FEATURES = [
    # Físicas
    "sqft_living", "sqft_lot", "sqft_above", "sqft_basement",
    "sqft_living15", "sqft_lot15",
    "bedrooms", "bathrooms", "floors",
    "waterfront", "view", "condition", "grade",
    "yr_built", "yr_renovated",
    # Geográficas
    "lat", "long", "zipcode",
    # Engenhadas
    "age", "was_renovated", "years_since_reno",
    "sqft_ratio", "basement_ratio", "above_ratio",
    "grade_sqft", "grade_age", "bath_bed_ratio",
    "high_season",
    # Demographics
    "medn_hshld_incm_amt", "medn_incm_per_prsn_amt",
    "hous_val_amt", "ppltn_qty",
    "per_bchlr", "per_prfsnl", "per_urbn",
    "income_grade",
]

# Garantir que todas as features existem
FEATURES = [f for f in FEATURES if f in df_feat.columns]
print(f"Total de features utilizadas: {len(FEATURES)}")

X = df_feat[FEATURES]
y = np.log1p(df_feat["price"])   # target em escala log para estabilizar variância

X_fut = fut_feat[FEATURES]

# ---------------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 5. COMPARAÇÃO DE MODELOS (5-Fold CV)
# ---------------------------------------------------------------------------
models = {
    "Ridge":              Ridge(alpha=10),
    "Random Forest":      RandomForestRegressor(n_estimators=200, max_depth=15,
                                                min_samples_leaf=5, n_jobs=-1, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                    max_depth=5, subsample=0.8, random_state=42),
    "XGBoost":            XGBRegressor(n_estimators=500, learning_rate=0.05,
                                       max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                       reg_alpha=0.1, reg_lambda=1.0,
                                       random_state=42, verbosity=0),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
print("\n--- 5-Fold Cross-Validation (RMSE no log-price) ---")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
    cv_results[name] = -cv_scores
    print(f"{name:25s}: RMSE = {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# 6. TREINO DO MELHOR MODELO (XGBoost)
# ---------------------------------------------------------------------------
best_model = models["XGBoost"]
best_model.fit(X_train, y_train)

# Avaliação no conjunto de teste
y_pred_log = best_model.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_true     = np.expm1(y_test)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"\n--- Métricas no Conjunto de Teste (XGBoost) ---")
print(f"MAE:  ${mae:,.0f}")
print(f"RMSE: ${rmse:,.0f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2),
           "R2": round(r2, 4), "MAPE": round(mape, 2)}
with open(os.path.join(OUT_DIR, "model_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------------------
# 7. COMPARAÇÃO DE MODELOS — GRÁFICO
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(cv_results.keys())
means  = [cv_results[m].mean() for m in model_names]
stds   = [cv_results[m].std()  for m in model_names]
colors = ["#C44E52" if m != "XGBoost" else "#55A868" for m in model_names]

bars = ax.bar(model_names, means, yerr=stds, color=colors,
              capsize=5, edgecolor="white", linewidth=0.8)
ax.set_ylabel("RMSE Médio (5-Fold CV, log-price)")
ax.set_title("Comparação de Modelos — Cross-Validation")
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "11_model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 11_model_comparison.png")

# ---------------------------------------------------------------------------
# 8. IMPORTÂNCIA DAS FEATURES
# ---------------------------------------------------------------------------
importances = pd.Series(best_model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False).head(25)

fig, ax = plt.subplots(figsize=(10, 9))
importances.plot(kind="barh", ax=ax, color="#4C72B0")
ax.invert_yaxis()
ax.set_xlabel("Importância (XGBoost gain)")
ax.set_title("Top 25 Features Mais Importantes")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "12_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 12_feature_importance.png")

# ---------------------------------------------------------------------------
# 9. REAL vs PREVISTO
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sample_idx = np.random.choice(len(y_true), size=min(2000, len(y_true)), replace=False)
y_true_s = np.array(y_true)[sample_idx]
y_pred_s = np.array(y_pred)[sample_idx]

axes[0].scatter(y_true_s/1e6, y_pred_s/1e6, alpha=0.3, s=10, color="#4C72B0")
lims = [min(y_true_s.min(), y_pred_s.min())/1e6, max(y_true_s.max(), y_pred_s.max())/1e6]
axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Previsão Perfeita")
axes[0].set_xlabel("Preço Real (milhões USD)")
axes[0].set_ylabel("Preço Previsto (milhões USD)")
axes[0].set_title(f"Real vs Previsto\nR² = {r2:.4f}")
axes[0].legend()

residuals = y_true_s - y_pred_s
axes[1].scatter(y_pred_s/1e6, residuals/1e3, alpha=0.3, s=10, color="#DD8452")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Preço Previsto (milhões USD)")
axes[1].set_ylabel("Resíduo (mil USD)")
axes[1].set_title("Resíduos vs Previsto")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "13_actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 13_actual_vs_predicted.png")

# ---------------------------------------------------------------------------
# 10. CROSS-VALIDATION — DETALHE XGBoost
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, 6), cv_results["XGBoost"], color="#4C72B0", edgecolor="white")
ax.axhline(cv_results["XGBoost"].mean(), color="red", linestyle="--",
           label=f'Média: {cv_results["XGBoost"].mean():.4f}')
ax.set_xlabel("Fold")
ax.set_ylabel("RMSE (log-price)")
ax.set_title("XGBoost — RMSE por Fold (5-Fold CV)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "14_cv_folds.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 14_cv_folds.png")

# ---------------------------------------------------------------------------
# 11. MÉTRICAS DE NEGÓCIO — GRÁFICO RESUMO
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Erro percentual por faixa de preço
df_test_eval = pd.DataFrame({"real": y_true, "previsto": y_pred})
bins = [0, 300e3, 500e3, 700e3, 1e6, float("inf")]
labels = ["< 300k", "300k-500k", "500k-700k", "700k-1M", "> 1M"]
df_test_eval["faixa"] = pd.cut(df_test_eval["real"], bins=bins, labels=labels)
df_test_eval["pct_error"] = np.abs(df_test_eval["real"] - df_test_eval["previsto"]) / df_test_eval["real"] * 100

faixa_error = df_test_eval.groupby("faixa", observed=True)["pct_error"].median()
axes[0].bar(faixa_error.index, faixa_error.values, color="#4C72B0")
axes[0].set_xlabel("Faixa de Preço")
axes[0].set_ylabel("Erro Percentual Absoluto Mediano (%)")
axes[0].set_title("Precisão do Modelo por Faixa de Preço")
axes[0].axhline(10, color="red", linestyle="--", label="Limite 10%")
axes[0].legend()

# Distribuição dos erros percentuais
axes[1].hist(df_test_eval["pct_error"].clip(0, 50), bins=50, color="#55A868",
             edgecolor="white", linewidth=0.5)
axes[1].axvline(mape, color="red", linestyle="--", label=f"MAPE: {mape:.1f}%")
axes[1].set_xlabel("Erro Percentual Absoluto (%)")
axes[1].set_ylabel("Quantidade de imóveis")
axes[1].set_title("Distribuição dos Erros de Previsão")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "15_business_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 15_business_metrics.png")

# ---------------------------------------------------------------------------
# 12. PREVISÕES PARA future_unseen_examples
# ---------------------------------------------------------------------------
# Adicionar colunas de engenharia que podem não existir no futuro
if "year_sold" not in fut_feat.columns:
    fut_feat["year_sold"]  = 2015
    fut_feat["month_sold"] = 6

# Verificar features ausentes
missing = [f for f in FEATURES if f not in fut_feat.columns]
print(f"\nFeatures ausentes no futuro: {missing}")
for col in missing:
    fut_feat[col] = 0

X_fut_final = fut_feat[FEATURES]
pred_log  = best_model.predict(X_fut_final)
pred_price = np.expm1(pred_log)

future_out = future.copy()
future_out["predicted_price"] = pred_price.round(2)
future_out.to_csv(os.path.join(OUT_DIR, "predictions_future.csv"), index=False)

print(f"\n--- Previsões para {len(future_out)} imóveis ---")
print(future_out[["zipcode","bedrooms","sqft_living","grade","predicted_price"]].to_string())
print(f"\nMédiana das previsões: ${future_out['predicted_price'].median():,.0f}")
print(f"Mínimo: ${future_out['predicted_price'].min():,.0f}")
print(f"Máximo: ${future_out['predicted_price'].max():,.0f}")

# Gráfico das previsões
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(future_out["predicted_price"]/1e6, bins=25, color="#4C72B0",
        edgecolor="white", linewidth=0.5)
ax.axvline(future_out["predicted_price"].median()/1e6, color="red", linestyle="--",
           label=f'Mediana: ${future_out["predicted_price"].median()/1e6:.2f}M')
ax.set_xlabel("Preço Previsto (milhões USD)")
ax.set_ylabel("Quantidade de imóveis")
ax.set_title("Distribuição das Previsões — future_unseen_examples")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "16_future_predictions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 16_future_predictions.png")

print("\n=== MODELAGEM CONCLUIDA ===")
print(f"Melhor modelo: XGBoost | R²={r2:.4f} | MAPE={mape:.2f}% | MAE=${mae:,.0f}")
