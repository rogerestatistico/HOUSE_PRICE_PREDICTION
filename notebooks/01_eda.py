# =============================================================================
# DESAFIO DATA SCIENCE — Seattle House Prices
# Script 01: Análise Exploratória de Dados (EDA)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

BASE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ---------------------------------------------------------------------------
# 1. CARREGAMENTO
# ---------------------------------------------------------------------------
house  = pd.read_csv(os.path.join(DATA_DIR, "kc_house_data.csv"))
demo   = pd.read_csv(os.path.join(DATA_DIR, "zipcode_demographics.csv"))
future = pd.read_csv(os.path.join(DATA_DIR, "future_unseen_examples.csv"))

print("=== kc_house_data ===")
print(f"Shape: {house.shape}")
print(house.dtypes.to_string())
print(house.describe().round(2).to_string())

print("\n=== zipcode_demographics ===")
print(f"Shape: {demo.shape}")
print(demo.dtypes.to_string())

print("\n=== future_unseen_examples ===")
print(f"Shape: {future.shape}")
print(future.dtypes.to_string())

# ---------------------------------------------------------------------------
# 2. LIMPEZA E PREPARAÇÃO
# ---------------------------------------------------------------------------
house["date"]       = pd.to_datetime(house["date"], format="%Y%m%dT%H%M%S")
house["year_sold"]  = house["date"].dt.year
house["month_sold"] = house["date"].dt.month
house_clean = house.drop(columns=["id", "date"])

print("\n--- Qualidade dos dados ---")
print("Nulos kc_house_data:", house_clean.isnull().sum().sum())
print("Nulos zipcode_demographics:", demo.isnull().sum().sum())
print("Duplicatas kc_house_data:", house_clean.duplicated().sum())

# ---------------------------------------------------------------------------
# 3. MERGE COM DEMOGRAPHICS
# ---------------------------------------------------------------------------
demo["zipcode"]       = demo["zipcode"].astype(int)
house_clean["zipcode"] = house_clean["zipcode"].astype(int)
df = house_clean.merge(demo, on="zipcode", how="left")
print(f"\nShape após merge: {df.shape}")
print("Nulos após merge:", df.isnull().sum().sum())
df.to_csv(os.path.join(OUT_DIR, "merged_dataset.csv"), index=False)

# Estatísticas descritivas
desc = df[["price","sqft_living","bedrooms","bathrooms","grade","yr_built",
           "medn_hshld_incm_amt","hous_val_amt"]].describe().round(2)
print("\nEstatísticas descritivas:\n", desc.to_string())
desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"))

# ---------------------------------------------------------------------------
# 4. DISTRIBUIÇÃO DO PREÇO
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["price"] / 1e6, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Preço (milhões USD)")
axes[0].set_ylabel("Quantidade de imóveis")
axes[0].set_title("Distribuição do Preço")
axes[0].axvline(df["price"].median()/1e6, color="red", linestyle="--",
                label=f'Mediana: ${df["price"].median()/1e6:.2f}M')
axes[0].legend()

axes[1].hist(np.log1p(df["price"]), bins=80, color="#55A868", edgecolor="white", linewidth=0.5)
axes[1].set_xlabel("log(Preço + 1)")
axes[1].set_ylabel("Quantidade de imóveis")
axes[1].set_title("Distribuição do Preço (escala log)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_price_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 01_price_distribution.png")

# ---------------------------------------------------------------------------
# 5. CORRELAÇÃO COM PREÇO
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_price = df[numeric_cols].corr()["price"].drop("price").sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 10))
colors = ["#4C72B0" if v > 0 else "#C44E52" for v in corr_price.values]
ax.barh(corr_price.index, corr_price.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Correlação de Pearson com Preço")
ax.set_title("Correlação das Features com o Preço")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_correlation_price.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 02_correlation_price.png")

# ---------------------------------------------------------------------------
# 6. HEATMAP DAS PRINCIPAIS FEATURES
# ---------------------------------------------------------------------------
top_features = ["price","sqft_living","grade","sqft_above","sqft_living15",
                "bathrooms","bedrooms","floors","yr_built","waterfront",
                "medn_hshld_incm_amt","hous_val_amt","medn_incm_per_prsn_amt"]
top_features = [c for c in top_features if c in df.columns]

fig, ax = plt.subplots(figsize=(13, 10))
corr_matrix = df[top_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
ax.set_title("Heatmap de Correlação — Principais Features")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 03_heatmap.png")

# ---------------------------------------------------------------------------
# 7. SCATTER: sqft_living e grade vs preço
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sample = df.sample(3000, random_state=42)

axes[0].scatter(sample["sqft_living"], sample["price"]/1e6, alpha=0.3, s=10, color="#4C72B0")
axes[0].set_xlabel("sqft_living (área interna)")
axes[0].set_ylabel("Preço (milhões USD)")
axes[0].set_title("Área Interna vs Preço")

axes[1].scatter(sample["grade"], sample["price"]/1e6, alpha=0.3, s=10, color="#DD8452")
axes[1].set_xlabel("Grade (qualidade da construção)")
axes[1].set_ylabel("Preço (milhões USD)")
axes[1].set_title("Grade vs Preço")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_scatter_key_features.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 04_scatter_key_features.png")

# ---------------------------------------------------------------------------
# 8. WATERFRONT e VIEW
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

wf_medians = df.groupby("waterfront")["price"].median() / 1e6
axes[0].bar(["Sem Vista para Água", "Com Vista para Água"], wf_medians.values,
            color=["#4C72B0", "#55A868"])
axes[0].set_ylabel("Mediana do Preço (milhões USD)")
axes[0].set_title("Impacto da Vista para Água no Preço")
for i, v in enumerate(wf_medians.values):
    axes[0].text(i, v + 0.02, f"${v:.2f}M", ha="center", fontweight="bold")

view_medians = df.groupby("view")["price"].median() / 1e6
axes[1].bar(view_medians.index.astype(str), view_medians.values,
            color=sns.color_palette("Blues_d", 5))
axes[1].set_xlabel("Score de Vista (0-4)")
axes[1].set_ylabel("Mediana do Preço (milhões USD)")
axes[1].set_title("Score de Vista vs Preço")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_waterfront_view.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 05_waterfront_view.png")

# ---------------------------------------------------------------------------
# 9. BOXPLOT PREÇO POR GRADE
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
grades = sorted(df["grade"].unique())
data_by_grade = [df[df["grade"] == g]["price"].values / 1e6 for g in grades]
bp = ax.boxplot(data_by_grade, labels=grades, patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=2, alpha=0.3))
palette = sns.color_palette("RdYlGn", len(grades))
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
ax.set_xlabel("Grade (Qualidade da Construção)")
ax.set_ylabel("Preço (milhões USD)")
ax.set_title("Distribuição do Preço por Grade — Identificação de Outliers")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_boxplot_grade.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 06_boxplot_grade.png")

# ---------------------------------------------------------------------------
# 10. PREÇO MEDIANO POR ZIPCODE (TOP 20)
# ---------------------------------------------------------------------------
zip_price = df.groupby("zipcode")["price"].median().sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(13, 6))
zip_price.plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_xlabel("Zipcode")
ax.set_ylabel("Mediana do Preço (USD)")
ax.set_title("Top 20 Zipcodes por Mediana de Preço")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_price_by_zipcode.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 07_price_by_zipcode.png")

# ---------------------------------------------------------------------------
# 11. RENDA vs PREÇO POR ZIPCODE
# ---------------------------------------------------------------------------
zip_agg = df.groupby("zipcode").agg(
    median_price=("price", "median"),
    median_income=("medn_hshld_incm_amt", "median"),
    count=("price", "count")
).reset_index()

fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(zip_agg["median_income"]/1e3, zip_agg["median_price"]/1e6,
                s=zip_agg["count"]*0.5, alpha=0.6, c=zip_agg["median_price"],
                cmap="RdYlGn", edgecolors="white", linewidth=0.4)
plt.colorbar(sc, ax=ax, label="Mediana Preço (USD)")
ax.set_xlabel("Renda Mediana Domiciliar (mil USD)")
ax.set_ylabel("Mediana do Preço (milhões USD)")
ax.set_title("Renda Domiciliar vs Preço por Zipcode\n(tamanho = nº de imóveis)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_income_vs_price.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 08_income_vs_price.png")

# ---------------------------------------------------------------------------
# 12. EVOLUÇÃO TEMPORAL
# ---------------------------------------------------------------------------
monthly = df.groupby(["year_sold","month_sold"])["price"].median().reset_index()
monthly["period"] = pd.to_datetime(
    monthly[["year_sold","month_sold"]].rename(
        columns={"year_sold":"year","month_sold":"month"}).assign(day=1))
monthly = monthly.sort_values("period")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly["period"], monthly["price"]/1e6, marker="o", linewidth=2, color="#4C72B0")
ax.fill_between(monthly["period"], monthly["price"]/1e6, alpha=0.15, color="#4C72B0")
ax.set_xlabel("Período")
ax.set_ylabel("Mediana do Preço (milhões USD)")
ax.set_title("Evolução Temporal da Mediana de Preço")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_price_over_time.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 09_price_over_time.png")

# ---------------------------------------------------------------------------
# 13. EDUCAÇÃO vs PREÇO
# ---------------------------------------------------------------------------
edu_corr = df[["per_bchlr","per_prfsnl","per_hsd","per_less_than_9","price"]].corr()["price"].drop("price")
labels = ["% Bacharelado","% Pós-Graduação","% Ensino Médio","% < 9 anos escola"]
colors = ["#55A868" if v > 0 else "#C44E52" for v in edu_corr.values]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, edu_corr.values, color=colors)
ax.set_ylabel("Correlação com Preço")
ax.set_title("Nível Educacional do Zipcode vs Preço do Imóvel")
ax.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "10_education_vs_price.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: 10_education_vs_price.png")

print("\n=== EDA CONCLUIDA — todos os graficos salvos em /outputs ===")
