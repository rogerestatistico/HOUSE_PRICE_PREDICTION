# =============================================================================
# DESAFIO DATA SCIENCE — Seattle House Prices
# Script 03: Diagrama de Deploy e Aprendizado Contínuo
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================================
# DIAGRAMA 1: ARQUITETURA DE DEPLOY
# ===========================================================================
fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis("off")
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#F8F9FA")

def box(ax, x, y, w, h, label, sublabel="", color="#4C72B0", fontsize=10, text_color="white"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor="white", linewidth=1.5,
                          zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
            label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha="center", va="center", fontsize=7.5,
                color=text_color, alpha=0.85, zorder=4)

def arrow(ax, x1, y1, x2, y2, label="", color="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=18), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.12, label, ha="center", va="bottom",
                fontsize=7.5, color=color, style="italic")

# ---- Título
ax.text(9, 11.4, "Arquitetura de Deploy — Previsão de Preços de Imóveis",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#2C3E50")

# ---- CAMADA 1: Clientes
ax.text(1.5, 10.8, "CLIENTES / CONSUMIDORES", ha="center", fontsize=8,
        color="#888", fontweight="bold")
box(ax, 0.2, 9.8, 2.4, 0.85, "Web App", "Dashboard Streamlit", "#2980B9")
box(ax, 2.8, 9.8, 2.4, 0.85, "Mobile / API", "REST JSON", "#2980B9")
box(ax, 5.4, 9.8, 2.4, 0.85, "Batch Jobs", "Previsão em lote", "#2980B9")

# ---- CAMADA 2: API Gateway / Load Balancer
ax.text(9, 9.55, "API GATEWAY & BALANCEAMENTO", ha="center", fontsize=8,
        color="#888", fontweight="bold")
box(ax, 6.5, 8.65, 5.0, 0.75, "API Gateway + Load Balancer",
    "FastAPI  |  Nginx  |  Rate Limiting  |  Auth JWT", "#E67E22")

# Setas clientes -> gateway
for cx in [1.4, 4.0, 6.6]:
    arrow(ax, cx, 9.8, cx + (6.5+2.5-cx)*0.5, 9.4, color="#2980B9")
    # simplificado
arrow(ax, 1.4, 9.8, 7.5, 9.4, color="#2980B9")
arrow(ax, 4.0, 9.8, 8.5, 9.4, color="#2980B9")
arrow(ax, 6.6, 9.8, 9.5, 9.4, color="#2980B9")

# ---- CAMADA 3: Serviço de Predição
ax.text(9, 8.5, "SERVIÇO DE PREDIÇÃO", ha="center", fontsize=8,
        color="#888", fontweight="bold")
box(ax, 6.5, 7.55, 5.0, 0.8, "Prediction Service",
    "Pré-proc. | Feature Eng. | XGBoost | Pós-proc.", "#27AE60")
arrow(ax, 9.0, 8.65, 9.0, 8.35, color="#E67E22")

# ---- CAMADA 4: Model Registry
box(ax, 12.2, 7.55, 3.2, 0.8, "Model Registry",
    "MLflow | v1, v2, v3...", "#8E44AD")
arrow(ax, 11.5, 7.95, 12.2, 7.95, label="load model", color="#8E44AD")

# ---- CAMADA 5: Feature Store & Cache
box(ax, 0.3, 7.55, 3.2, 0.8, "Feature Store / Cache",
    "Redis | Demographics ZIP", "#16A085")
arrow(ax, 3.5, 7.95, 6.5, 7.95, label="features", color="#16A085")

# ---- CAMADA 6: Infraestrutura
ax.text(5.5, 7.3, "INFRAESTRUTURA", ha="center", fontsize=8,
        color="#888", fontweight="bold")
box(ax, 0.3, 6.3, 2.3, 0.8, "Docker\nContainers", "", "#7F8C8D", fontsize=9)
box(ax, 2.8, 6.3, 2.3, 0.8, "Kubernetes\nOrchestration", "", "#7F8C8D", fontsize=9)
box(ax, 5.3, 6.3, 2.3, 0.8, "AWS / GCP\nCloud", "", "#7F8C8D", fontsize=9)
box(ax, 7.8, 6.3, 2.3, 0.8, "CI/CD\nGitHub Actions", "", "#7F8C8D", fontsize=9)
box(ax, 10.3, 6.3, 2.3, 0.8, "Auto-Scaling\nPods", "", "#7F8C8D", fontsize=9)
box(ax, 12.8, 6.3, 2.3, 0.8, "S3 / GCS\nArtifacts", "", "#7F8C8D", fontsize=9)

for xi in [1.45, 3.95, 6.45, 8.95, 11.45, 13.95]:
    arrow(ax, xi, 7.55, xi, 7.1, color="#7F8C8D")

# ---- CAMADA 7: Monitoramento
ax.text(9, 5.95, "MONITORAMENTO & ALERTAS", ha="center", fontsize=8,
        color="#888", fontweight="bold")
box(ax, 0.3, 4.95, 3.5, 0.8, "Data Drift Monitor",
    "Evidently AI | Kolmogorov-Smirnov", "#C0392B")
box(ax, 4.1, 4.95, 3.5, 0.8, "Performance Monitor",
    "MAE, RMSE, R² em produção", "#C0392B")
box(ax, 7.9, 4.95, 3.5, 0.8, "Infra Monitor",
    "Prometheus + Grafana", "#C0392B")
box(ax, 11.7, 4.95, 3.5, 0.8, "Alerting",
    "PagerDuty | Slack | Email", "#C0392B")

for xi in [2.05, 5.85, 9.65, 13.45]:
    arrow(ax, xi, 6.3, xi, 5.75, color="#C0392B")

# ---- CAMADA 8: Retraining Pipeline
ax.text(9, 4.65, "PIPELINE DE RETRAINING (APRENDIZADO CONTÍNUO)", ha="center",
        fontsize=8, color="#888", fontweight="bold")
box(ax, 0.3, 3.65, 3.5, 0.8, "Novos Dados",
    "Fonte: MLS, Registros públicos", "#1A5276", text_color="white")
box(ax, 4.1, 3.65, 3.5, 0.8, "Data Pipeline",
    "Airflow | Validação Great Expectations", "#1A5276", text_color="white")
box(ax, 7.9, 3.65, 3.5, 0.8, "Retraining Job",
    "Spark | Sklearn | XGBoost", "#1A5276", text_color="white")
box(ax, 11.7, 3.65, 3.5, 0.8, "A/B Testing",
    "Novo vs Atual | Champion/Challenger", "#1A5276", text_color="white")

for xi in [2.05, 5.85, 9.65]:
    arrow(ax, xi + 3.5 - 1.45, 4.05, xi + 3.5 + 0.1, 4.05, color="#1A5276")

# Seta volta ao model registry
arrow(ax, 13.45, 3.65, 13.75, 8.35, label="deploy\nnovo modelo", color="#8E44AD")

# Seta monitoramento dispara retraining
arrow(ax, 5.85, 4.95, 2.05, 4.45, label="trigger", color="#C0392B")

# ---- CAMADA 9: Data Storage
ax.text(9, 3.35, "ARMAZENAMENTO DE DADOS", ha="center", fontsize=8,
        color="#888", fontweight="bold")
box(ax, 0.3, 2.35, 3.5, 0.8, "Data Warehouse",
    "BigQuery | Redshift — dados históricos", "#566573")
box(ax, 4.1, 2.35, 3.5, 0.8, "Feature Store DB",
    "PostgreSQL | Hopsworks", "#566573")
box(ax, 7.9, 2.35, 3.5, 0.8, "Prediction Logs",
    "MongoDB | S3 — todas as predições", "#566573")
box(ax, 11.7, 2.35, 3.5, 0.8, "Ground Truth Store",
    "Preços reais após fechamento", "#566573")

for xi in [2.05, 5.85, 9.65, 13.45]:
    arrow(ax, xi, 3.65, xi, 3.15, color="#566573")

# ---- LEGENDA
legend_items = [
    mpatches.Patch(color="#2980B9", label="Clientes / Interfaces"),
    mpatches.Patch(color="#E67E22", label="API Gateway"),
    mpatches.Patch(color="#27AE60", label="Serviço de Predição"),
    mpatches.Patch(color="#8E44AD", label="Model Registry (MLflow)"),
    mpatches.Patch(color="#16A085", label="Feature Store"),
    mpatches.Patch(color="#7F8C8D", label="Infraestrutura Cloud"),
    mpatches.Patch(color="#C0392B", label="Monitoramento"),
    mpatches.Patch(color="#1A5276", label="Retraining Pipeline"),
    mpatches.Patch(color="#566573", label="Armazenamento"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0, 0),
          ncol=3, fontsize=8, framealpha=0.9, edgecolor="#CCC")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "deploy_architecture.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: diagrams/deploy_architecture.png")

# ===========================================================================
# DIAGRAMA 2: FLUXO DE APRENDIZADO CONTÍNUO
# ===========================================================================
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#F8F9FA")

ax.text(8, 7.5, "Ciclo de Aprendizado Contínuo — House Price Model",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#2C3E50")

steps = [
    (1.5, 5.5, "1. Coleta\nde Novos\nDados",    "#2980B9"),
    (4.0, 5.5, "2. Validação\n& Feature\nEng.", "#16A085"),
    (6.5, 5.5, "3. Retraining\nXGBoost",        "#27AE60"),
    (9.0, 5.5, "4. Avaliação\nMAE/RMSE/R²",     "#F39C12"),
    (11.5, 5.5, "5. A/B Test\nChampion vs\nChallenger", "#8E44AD"),
    (14.0, 5.5, "6. Deploy\nem Produção",        "#E74C3C"),
]

for (x, y, label, color) in steps:
    circle = plt.Circle((x, y), 0.9, color=color, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white", zorder=4, ma="center")

for i in range(len(steps) - 1):
    x1 = steps[i][0] + 0.9
    x2 = steps[i+1][0] - 0.9
    y  = steps[i][1]
    arrow(ax, x1, y, x2, y, color="#555")

# Seta de retorno (loop)
ax.annotate("", xy=(1.5, 4.5), xytext=(14.0, 4.5),
            arrowprops=dict(arrowstyle="-|>", color="#C0392B",
                            lw=2, mutation_scale=18,
                            connectionstyle="arc3,rad=-0.3"), zorder=2)
ax.text(7.75, 3.5, "Feedback Loop: preços reais retroalimentam o modelo",
        ha="center", fontsize=9, color="#C0392B", style="italic")

# Caixa de gatilho
box(ax, 0.5, 1.5, 15, 1.5,
    "Gatilhos de Retraining",
    "• Data Drift detectado (KS-test p < 0.05)  •  Degradação de performance (MAE > threshold)\n"
    "• Novo volume de dados (ex: +500 imóveis)  •  Agendamento periódico (mensal)",
    "#2C3E50", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "continuous_learning.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Salvo: diagrams/continuous_learning.png")

print("\n=== DIAGRAMAS CONCLUIDOS ===")
