"""
Converte os 3 scripts .py em Jupyter Notebooks .ipynb e faz upload no GitHub.
"""
import os, json, base64, urllib.request, urllib.error, nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

TOKEN    = os.environ.get("GITHUB_TOKEN", "")
REPO     = "HOUSE_PRICE_PREDICTION"
USERNAME = "rogerestatistico"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NB_DIR   = os.path.join(BASE_DIR, "notebooks")

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json",
    "Content-Type": "application/json",
    "X-GitHub-Api-Version": "2022-11-28",
}

def gh_put(rel_path, content_bytes, message):
    url = f"https://api.github.com/repos/{USERNAME}/{REPO}/contents/{rel_path}"
    b64 = base64.b64encode(content_bytes).decode()
    # pegar SHA se existir
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req) as r:
            sha = json.loads(r.read().decode()).get("sha")
    except urllib.error.HTTPError:
        sha = None
    payload = {"message": message, "content": b64}
    if sha:
        payload["sha"] = sha
    data = json.dumps(payload).encode()
    req2 = urllib.request.Request(url, data=data, headers=HEADERS, method="PUT")
    try:
        with urllib.request.urlopen(req2) as r:
            return r.status
    except urllib.error.HTTPError as e:
        print(f"  ERRO {e.code}: {e.read().decode()[:200]}")
        return e.code

# ---------------------------------------------------------------------------
# Utilitário: quebrar script em células por blocos de comentário "# ---..."
# ---------------------------------------------------------------------------
def script_to_cells(script_path):
    with open(script_path, encoding="utf-8") as f:
        lines = f.readlines()

    cells = []
    current_block = []
    current_title = None

    def flush(block, title):
        code = "".join(block).strip()
        if not code:
            return
        cell_list = []
        if title:
            cell_list.append(new_markdown_cell(f"## {title}"))
        cell_list.append(new_code_cell(code))
        return cell_list

    for line in lines:
        stripped = line.strip()
        # Detectar separador de seção: "# ---... TITULO ..."
        if stripped.startswith("# ---") and len(stripped) > 10:
            result = flush(current_block, current_title)
            if result:
                cells.extend(result)
            current_block = []
            # extrair título do separador
            title_raw = stripped.strip("# -").strip()
            current_title = title_raw if title_raw else None
        else:
            current_block.append(line)

    result = flush(current_block, current_title)
    if result:
        cells.extend(result)

    return cells

# ---------------------------------------------------------------------------
# Definição dos notebooks
# ---------------------------------------------------------------------------
notebooks = [
    {
        "script": os.path.join(NB_DIR, "01_eda.py"),
        "output": os.path.join(NB_DIR, "01_EDA_Analise_Exploratoria.ipynb"),
        "gh_path": "notebooks/01_EDA_Analise_Exploratoria.ipynb",
        "title": "EDA — Análise Exploratória de Dados",
        "intro": """# Análise Exploratória de Dados (EDA)
## Desafio Data Science — Previsão de Preços de Imóveis em Seattle

Este notebook cobre:
- Entendimento das variáveis e qualidade dos dados
- Merge dos dados físicos com dados demográficos por zipcode
- Distribuição de preços, correlações e outliers
- Padrões geográficos, temporais e socioeconômicos
""",
    },
    {
        "script": os.path.join(NB_DIR, "02_modeling.py"),
        "output": os.path.join(NB_DIR, "02_Feature_Engineering_Modelagem.ipynb"),
        "gh_path": "notebooks/02_Feature_Engineering_Modelagem.ipynb",
        "title": "Feature Engineering + Modelagem ML + Previsões",
        "intro": """# Feature Engineering, Modelagem ML e Previsões
## Desafio Data Science — Previsão de Preços de Imóveis em Seattle

Este notebook cobre:
- Criação de novas features (engenharia de variáveis)
- Comparação de modelos via 5-Fold Cross-Validation
- Treinamento e avaliação do XGBoost (modelo selecionado)
- Importância das features e análise de resíduos
- Previsões para os 100 imóveis sem preço (`future_unseen_examples.csv`)

**Resultados:** R² = 0.9138 | MAE = $63.127 | MAPE = 11.56%
""",
    },
    {
        "script": os.path.join(NB_DIR, "03_deploy_diagram.py"),
        "output": os.path.join(NB_DIR, "03_Estrategia_Deploy_Aprendizado_Continuo.ipynb"),
        "gh_path": "notebooks/03_Estrategia_Deploy_Aprendizado_Continuo.ipynb",
        "title": "Estratégia de Deploy e Aprendizado Contínuo",
        "intro": """# Estratégia de Deploy e Aprendizado Contínuo
## Desafio Data Science — Previsão de Preços de Imóveis em Seattle

Este notebook cobre:
- Diagrama completo de arquitetura de deploy em produção
- Camadas: API Gateway, Prediction Service, Model Registry, Monitoramento
- Pipeline de retraining com gatilhos automáticos
- Ciclo de aprendizado contínuo (Champion/Challenger, A/B Testing)

> **Nota:** O deploy não foi implementado — apenas documentado e diagramado conforme solicitado.
""",
    },
]

# ---------------------------------------------------------------------------
# Gerar e fazer upload
# ---------------------------------------------------------------------------
for nb_def in notebooks:
    print(f"\nProcessando: {os.path.basename(nb_def['output'])}")

    cells = [new_markdown_cell(nb_def["intro"])]
    cells.extend(script_to_cells(nb_def["script"]))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.12.7"
    }

    nb_bytes = nbformat.writes(nb).encode("utf-8")

    # Salvar localmente
    with open(nb_def["output"], "wb") as f:
        f.write(nb_bytes)
    print(f"  Salvo localmente: {nb_def['output']}")

    # Upload GitHub
    status = gh_put(nb_def["gh_path"], nb_bytes,
                    f"add notebook: {os.path.basename(nb_def['output'])}")
    print(f"  GitHub [{status}]: {nb_def['gh_path']}")

print("\n=== Notebooks gerados e enviados ao GitHub ===")
print(f"https://github.com/{USERNAME}/{REPO}/tree/main/notebooks")
