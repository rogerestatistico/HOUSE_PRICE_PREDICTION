"""
Cria repositório no GitHub e faz upload de todos os arquivos via API REST.
"""
import os
import base64
import json
import urllib.request
import urllib.error

TOKEN    = os.environ.get("GITHUB_TOKEN", "")  # defina via variável de ambiente
REPO     = "HOUSE_PRICE_PREDICTION"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json",
    "Content-Type": "application/json",
    "X-GitHub-Api-Version": "2022-11-28",
}

def gh_request(method, url, data=None):
    body = json.dumps(data).encode() if data else None
    req  = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        msg = e.read().decode()
        return json.loads(msg) if msg else {}, e.code

# 1. Descobrir username
user_resp, _ = gh_request("GET", "https://api.github.com/user")
username = user_resp["login"]
print(f"Usuário: {username}")

# 2. Criar repositório (se não existir)
repo_check, status = gh_request("GET", f"https://api.github.com/repos/{username}/{REPO}")
if status == 404:
    create_resp, sc = gh_request("POST", "https://api.github.com/user/repos", {
        "name": REPO,
        "description": "Desafio Data Science — Previsão de Preços de Imóveis em Seattle (XGBoost, R²=0.91)",
        "private": False,
        "auto_init": True,
    })
    print(f"Repositório criado: {create_resp.get('html_url')}")
else:
    print(f"Repositório já existe: {repo_check.get('html_url')}")

# 3. Coletar todos os arquivos para upload
def collect_files(base_dir):
    files = []
    skip_dirs = {"__pycache__", ".git"}
    skip_exts = {".pyc"}
    for root, dirs, filenames in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            if any(fname.endswith(e) for e in skip_exts):
                continue
            full_path = os.path.join(root, fname)
            rel_path  = os.path.relpath(full_path, base_dir).replace("\\", "/")
            files.append((full_path, rel_path))
    return files

files = collect_files(BASE_DIR)
print(f"\nArquivos a enviar: {len(files)}")
for _, rp in files:
    print(f"  {rp}")

# 4. Upload de cada arquivo
API_BASE = f"https://api.github.com/repos/{username}/{REPO}/contents"
success, errors = 0, 0

for full_path, rel_path in files:
    with open(full_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()

    # Verificar se já existe (para obter SHA e atualizar)
    existing, ex_status = gh_request("GET", f"{API_BASE}/{rel_path}")
    sha = existing.get("sha") if ex_status == 200 else None

    payload = {
        "message": f"add {rel_path}",
        "content": content_b64,
    }
    if sha:
        payload["sha"] = sha
        payload["message"] = f"update {rel_path}"

    resp, sc = gh_request("PUT", f"{API_BASE}/{rel_path}", payload)
    if sc in (200, 201):
        print(f"  OK [{sc}] {rel_path}")
        success += 1
    else:
        print(f"  ERRO [{sc}] {rel_path}: {resp.get('message','')}")
        errors += 1

print(f"\n=== Upload concluído: {success} OK, {errors} erros ===")
print(f"Repositório: https://github.com/{username}/{REPO}")
