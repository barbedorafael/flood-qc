import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

TEMPO_URL = "https://tempo.inmet.gov.br/"
FRONT_URL = "https://apitempo.inmet.gov.br/estacao/front/"

@dataclass
class Tokens:
    seed: str
    gcap: str

def wait_for_tokens(page, timeout_ms: int = 120000) -> Tokens:
    """
    Captura seed/gcap escutando o request POST para FRONT_URL.
    Você pode navegar/clicar no site; quando a chamada acontecer, os tokens são capturados.
    """
    captured: Dict[str, Optional[Tokens]] = {"tokens": None}

    def on_request(req):
        if req.method == "POST" and req.url == FRONT_URL:
            try:
                data = req.post_data_json
                if isinstance(data, dict) and "seed" in data and "gcap" in data:
                    captured["tokens"] = Tokens(seed=data["seed"], gcap=data["gcap"])
            except Exception:
                pass

    page.on("request", on_request)

    waited = 0
    step = 250
    while waited < timeout_ms and captured["tokens"] is None:
        page.wait_for_timeout(step)
        waited += step

    page.off("request", on_request)

    if captured["tokens"] is None:
        raise PWTimeout("Não capturei seed/gcap. Você clicou para gerar/carregar os dados (o botão da tabela)?")

    return captured["tokens"]

def post_front(context, tokens: Tokens, estacao: str, data_ini: str, data_fim: str) -> List[Dict[str, Any]]:
    payload = {
        "data_inicio": data_ini,  # YYYY-MM-DD
        "data_fim": data_fim,     # YYYY-MM-DD
        "estacao": estacao,
        "seed": tokens.seed,
        "gcap": tokens.gcap,
    }

    resp = context.request.post(
        FRONT_URL,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://tempo.inmet.gov.br",
            "Referer": "https://tempo.inmet.gov.br/",
        },
        data=json.dumps(payload),
        timeout=60_000,
    )

    if resp.status != 200:
        # normalmente aqui é token expirado / sessão expirada
        raise RuntimeError(f"HTTP {resp.status}: {resp.text()[:200]}")

    return resp.json()

def main():
    # ======= configure aqui =======
    estacoes = ["A836", "A001", "A002"]  # coloque as que você quiser
    data_ini = "2026-02-18"
    data_fim = "2026-02-25"
    # =============================

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # abre janela pra você resolver captcha
        context = browser.new_context()
        page = context.new_page()

        page.goto(TEMPO_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)
        page.goto("https://tempo.inmet.gov.br/TabelaEstacoes/A836", wait_until="domcontentloaded")

        print("\n1) Se aparecer captcha/checagem, resolva no navegador.")
        print("2) Depois, clique para GERAR/CARREGAR a tabela (só 1 vez, qualquer estação/período).")
        input("3) Quando a tabela carregar (ou começar a carregar), volte aqui e aperte Enter... ")

        # captura tokens enquanto você interage
        tokens = wait_for_tokens(page, timeout_ms=120000)
        print("✅ Tokens capturados. Vou começar o lote.")

        ok = 0
        for estacao in estacoes:
            try:
                data = post_front(context, tokens, estacao, data_ini, data_fim)
                out = f"inmet_{estacao}_{data_ini}_to_{data_fim}.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                print(f"✅ {estacao}: {len(data)} registros -> {out}")
                ok += 1
            except Exception as e:
                print(f"❌ Falhou em {estacao}: {e}")
                print("Provável expiração do token/sessão. Reabra/regenere e rode de novo a partir desta estação.")
                break

        print(f"\nConcluído: {ok}/{len(estacoes)} estações.")
        browser.close()

if __name__ == "__main__":
    main()