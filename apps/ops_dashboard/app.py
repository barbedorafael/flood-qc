from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import history_db_path, runs_dir, spatial_dir, timeseries_dir


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit("Streamlit nao encontrado. Instale com: pip install -e .[ui]") from exc

    st.set_page_config(page_title="Hidrologia Operacional", layout="wide")
    st.title("Hidrologia Operacional")
    st.caption("Dashboard inicial para navegacao de runs, QC e relatorios.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Banco historico", history_db_path().as_posix())
    col2.metric("Diretorio de runs", runs_dir().as_posix())
    col3.metric("Camadas espaciais", spatial_dir().as_posix())

    st.subheader("Escopo desta etapa")
    st.markdown(
        """
- Navegacao basica da arquitetura
- Ponto de entrada unico do dashboard
- Preparacao para browser de runs e revisao operacional
- Nenhuma integracao real com APIs nesta fase
        """
    )

    st.subheader("Secoes previstas")
    st.markdown(
        """
- Runs: listar runs automaticos e manuais
- QC: consolidar flags e revisoes pendentes
- Inputs: inspecionar series e assets usados no run
- Outputs: resumir resultados do modelo e comparacoes
- Relatorios: publicar produtos operacionais
        """
    )

    st.subheader("Paths operacionais")
    st.code(
        "\n".join(
            [
                f"history: {history_db_path().as_posix()}",
                f"runs: {runs_dir().as_posix()}",
                f"timeseries: {timeseries_dir().as_posix()}",
                f"spatial: {spatial_dir().as_posix()}",
            ]
        ),
        language="text",
    )


if __name__ == "__main__":
    main()