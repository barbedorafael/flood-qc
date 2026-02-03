from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns to keep from source CSVs
COLUMNS = ["ID", "CODIGO", "LAT", "LON", "ALT", "NOME"]


def load_minimal(path: Path) -> pd.DataFrame:
    """Load a station CSV and return only the minimal columns, cleaned."""
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        usecols=lambda c: c.split(" ")[0] in COLUMNS or c in COLUMNS,  # tolerate trailing notes
    )

    # Standardize column order and casing
    df = df.rename(columns=str.upper)[COLUMNS]

    # Basic cleanup: strip whitespace and drop duplicates by CODIGO
    df["CODIGO"] = df["CODIGO"].astype(str).str.strip()
    df["NOME"] = df["NOME"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["CODIGO"]).reset_index(drop=True)

    # Sort for reproducibility
    df = df.sort_values(by="CODIGO").reset_index(drop=True)
    return df


def write_outputs(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", sep=";", index=False, encoding="utf-8")


def main() -> None:
    nivel = Path(__file__).resolve().parent.parent / "data/spatial/EstacoesNivel.csv"
    pluv = Path(__file__).resolve().parent.parent / "data/spatial/EstacoesPluv.csv"

    out_dir = Path("data")

    nivel_df = load_minimal(nivel)
    pluv_df = load_minimal(pluv)

    write_outputs(nivel_df, "estacoes_nivel", out_dir)
    write_outputs(pluv_df, "estacoes_pluv", out_dir)

    print(f"Salvo: {out_dir / 'estacoes_nivel.csv'}")
    print(f"Salvo: {out_dir / 'estacoes_pluv.csv'}")


if __name__ == "__main__":
    main()
