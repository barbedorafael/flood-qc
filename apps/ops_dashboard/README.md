# Ops Dashboard

Entry point principal da interface operacional em Streamlit.

Uso esperado:

```bash
streamlit run apps/ops_dashboard/app.py
```

O dashboard consome:

- `data/history.sqlite` para cadastro de estacoes e series observadas;
- `data/interim/model_outputs.sqlite` para series MGB;
- `data/interim/accum_*h.tif` para rasters de chuva acumulada;
- `data/legacy/app_layers/rios_mini.geojson` para clique nas minis MGB.
