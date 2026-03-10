# QGIS Project

O QGIS e um cliente complementar desta arquitetura. Ele consome os artefatos gerados pelo pipeline, mas nao orquestra o sistema.

Convencoes iniciais:

- vetores preferencialmente em GeoPackage;
- rasters preferencialmente em GeoTIFF;
- referencias a arquivos mantidas no `history.sqlite` e nos `run.sqlite`;
- assets aprovados localizados em `data/spatial/` ou referenciados a partir de um run.

Itens futuros:

- template de projeto QGIS por run;
- estilos padrao para camadas de nivel, vazao e chuva;
- links diretos entre banco do run e camadas publicadas.