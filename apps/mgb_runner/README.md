# MGB Runner

Esta pasta guarda apenas os artefatos do MGB:

- `Input/`: inputs locais versionados que serao copiados para `C:/mgb-hora/Input`
- `Output/`: espelho local do `C:/mgb-hora/Output` apos cada execucao
- `MGB_Inercial_PrevRS_FORTRAN.exe`: executavel Windows usado pelo runner

O codigo do runner fica em `src/model/`.

Uso:

```bash
python src/model/run_mgb.py --dry-run
python src/model/run_mgb.py
```

