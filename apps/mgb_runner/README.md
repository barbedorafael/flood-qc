# MGB Runner

Camada dedicada para preparar e futuramente executar o MGB como processo externo em Windows.

Uso esperado:

```bash
python apps/mgb_runner/main.py --run-db data/runs/20260310T120000.sqlite --dry-run
```

Nesta fase o runner apenas monta e exibe o plano de execucao. Nenhum binario real e chamado.