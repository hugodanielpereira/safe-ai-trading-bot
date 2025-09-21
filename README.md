# AI Bot Starter — Guia Rápido

## Estrutura de dados
```bash
data/
BTCUSDT/
1m/
2020.csv
2021.csv
5m/
2020.csv
models/
gbm_BTCUSDT_1m_2017-2022.pkl
metrics_summary.csv
web/
backtests/
```

## Download de dados
- Garanta que grava por subpasta de timeframe: `data/{SYMBOL}/{INTERVAL}/{YYYY}.csv`.
- Exemplo de função de gravação em `tools/save_prices.py`.

## Executar
```bash
uvicorn app.main:app --reload
# UI em http://localhost:8000/web/
```
## Treinar modelo (UI)
	•	Abra o cartão Treinar modelo.
	•	Escolha SYMBOL, INTERVAL, fonte de CSV (data/SYMBOL/INTERVAL/*.csv), date_from/date_to (janela de treino) e out_path (opcional).
	•	Clique Treinar. Acompanhe percent/tempo.
	•	No fim, o campo de Backtest “model_path” pode ser pré-preenchido.

## Backtest (UI)
	•	Strategy: AI (modelo) ou SMA.
	•	csv_path/glob/dir com dados do período de teste (ex.: 2024).
	•	date_from/date_to: delimitar período de teste.
	•	model_path: ficheiro .pkl treinado antes do período de teste.
	•	Correr backtest e ver equity/metrics.

## CLI (opcional)

Se preferir CLI:
```bash
make retrain-multi SYMBOL=BTCUSDT INTERVAL=1m \
  CSV="data/BTCUSDT/1m/*.csv" FROM=2017-01-01 TO=2023-01-01 \
  OUT=models/gbm_BTCUSDT_1m_2017-2022.pkl
```

## Boas práticas
	•	Evitar leakage: treinar até T, testar depois de T.
	•	Versionar modelos por símbolo/intervalo/janela.
	•	Usar walk-forward para robustez.
---

## Notas finais

- Com o **downloader** a guardar por `{symbol}/{interval}/`, a própria `_load_prices` já impede misturas.  
- Com o **/train_run_async + UI**, passas a **conseguir fazer tudo pela interface**: treinar, apontar `model_path` e backtestar.  
- Se quiseres, no próximo passo posso mandar-te um `app/train.py` de exemplo (scikit-learn + joblib) para teres um trainer Python direto (sem depender do Make), mas como já tens Make targets, o fallback de CLI já te serve hoje.