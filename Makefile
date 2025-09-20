# Quick Dev Makefile (versão limpa e corrigida)

PY ?= python
APP ?= app.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
BASE ?= http://localhost:8000
TOKEN ?= ALGOTOKEN

# ==== Parametrização de dados/modelo ====
SYMBOL ?= BTCUSDT
SYMBOLS ?= BTCUSDT ETHUSDT BNBUSDT
INTERVAL ?= 1m
START ?= 2024-09-20
END ?= 2025-09-20
PAGES ?= 30
KL ?= data/klines.csv
FEAT ?= data/features.csv
MODEL ?= models/gbm.pkl
FEATURE_H ?= 10
Q ?= 0.62

VENV = . .venv/bin/activate &&
PYENV = $(VENV) $(PY)

.PHONY: venv install run docker-up docker-down start stop status logs python-info \
        fetch-testnet features train backtest data-range data-year fetch-pages concat \
        retrain retrain-years model-metrics ticker trades orders balance

venv:
	$(PY) -m venv .venv

install: venv
	$(PYENV) -m pip install -r requirements.txt

run:
	$(PYENV) -m uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

start:
	curl -s -X POST $(BASE)/start -H "X-API-Token: $(TOKEN)"

stop:
	curl -s -X POST $(BASE)/stop -H "X-API-Token: $(TOKEN)"

status:
	curl -s $(BASE)/status

logs:
	curl -s $(BASE)/logs

python-info:
	$(PYENV) -c "import sys,ssl;print(sys.version);print(ssl.OPENSSL_VERSION)"

# === Pipelines ===

features:
	$(PYENV) -m scripts.make_features --klines $(KL) --outfile $(FEAT) \
		--horizon $(FEATURE_H) --use_quantiles --q $(Q)

fetch-testnet:
	$(PYENV) -m scripts.fetch_klines --symbol $(SYMBOL) --interval 1m \
		--limit 1000 --outfile $(KL) --testnet

train:
	$(PYENV) -m scripts.train_lightgbm --features $(FEAT) --model_out $(MODEL)

backtest:
	$(PYENV) -m scripts.backtest_vectorized --features $(FEAT) --prices_csv $(KL)

# === Fetch MT5-style ===

data-range:
	mkdir -p data
	$(PYENV) -m scripts.fetch_klines --symbol $(SYMBOL) --interval $(INTERVAL) \
		--start $(START) --end $(END) --source mainnet --outfile $(KL)

data-year:
	mkdir -p data
	$(PYENV) -m scripts.fetch_klines --symbol $(SYMBOL) --interval $(INTERVAL) \
		--start $(YEAR)-01-01 --end $(YEAR)-12-31 --source mainnet \
		--outfile data/klines_$(INTERVAL)_$(YEAR).csv

fetch-pages:
	mkdir -p data
	$(PYENV) -m scripts.fetch_klines --symbol $(SYMBOL) --interval $(INTERVAL) \
		--pages $(PAGES) --source mainnet --outfile $(KL)

concat:
	mkdir -p data
	$(PYENV) -m scripts.concat_csv --out $(KL) $(FILES)

retrain: data-range features train

# Busca e concatena para TODOS os símbolos/anos do config
fetch-multi:
	$(PYENV) -m scripts.fetch_multi --config config/symbols.yaml

fetch-multi-quick:
	$(PYENV) -m scripts.fetch_multi --config config/symbols.yaml --only $(SYMBOLS) --years $(YEARS) --verbose --sleep_ms 120

# Treina para TODOS os símbolos do config e gera comparação
train-multi:
	$(PYENV) -m scripts.train_multi --config config/symbols.yaml --out_csv models/metrics_summary.csv

# Pipeline completo: fetch -> train (multi-símbolo)
retrain-multi: fetch-multi train-multi

retrain-years:
	@set -e; \
	mkdir -p data; \
	for Y in $(YEARS); do \
		echo ">> Fetch ano $$Y"; \
		$(PYENV) -m scripts.fetch_klines --symbol $(SYMBOL) --interval $(INTERVAL) \
			--start $$Y-01-01 --end $$Y-12-31 --source mainnet \
			--outfile data/klines_$(INTERVAL)_$$Y.csv; \
	done; \
	echo ">> Concat"; \
	FILELIST=`ls -1 data/klines_$(INTERVAL)_*.csv | tr '\n' ' '`; \
	$(PYENV) -m scripts.concat_csv --out $(KL) $$FILELIST; \
	echo ">> Features"; \
	$(PYENV) -m scripts.make_features --klines $(KL) --outfile $(FEAT) \
		--horizon $(FEATURE_H) --use_quantiles --q $(Q); \
	echo ">> Train"; \
	$(PYENV) -m scripts.train_lightgbm --features $(FEAT) --model_out $(MODEL)

ticker:
	curl -s "$(BASE)/ticker"

trades:
	curl -s "$(BASE)/trades?limit=10" -H "X-API-Token: $(TOKEN)"

orders:
	curl -s "$(BASE)/orders?limit=10" -H "X-API-Token: $(TOKEN)"

balance:
	curl -s "$(BASE)/balance" -H "X-API-Token: $(TOKEN)"

model-metrics:
	$(PYENV) -m scripts.model_metrics