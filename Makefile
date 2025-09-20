# Quick Dev Makefile
PY ?= python
APP ?= app.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
BASE ?= http://localhost:8000
TOKEN ?= algotokenqualquer

.PHONY: venv install run docker-up docker-down start stop status logs python-info

venv:
	$(PY) -m venv .venv

install: venv
	. .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && uvicorn $(APP) --host $(HOST) --port $(PORT) --reload

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
	. .venv/bin/activate && python -c "import sys,ssl;print(sys.version);print(ssl.OPENSSL_VERSION)"