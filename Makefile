# =====================================================================
# Makefile - Pipeline MLOps Bank Customer Balance Regression
# Reglas reproducibles: install, lint, test, train, mlflow-ui, clean
# =====================================================================

PYTHON ?= python
PIP    ?= pip

.PHONY: help install lint test train mlflow-ui clean all

help:  ## Muestra esta ayuda
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make install     -> Instala dependencias desde requirements.txt"
	@echo "  make lint        -> Corre flake8 sobre src/ y tests/"
	@echo "  make test        -> Ejecuta los tests con pytest"
	@echo "  make train       -> Ejecuta el pipeline completo (data + train + MLflow)"
	@echo "  make mlflow-ui   -> Levanta la UI de MLflow en http://127.0.0.1:5000"
	@echo "  make clean       -> Borra artefactos generados (mlruns, artifacts, cache)"
	@echo "  make all         -> install + lint + test + train"
	@echo ""

install:  ## Instala dependencias
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:  ## Corre flake8 (no falla por warnings menores)
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=110 --extend-ignore=E501,W503,E203 --exclude=__pycache__,.venv,venv --statistics

test:  ## Ejecuta tests con pytest
	$(PYTHON) -m pytest tests/ -v --tb=short

train:  ## Pipeline completo: descarga + preprocesa + entrena + registra en MLflow
	$(PYTHON) -m src.train --config config.yaml

mlflow-ui:  ## Lanza el servidor de la UI de MLflow
	$(PYTHON) -m mlflow ui --host 127.0.0.1 --port 5000

clean:  ## Limpia artefactos generados
	rm -rf mlruns artifacts data/bank_churn.csv .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Limpieza completa."

all: install lint test train  ## Pipeline completo end-to-end
