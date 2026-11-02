PY ?= .venv/bin/python
RUN = gtimeout 2700 nice -n 10 $(PY)

.PHONY: setup data train eval bench demo test lint plots smoke

setup:
	uv venv -q -p 3.12 && uv pip install -q -e ".[bench,dev]"

data:
	$(PY) scripts/prepare_data.py

train: eval

eval:
	$(RUN) scripts/leakage_experiment.py

bench:
	$(RUN) scripts/bench_asof.py

plots:
	$(PY) scripts/make_tables.py

demo:
	$(PY) -m pitstore.cli demo

smoke:
	$(PY) -m pytest -q tests/test_store.py -k leakage

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m mypy
